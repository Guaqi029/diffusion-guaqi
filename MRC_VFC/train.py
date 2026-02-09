# train the encoder
import os
import time
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import ProbabilityLoss, BatchLoss, ChannelLoss, GaussianPriorLoss
import torch.distributed as dist
from utils import ramps, epochVal
from utils.metrics import compute_avg_metrics


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def _write_local_log(log_f, msg):
    if log_f is None:
        return
    log_f.write(msg + "\n")
    log_f.flush()


def _get_state_dict(model):
    if model is None:
        return None
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _forward_lite_eval(lite_vae, lite_classifier, img):
    mu, logvar, z, _ = lite_vae(img)
    if lite_classifier is None:
        raise ValueError("lite_classifier is required for lite eval")
    logits = lite_classifier(z)
    return logits


def trainEncoder(
    model,
    ema_model,
    dataloader,
    optimizer,
    logger,
    args,
    aux_vae=None,
    lite_vae=None,
    lite_classifier=None,
    kd_feat_proj=None,
    log_f=None,
):
    probability_loss_func = ProbabilityLoss()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)
    channel_sim_loss_func = ChannelLoss(args.batch_size, args.world_size)
    gaussian_prior_loss_func = GaussianPriorLoss(
        num_classes=args.num_classes,
        ema_momentum=args.gaussian_ema_momentum,
        var_floor=args.gaussian_var_floor,
        mode=args.gaussian_prior_mode,
        fixed_var_value=args.gaussian_fixed_var_value,
    )
    classification_loss_func = nn.CrossEntropyLoss()
    if args.aux_vae_recon_type == "mse":
        recon_loss_func = nn.MSELoss()
    else:
        recon_loss_func = nn.L1Loss()
    if args.lite_vae_recon_type == "mse":
        lite_recon_loss_func = nn.MSELoss()
    else:
        lite_recon_loss_func = nn.L1Loss()
    start = time.time()
    cur_iters = 0
    model.train()
    train_loader, val_loader, test_loader = dataloader
    cur_lr = args.lr
    best_test = None
    best_val = None
    best_test_epoch = -1
    best_val_epoch = -1
    def _epoch_val_lite(lite_vae, lite_classifier, data_loader):
        if lite_vae is None or lite_classifier is None:
            return None
        training_vae = lite_vae.training
        training_cls = lite_classifier.training
        lite_vae.eval()
        lite_classifier.eval()
        groundTruth = torch.Tensor().cuda()
        activations = torch.Tensor().cuda()
        with torch.no_grad():
            for image, label in data_loader:
                image, label = image.cuda(), label.cuda()
                logits = _forward_lite_eval(lite_vae, lite_classifier, image)
                logits = F.softmax(logits, dim=1)
                groundTruth = torch.cat((groundTruth, label))
                activations = torch.cat((activations, logits))
        acc, f1, auc, bac, sens, spec = compute_avg_metrics(groundTruth, activations)
        lite_vae.train(training_vae)
        lite_classifier.train(training_cls)
        return acc, f1, auc, bac, sens, spec

    if args.lite_eval_only:
        if args.rank == 0 and args.lite_eval_enable and args.lite_eval_use_classifier and lite_vae is not None and lite_classifier is not None:
            lite_val = _epoch_val_lite(lite_vae, lite_classifier, val_loader)
            lite_test = _epoch_val_lite(lite_vae, lite_classifier, test_loader)
            if lite_val is not None and lite_test is not None:
                lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec = lite_val
                ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec = lite_test
                msg_val = "lite_val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                    lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec
                )
                msg_test = "lite_test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                    ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec
                )
                print(msg_val)
                print(msg_test)
                _write_local_log(log_f, msg_val)
                _write_local_log(log_f, msg_test)
                if logger is not None:
                    logger.log({'lite_validation': {'Accuracy': lval_acc,
                                                    'F1 score': lval_f1,
                                                    'AUC': lval_auc,
                                                    'Balanced Accuracy': lval_bac,
                                                    'Sensitivity': lval_sens,
                                                    'Specificity': lval_spec}})
                    logger.log({'lite_test': {'Accuracy': ltest_acc,
                                              'F1 score': ltest_f1,
                                              'AUC': ltest_auc,
                                              'Balanced Accuracy': ltest_bac,
                                              'Sensitivity': ltest_sens,
                                              'Specificity': ltest_spec}})
        return

    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, ((img, ema_img), label) in enumerate(train_loader):
            img, ema_img, label = img.cuda(non_blocking=True), ema_img.cuda(non_blocking=True), label.cuda(
                non_blocking=True)

            kd_only = args.kd_enable and args.kd_only
            if args.kd_enable and args.kd_freeze_teacher:
                with torch.no_grad():
                    activations, outputs = model(img)
            else:
                activations, outputs = model(img)

            if not kd_only:
                with torch.no_grad():
                    ema_activations, ema_output = ema_model(ema_img)
            else:
                ema_activations, ema_output = None, None

            # classification loss
            classification_loss = classification_loss_func(outputs, label)

            # probability distribution loss
            if not kd_only:
                probability_loss = torch.sum(probability_loss_func(outputs, ema_output)) / args.batch_size
            else:
                probability_loss = torch.tensor(0.0, device=activations.device)
            
            # batch loss
            if not kd_only:
                batch_sim_loss = torch.sum(batch_sim_loss_func(activations, ema_activations))
            else:
                batch_sim_loss = torch.tensor(0.0, device=activations.device)

            # channel loss
            if not kd_only:
                channel_sim_loss = torch.sum(channel_sim_loss_func(activations, ema_activations))
            else:
                channel_sim_loss = torch.tensor(0.0, device=activations.device)

            base_loss = classification_loss * args.classification_loss_weight
            if not kd_only and epoch > 20:
                base_loss = base_loss + probability_loss * args.probability_loss_weight + batch_sim_loss * args.batch_loss_weight + channel_sim_loss * args.channel_loss_weight
            if not kd_only and epoch >= args.gaussian_prior_start_epoch and args.gaussian_prior_weight > 0:
                gaussian_prior_loss = gaussian_prior_loss_func(activations, label)
                base_loss = base_loss + gaussian_prior_loss * args.gaussian_prior_weight
            else:
                gaussian_prior_loss = torch.tensor(0.0, device=activations.device)

            if aux_vae is not None and args.use_aux_vae and epoch >= args.aux_vae_start_epoch:
                aux_in = img if args.aux_vae_input == "image" else activations
                mu, logvar, recon = aux_vae(aux_in)
                recon_loss = recon_loss_func(recon, img)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                base_loss = base_loss + recon_loss * args.aux_vae_recon_weight + kl_loss * args.aux_vae_kl_weight
            else:
                recon_loss = torch.tensor(0.0, device=activations.device)
                kl_loss = torch.tensor(0.0, device=activations.device)

            # KD + LiteVAE student losses
            kd_logit_loss = torch.tensor(0.0, device=activations.device)
            kd_feat_loss = torch.tensor(0.0, device=activations.device)
            lite_recon_loss = torch.tensor(0.0, device=activations.device)
            lite_kl_loss = torch.tensor(0.0, device=activations.device)
            lite_ce_loss = torch.tensor(0.0, device=activations.device)
            lite_acc = torch.tensor(0.0, device=activations.device)

            if args.kd_enable and lite_vae is not None and lite_classifier is not None:
                lite_mu, lite_logvar, lite_z, lite_recon = lite_vae(img)
                lite_logits = lite_classifier(lite_z)

                if args.kd_logit_weight > 0:
                    t = args.kd_temperature
                    teacher_logits = outputs.detach()
                    kd_logit_loss = F.kl_div(
                        F.log_softmax(lite_logits / t, dim=1),
                        F.softmax(teacher_logits / t, dim=1),
                        reduction="batchmean",
                    ) * (t * t)

                if args.kd_feat_weight > 0:
                    feat_s = lite_z
                    feat_t = activations.detach()
                    if kd_feat_proj is not None:
                        feat_s = kd_feat_proj(feat_s)
                    elif feat_s.size(1) != feat_t.size(1):
                        raise ValueError("kd_feat_project is False but feature dims do not match")
                    if args.kd_feat_norm == "l2":
                        feat_s = F.normalize(feat_s, dim=1)
                        feat_t = F.normalize(feat_t, dim=1)
                    kd_feat_loss = F.mse_loss(feat_s, feat_t)

                if args.lite_vae_recon_weight > 0:
                    lite_recon_loss = lite_recon_loss_func(lite_recon, img)
                if args.lite_vae_kl_weight > 0:
                    lite_kl_loss = -0.5 * torch.mean(1 + lite_logvar - lite_mu.pow(2) - lite_logvar.exp())
                if args.lite_student_ce_weight > 0:
                    lite_ce_loss = classification_loss_func(lite_logits, label)

                with torch.no_grad():
                    lite_pred = lite_logits.argmax(1)
                    lite_acc = (lite_pred == label).float().mean()

            loss = torch.tensor(0.0, device=activations.device)
            if not kd_only:
                loss = loss + base_loss
            loss = loss + kd_logit_loss * args.kd_logit_weight
            loss = loss + kd_feat_loss * args.kd_feat_weight
            loss = loss + lite_recon_loss * args.lite_vae_recon_weight
            loss = loss + lite_kl_loss * args.lite_vae_kl_weight
            loss = loss + lite_ce_loss * args.lite_student_ce_weight

            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update ema model
            if not kd_only:
                update_ema_variables(model, ema_model, args.ema_decay, cur_iters)

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))


            cur_iters += 1
            if args.rank == 0:
                if cur_iters % 500 == 1 and logger is not None:
                    logger.log({'Strong augmentation': [wandb.Image(item) for item in img.permute(0,2,3,1).detach().cpu().numpy()[:5]]})
                    logger.log({'Weak augmentation': [wandb.Image(item) for item in ema_img.permute(0,2,3,1).detach().cpu().numpy()[:5]]})
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    val_acc, val_f1, val_auc, val_bac, val_sens, val_spec = epochVal(model, val_loader)
                    test_acc, test_f1, test_auc, test_bac, test_sens, test_spec = epochVal(model, test_loader)
                    lite_metrics = None
                    if args.lite_eval_enable and args.lite_eval_use_classifier and lite_vae is not None and lite_classifier is not None:
                        lite_val = _epoch_val_lite(lite_vae, lite_classifier, val_loader)
                        lite_test = _epoch_val_lite(lite_vae, lite_classifier, test_loader)
                        lite_metrics = (lite_val, lite_test)
                    if logger is not None:
                        logger.log({'training': {'total loss': rank0_loss,
                                                 'probability loss': probability_loss.item(),
                                                 'batch similarity loss': batch_sim_loss.item(),
                                                 'channel similarity loss': channel_sim_loss.item(),
                                                'classification loss': classification_loss.item(),
                                                 'gaussian prior loss': gaussian_prior_loss.item(),
                                                 'aux recon loss': recon_loss.item(),
                                                 'aux kl loss': kl_loss.item(),
                                                 'kd logit loss': kd_logit_loss.item(),
                                                 'kd feat loss': kd_feat_loss.item(),
                                                 'lite recon loss': lite_recon_loss.item(),
                                                 'lite kl loss': lite_kl_loss.item(),
                                                 'lite ce loss': lite_ce_loss.item(),
                                                 'lite acc (batch)': lite_acc.item()}})
                        logger.log({'test': {'Accuracy': test_acc,
                                             'F1 score': test_f1,
                                             'AUC': test_auc,
                                             'Balanced Accuracy': test_bac,
                                             'Sensitivity': test_sens,
                                             'Specificity': test_spec},
                                    'validation': {'Accuracy': val_acc,
                                                   'F1 score': val_f1,
                                                   'AUC': val_auc,
                                                   'Balanced Accuracy': val_bac,
                                                   'Sensitivity': val_sens,
                                                   'Specificity': val_spec}})
                        if lite_metrics is not None:
                            (lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec), (ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec) = lite_metrics
                            logger.log({'lite_test': {'Accuracy': ltest_acc,
                                                      'F1 score': ltest_f1,
                                                      'AUC': ltest_auc,
                                                      'Balanced Accuracy': ltest_bac,
                                                      'Sensitivity': ltest_sens,
                                                      'Specificity': ltest_spec},
                                        'lite_validation': {'Accuracy': lval_acc,
                                                            'F1 score': lval_f1,
                                                            'AUC': lval_auc,
                                                            'Balanced Accuracy': lval_bac,
                                                            'Sensitivity': lval_sens,
                                                            'Specificity': lval_spec}})
                    _write_local_log(
                        log_f,
                        "epoch={:d} iter={:d} train: total={:.6f}, prob={:.6f}, batch={:.6f}, channel={:.6f}, cls={:.6f}, "
                        "gauss={:.6f}, aux_recon={:.6f}, aux_kl={:.6f}, kd_logit={:.6f}, kd_feat={:.6f}, "
                        "lite_recon={:.6f}, lite_kl={:.6f}, lite_ce={:.6f}, lite_acc={:.6f}".format(
                            epoch + 1,
                            i + 1,
                            rank0_loss,
                            probability_loss.item(),
                            batch_sim_loss.item(),
                            channel_sim_loss.item(),
                            classification_loss.item(),
                            gaussian_prior_loss.item(),
                            recon_loss.item(),
                            kl_loss.item(),
                            kd_logit_loss.item(),
                            kd_feat_loss.item(),
                            lite_recon_loss.item(),
                            lite_kl_loss.item(),
                            lite_ce_loss.item(),
                            lite_acc.item(),
                        ),
                    )
                    _write_local_log(
                        log_f,
                        "epoch={:d} test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                            epoch + 1,
                            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec
                        ),
                    )
                    _write_local_log(
                        log_f,
                        "epoch={:d} val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                            epoch + 1,
                            val_acc, val_f1, val_auc, val_bac, val_sens, val_spec
                        ),
                    )
                    if lite_metrics is not None:
                        (lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec), (ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec) = lite_metrics
                        _write_local_log(
                            log_f,
                            "epoch={:d} lite_test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec
                            ),
                        )
                        _write_local_log(
                            log_f,
                            "epoch={:d} lite_val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec
                            ),
                        )
                    if best_test is None or test_acc > best_test["acc"]:
                        best_test = {
                            "acc": test_acc,
                            "f1": test_f1,
                            "auc": test_auc,
                            "bac": test_bac,
                            "sens": test_sens,
                            "spec": test_spec,
                        }
                        best_test_epoch = epoch + 1
                    if best_val is None or val_acc > best_val["acc"]:
                        best_val = {
                            "acc": val_acc,
                            "f1": val_f1,
                            "auc": val_auc,
                            "bac": val_bac,
                            "sens": val_sens,
                            "spec": val_spec,
                        }
                        best_val_epoch = epoch + 1
                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True)

        if args.rank == 0:
            saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_.pth'.format(epoch + 1))
            state_dict = _get_state_dict(model)
            torch.save(state_dict, saveModelPath)

            if args.kd_enable and args.kd_save_lite and lite_vae is not None:
                if args.kd_save_every_epoch:
                    lite_path = os.path.join(args.checkpoints, "litevae_epoch_{:d}_.pth".format(epoch + 1))
                    torch.save(_get_state_dict(lite_vae), lite_path)
                    if lite_classifier is not None:
                        cls_path = os.path.join(args.checkpoints, "lite_classifier_epoch_{:d}_.pth".format(epoch + 1))
                        torch.save(_get_state_dict(lite_classifier), cls_path)
                    if kd_feat_proj is not None:
                        proj_path = os.path.join(args.checkpoints, "kd_feat_proj_epoch_{:d}_.pth".format(epoch + 1))
                        torch.save(_get_state_dict(kd_feat_proj), proj_path)
                if args.kd_save_latest:
                    lite_path = os.path.join(args.checkpoints, "litevae_latest.pth")
                    torch.save(_get_state_dict(lite_vae), lite_path)
                    if lite_classifier is not None:
                        cls_path = os.path.join(args.checkpoints, "lite_classifier_latest.pth")
                        torch.save(_get_state_dict(lite_classifier), cls_path)
                    if kd_feat_proj is not None:
                        proj_path = os.path.join(args.checkpoints, "kd_feat_proj_latest.pth")
                        torch.save(_get_state_dict(kd_feat_proj), proj_path)

    if args.rank == 0 and log_f is not None:
        if best_test is not None:
            _write_local_log(
                log_f,
                "best_test(epoch={}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                    best_test_epoch,
                    best_test["acc"],
                    best_test["f1"],
                    best_test["auc"],
                    best_test["bac"],
                    best_test["sens"],
                    best_test["spec"],
                ),
            )
        if best_val is not None:
            _write_local_log(
                log_f,
                "best_val(epoch={}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                    best_val_epoch,
                    best_val["acc"],
                    best_val["f1"],
                    best_val["auc"],
                    best_val["bac"],
                    best_val["sens"],
                    best_val["spec"],
                ),
            )


        
