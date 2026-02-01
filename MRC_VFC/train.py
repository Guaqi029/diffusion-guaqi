# train the encoder
import os
import time
import torch
import wandb
import torch.nn as nn
from utils.loss import ProbabilityLoss, BatchLoss, ChannelLoss, GaussianPriorLoss
import torch.distributed as dist
from utils import ramps, epochVal


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


def trainEncoder(model, ema_model, dataloader, optimizer, logger, args, aux_vae=None, log_f=None):
    probability_loss_func = ProbabilityLoss()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)
    channel_sim_loss_func = ChannelLoss(args.batch_size, args.world_size)
    gaussian_prior_loss_func = GaussianPriorLoss(
        num_classes=args.num_classes,
        ema_momentum=args.gaussian_ema_momentum,
        var_floor=args.gaussian_var_floor,
    )
    classification_loss_func = nn.CrossEntropyLoss()
    if args.aux_vae_recon_type == "mse":
        recon_loss_func = nn.MSELoss()
    else:
        recon_loss_func = nn.L1Loss()
    start = time.time()
    cur_iters = 0
    model.train()
    train_loader, val_loader, test_loader = dataloader
    cur_lr = args.lr
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, ((img, ema_img), label) in enumerate(train_loader):
            img, ema_img, label = img.cuda(non_blocking=True), ema_img.cuda(non_blocking=True), label.cuda(
                non_blocking=True)

            activations, outputs = model(img)
            with torch.no_grad():
                ema_activations, ema_output = ema_model(ema_img)

            # classification loss
            classification_loss = classification_loss_func(outputs, label)

            # probability distribution loss
            probability_loss = torch.sum(probability_loss_func(outputs, ema_output)) / args.batch_size
            
            # batch loss
            batch_sim_loss = torch.sum(batch_sim_loss_func(activations, ema_activations))

            # channel loss
            channel_sim_loss = torch.sum(channel_sim_loss_func(activations, ema_activations))

            loss = classification_loss * args.classification_loss_weight
            if epoch > 20:
                loss = loss + probability_loss * args.probability_loss_weight + batch_sim_loss * args.batch_loss_weight + channel_sim_loss * args.channel_loss_weight
            if epoch >= args.gaussian_prior_start_epoch and args.gaussian_prior_weight > 0:
                gaussian_prior_loss = gaussian_prior_loss_func(activations, label)
                loss = loss + gaussian_prior_loss * args.gaussian_prior_weight
            else:
                gaussian_prior_loss = torch.tensor(0.0, device=activations.device)

            if aux_vae is not None and args.use_aux_vae and epoch >= args.aux_vae_start_epoch:
                mu, logvar, recon = aux_vae(activations)
                recon_loss = recon_loss_func(recon, img)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + recon_loss * args.aux_vae_recon_weight + kl_loss * args.aux_vae_kl_weight
            else:
                recon_loss = torch.tensor(0.0, device=activations.device)
                kl_loss = torch.tensor(0.0, device=activations.device)

            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update ema model
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
                    if logger is not None:
                        logger.log({'training': {'total loss': rank0_loss,
                                                 'probability loss': probability_loss.item(),
                                                 'batch similarity loss': batch_sim_loss.item(),
                                                 'channel similarity loss': channel_sim_loss.item(),
                                                'classification loss': classification_loss.item(),
                                                 'gaussian prior loss': gaussian_prior_loss.item(),
                                                 'aux recon loss': recon_loss.item(),
                                                 'aux kl loss': kl_loss.item()}})
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
                    _write_local_log(
                        log_f,
                        "train: total={:.6f}, prob={:.6f}, batch={:.6f}, channel={:.6f}, cls={:.6f}, "
                        "gauss={:.6f}, aux_recon={:.6f}, aux_kl={:.6f}".format(
                            rank0_loss,
                            probability_loss.item(),
                            batch_sim_loss.item(),
                            channel_sim_loss.item(),
                            classification_loss.item(),
                            gaussian_prior_loss.item(),
                            recon_loss.item(),
                            kl_loss.item(),
                        ),
                    )
                    _write_local_log(
                        log_f,
                        "test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec
                        ),
                    )
                    _write_local_log(
                        log_f,
                        "val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                            val_acc, val_f1, val_auc, val_bac, val_sens, val_spec
                        ),
                    )
                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True)

        if args.rank == 0:
            saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_.pth'.format(epoch + 1))
            if dist.is_available() and dist.is_initialized():
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, saveModelPath)


        
