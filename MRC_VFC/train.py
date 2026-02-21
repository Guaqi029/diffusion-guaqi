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


def _save_gaussian_prior_stats(path, gaussian_prior_loss_func):
    if gaussian_prior_loss_func is None:
        return False
    means = gaussian_prior_loss_func.means
    vars_ = gaussian_prior_loss_func.vars
    if means is None or vars_ is None:
        return False
    payload = {
        "means": means.detach().cpu(),
        "vars": vars_.detach().cpu(),
        "num_classes": int(gaussian_prior_loss_func.num_classes),
        "var_floor": float(gaussian_prior_loss_func.var_floor),
        "mode": str(gaussian_prior_loss_func.mode),
    }
    torch.save(payload, path)
    return True


def _select_lite_feature(mu, z, mode="mu"):
    if mode == "z":
        return z
    if mode == "mu":
        return mu
    raise ValueError(f"Unsupported lite feature mode: {mode}")


def _forward_lite_eval(lite_vae, lite_classifier, img, feature_mode="mu"):
    mu, logvar, z, _ = _forward_lite_model(lite_vae, img, need_recon=False)
    if lite_classifier is None:
        raise ValueError("lite_classifier is required for lite eval")
    feat = _select_lite_feature(mu, z, feature_mode)
    logits = lite_classifier(feat)
    return logits


def _forward_lite_model(lite_vae, img, need_recon=False):
    if need_recon:
        mu, logvar, z, recon = lite_vae(img)
        return mu, logvar, z, recon
    if hasattr(lite_vae, "encode"):
        mu, logvar, z = lite_vae.encode(img)
        return mu, logvar, z, None
    mu, logvar, z, recon = lite_vae(img)
    return mu, logvar, z, recon


def _get_classifier(model):
    if hasattr(model, "module"):
        return model.module.classifier
    return model.classifier


def _get_encoder(model):
    if hasattr(model, "module"):
        return model.module.encoder
    return model.encoder


def _compute_mix_alpha(epoch, start_epoch, end_epoch, alpha_start, alpha_end, schedule):
    if epoch <= start_epoch:
        return alpha_start
    if epoch >= end_epoch:
        return alpha_end
    t = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    if schedule == "cosine":
        import math
        return alpha_start + 0.5 * (1 - math.cos(t * math.pi)) * (alpha_end - alpha_start)
    return alpha_start + t * (alpha_end - alpha_start)


def _batch_gram(features, norm="l2", center=False):
    if center:
        features = features - features.mean(dim=0, keepdim=True)
    if norm == "l2":
        features = F.normalize(features, dim=1)
    gram = torch.matmul(features, features.t())
    return gram / max(1, features.size(1))


def _linear_cka_similarity(x, y, center=True, eps=1e-8):
    if center:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
    xty = torch.matmul(x.t(), y)
    xtx = torch.matmul(x.t(), x)
    yty = torch.matmul(y.t(), y)
    hsic = torch.norm(xty, p="fro").pow(2)
    norm_x = torch.norm(xtx, p="fro")
    norm_y = torch.norm(yty, p="fro")
    return hsic / (norm_x * norm_y + float(eps))


def _build_class_weights(
    labels,
    num_classes,
    power=1.0,
    min_weight=0.0,
    max_weight=-1.0,
    eps=1e-6,
):
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=num_classes).float()
    weights = torch.pow(counts + float(eps), -float(power))
    weights = weights / weights.mean().clamp_min(float(eps))
    if float(min_weight) > 0:
        weights = torch.clamp(weights, min=float(min_weight))
    if float(max_weight) > 0:
        weights = torch.clamp(weights, max=float(max_weight))
    weights = weights / weights.mean().clamp_min(float(eps))
    return weights, counts


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
    lite_vae_teacher=None,
    lite_classifier_teacher=None,
    vavae_teacher=None,
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
    lite_gaussian_prior_loss_func = None
    if getattr(args, "save_stage1_lite_gaussian_stats", False):
        lite_gaussian_prior_loss_func = GaussianPriorLoss(
            num_classes=args.num_classes,
            ema_momentum=args.gaussian_ema_momentum,
            var_floor=float(getattr(args, "stage1_lite_gaussian_var_floor", args.gaussian_var_floor)),
            mode="nll",
            fixed_var_value=args.gaussian_fixed_var_value,
        )
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
    if model is not None:
        model.train()
    train_loader, val_loader, test_loader = dataloader
    class_weights = None
    class_counts = None
    if bool(getattr(args, "use_class_weight", False)):
        labels = None
        if hasattr(train_loader.dataset, "get_labels"):
            labels = train_loader.dataset.get_labels()
        elif hasattr(train_loader.dataset, "labels"):
            labels = train_loader.dataset.labels
        if labels is not None:
            class_weights, class_counts = _build_class_weights(
                labels,
                args.num_classes,
                power=float(getattr(args, "class_weight_power", 1.0)),
                min_weight=float(getattr(args, "class_weight_min", 0.0)),
                max_weight=float(getattr(args, "class_weight_max", -1.0)),
                eps=float(getattr(args, "class_weight_eps", 1e-6)),
            )
            class_weights = class_weights.to(args.device)
    classification_loss_func = nn.CrossEntropyLoss(weight=class_weights)
    lite_feature_mode = getattr(args, "lite_student_feature_mode", "mu")
    mix_feature_mode = getattr(args, "mix_lite_feature_mode", lite_feature_mode)
    show_teacher_metrics = bool(getattr(args, "show_teacher_metrics", False))
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    need_lite_recon_forward = float(getattr(args, "lite_vae_recon_weight", 0.0)) > 0.0
    cur_lr = args.lr
    best_test = None
    best_val = None
    best_test_epoch = -1
    best_val_epoch = -1
    kd_feat_start_epoch = int(getattr(args, "kd_feat_start_epoch", 0))
    kd_struct_start_epoch = int(getattr(args, "kd_struct_start_epoch", 0))
    kd_struct_type = str(getattr(args, "kd_struct_type", "gram")).lower()
    kd_teacher_source = str(getattr(args, "kd_teacher_source", "resnet")).lower()
    kd_lite_teacher_use_weak_aug = bool(getattr(args, "kd_lite_teacher_use_weak_aug", True))
    kd_lite_teacher_use_ema = bool(getattr(args, "kd_lite_teacher_use_ema", True))
    kd_lite_teacher_ema_decay = float(getattr(args, "kd_lite_teacher_ema_decay", args.ema_decay))
    kd_vavae_teacher_use_weak_aug = bool(getattr(args, "kd_vavae_teacher_use_weak_aug", True))
    if kd_teacher_source not in ("resnet", "lite", "vavae"):
        raise ValueError("kd_teacher_source must be one of: resnet | lite | vavae")
    if kd_struct_type not in ("gram", "cka"):
        raise ValueError("kd_struct_type must be one of: gram | cka")
    if args.rank == 0:
        if class_weights is not None and class_counts is not None:
            _write_local_log(
                log_f,
                "class_weight enabled: counts={}, weights={}".format(
                    class_counts.tolist(),
                    [round(float(x), 6) for x in class_weights.detach().cpu().tolist()],
                ),
            )
        else:
            _write_local_log(log_f, "class_weight disabled")
        _write_local_log(
            log_f,
            f"kd schedule: feat_start={kd_feat_start_epoch}, struct_start={kd_struct_start_epoch}, struct_type={kd_struct_type}",
        )
        _write_local_log(
            log_f,
            (
                f"kd_teacher_source={kd_teacher_source}, "
                f"kd_lite_teacher_use_weak_aug={kd_lite_teacher_use_weak_aug}, "
                f"kd_lite_teacher_use_ema={kd_lite_teacher_use_ema}, "
                f"kd_vavae_teacher_use_weak_aug={kd_vavae_teacher_use_weak_aug}"
            ),
        )
        _write_local_log(log_f, f"lite_feature_mode={lite_feature_mode}, mix_feature_mode={mix_feature_mode}")
        _write_local_log(log_f, f"show_teacher_metrics={show_teacher_metrics}")
        _write_local_log(log_f, f"grad_accum_steps={grad_accum_steps}, need_lite_recon_forward={need_lite_recon_forward}")
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
                logits = _forward_lite_eval(lite_vae, lite_classifier, image, feature_mode=lite_feature_mode)
                logits = F.softmax(logits, dim=1)
                groundTruth = torch.cat((groundTruth, label))
                activations = torch.cat((activations, logits))
        acc, f1, auc, bac, sens, spec = compute_avg_metrics(groundTruth, activations)
        lite_vae.train(training_vae)
        lite_classifier.train(training_cls)
        return acc, f1, auc, bac, sens, spec

    def _epoch_val_mix(model, lite_vae, kd_feat_proj, data_loader, alpha):
        if lite_vae is None:
            return None
        training_model = model.training
        training_vae = lite_vae.training
        model.eval()
        lite_vae.eval()
        groundTruth = torch.Tensor().cuda()
        activations = torch.Tensor().cuda()
        encoder = _get_encoder(model)
        classifier = _get_classifier(model)
        with torch.no_grad():
            for image, label in data_loader:
                image, label = image.cuda(), label.cuda()
                feat_t = encoder(image)
                mu, _, z, _ = lite_vae(image)
                feat_s = _select_lite_feature(mu, z, mix_feature_mode)
                if kd_feat_proj is not None:
                    feat_s = kd_feat_proj(feat_s)
                elif feat_s.size(1) != feat_t.size(1):
                    raise ValueError("Feature dims do not match and no projection is provided for mix eval")
                mix_feat = (1 - alpha) * feat_t + alpha * feat_s
                logits = classifier(mix_feat)
                logits = F.softmax(logits, dim=1)
                groundTruth = torch.cat((groundTruth, label))
                activations = torch.cat((activations, logits))
        acc, f1, auc, bac, sens, spec = compute_avg_metrics(groundTruth, activations)
        model.train(training_model)
        lite_vae.train(training_vae)
        return acc, f1, auc, bac, sens, spec

    if args.lite_eval_only:
        if args.rank == 0:
            if args.lite_eval_enable and args.lite_eval_use_classifier and lite_vae is not None and lite_classifier is not None:
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
            if args.mix_eval_enable and lite_vae is not None and model is not None:
                alpha_eval = _compute_mix_alpha(
                    0,
                    args.mix_start_epoch,
                    args.mix_end_epoch,
                    args.mix_alpha_start,
                    args.mix_alpha_end,
                    args.mix_schedule,
                )
                mix_val = _epoch_val_mix(model, lite_vae, kd_feat_proj, val_loader, alpha_eval)
                mix_test = _epoch_val_mix(model, lite_vae, kd_feat_proj, test_loader, alpha_eval)
                if mix_val is not None and mix_test is not None:
                    mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec = mix_val
                    mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec = mix_test
                    msg_val = "mix_val(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                        float(alpha_eval),
                        mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec
                    )
                    msg_test = "mix_test(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                        float(alpha_eval),
                        mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec
                    )
                    print(msg_val)
                    print(msg_test)
                    _write_local_log(log_f, msg_val)
                    _write_local_log(log_f, msg_test)
                    if logger is not None:
                        logger.log({'mix_validation': {'Accuracy': mval_acc,
                                                       'F1 score': mval_f1,
                                                       'AUC': mval_auc,
                                                       'Balanced Accuracy': mval_bac,
                                                       'Sensitivity': mval_sens,
                                                       'Specificity': mval_spec}})
                        logger.log({'mix_test': {'Accuracy': mtest_acc,
                                                 'F1 score': mtest_f1,
                                                 'AUC': mtest_auc,
                                                 'Balanced Accuracy': mtest_bac,
                                                 'Sensitivity': mtest_sens,
                                                 'Specificity': mtest_spec}})
        return

    if model is None and args.mix_enable:
        raise RuntimeError("mix_enable=True requires ResNet model, but model is None.")
    if model is None and not (args.kd_enable and args.kd_only):
        raise RuntimeError("ResNet model is None, but training is not in kd_only mode.")
    if model is None and kd_teacher_source == "resnet":
        raise RuntimeError("kd_teacher_source=resnet requires ResNet model, but model is None.")
    if model is None and show_teacher_metrics:
        if args.rank == 0:
            _write_local_log(log_f, "[Warn] show_teacher_metrics=True ignored because ResNet model is None.")
        show_teacher_metrics = False

    warned_no_teacher_logits = False
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, ((img, ema_img), label) in enumerate(train_loader): #batch
            img, ema_img, label = img.cuda(non_blocking=True), ema_img.cuda(non_blocking=True), label.cuda(
                non_blocking=True)
            is_last_iter = (i + 1) == len(train_loader)

            kd_only = args.kd_enable and args.kd_only
            disable_mrc = kd_only or bool(getattr(args, "mix_disable_mrc", False)) or (model is None)
            activations, outputs = None, None
            if model is not None:
                if args.kd_enable and args.kd_freeze_teacher:
                    with torch.no_grad():
                        activations, outputs = model(img)
                else:
                    activations, outputs = model(img)

            teacher_outputs = outputs
            teacher_feat_for_kd = activations
            mix_alpha = None
            lite_mu = lite_logvar = lite_z = lite_recon = None
            lite_feat = None

            if (args.mix_enable or args.kd_enable) and lite_vae is not None:
                lite_mu, lite_logvar, lite_z, lite_recon = _forward_lite_model(
                    lite_vae,
                    img,
                    need_recon=need_lite_recon_forward,
                )

            if (
                args.kd_enable
                and kd_teacher_source == "lite"
                and lite_vae is not None
                and lite_classifier is not None
            ):
                teacher_input = ema_img if kd_lite_teacher_use_weak_aug else img
                with torch.no_grad():
                    if lite_vae_teacher is not None and lite_classifier_teacher is not None:
                        t_mu, _, t_z, _ = _forward_lite_model(lite_vae_teacher, teacher_input, need_recon=False)
                        teacher_feat_for_kd = _select_lite_feature(t_mu, t_z, lite_feature_mode)
                        teacher_outputs = lite_classifier_teacher(teacher_feat_for_kd)
                    else:
                        t_mu, _, t_z, _ = _forward_lite_model(lite_vae, teacher_input, need_recon=False)
                        teacher_feat_for_kd = _select_lite_feature(t_mu, t_z, lite_feature_mode)
                        teacher_outputs = lite_classifier(teacher_feat_for_kd)
            if (
                args.kd_enable
                and kd_teacher_source == "vavae"
                and vavae_teacher is not None
            ):
                teacher_input = ema_img if kd_vavae_teacher_use_weak_aug else img
                with torch.no_grad():
                    teacher_feat_for_kd = vavae_teacher(teacher_input)
                teacher_outputs = None

            if args.mix_enable and lite_z is not None:
                lite_feat = _select_lite_feature(lite_mu, lite_z, mix_feature_mode)
                if kd_feat_proj is not None:
                    lite_feat = kd_feat_proj(lite_feat)
                elif lite_feat.size(1) != activations.size(1):
                    raise ValueError("Feature dims do not match and no projection is provided for mix")
                mix_alpha = _compute_mix_alpha(
                    epoch,
                    args.mix_start_epoch,
                    args.mix_end_epoch,
                    args.mix_alpha_start,
                    args.mix_alpha_end,
                    args.mix_schedule,
                )
                mix_feat = (1 - mix_alpha) * activations + mix_alpha * lite_feat
                outputs = _get_classifier(model)(mix_feat)

            if not disable_mrc:
                with torch.no_grad():
                    ema_activations, ema_output = ema_model(ema_img)
            else:
                ema_activations, ema_output = None, None

            # classification loss
            if outputs is None:
                classification_loss = torch.tensor(0.0, device=img.device)
            else:
                classification_loss = classification_loss_func(outputs, label)

            # probability distribution loss
            if not disable_mrc:
                probability_loss = torch.sum(probability_loss_func(outputs, ema_output)) / args.batch_size
            else:
                probability_loss = torch.tensor(0.0, device=activations.device)
            
            # batch loss
            if not disable_mrc:
                batch_sim_loss = torch.sum(batch_sim_loss_func(activations, ema_activations))
            else:
                batch_sim_loss = torch.tensor(0.0, device=activations.device)

            # channel loss
            if not disable_mrc:
                channel_sim_loss = torch.sum(channel_sim_loss_func(activations, ema_activations))
            else:
                channel_sim_loss = torch.tensor(0.0, device=activations.device)

            base_loss = classification_loss * args.classification_loss_weight
            if not disable_mrc and epoch > 20:
                base_loss = base_loss + probability_loss * args.probability_loss_weight + batch_sim_loss * args.batch_loss_weight + channel_sim_loss * args.channel_loss_weight
            if not disable_mrc and epoch >= args.gaussian_prior_start_epoch and args.gaussian_prior_weight > 0:
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
            kd_struct_loss = torch.tensor(0.0, device=activations.device)
            lite_recon_loss = torch.tensor(0.0, device=activations.device)
            lite_kl_loss = torch.tensor(0.0, device=activations.device)
            lite_ce_loss = torch.tensor(0.0, device=activations.device)
            lite_acc = torch.tensor(0.0, device=activations.device)

            if args.kd_enable and lite_vae is not None and lite_classifier is not None and lite_z is not None:
                lite_student_feat = _select_lite_feature(lite_mu, lite_z, lite_feature_mode)
                if lite_gaussian_prior_loss_func is not None:
                    with torch.no_grad():
                        lite_gaussian_prior_loss_func(lite_student_feat.detach(), label)
                lite_logits = lite_classifier(lite_student_feat)

                if args.kd_logit_weight > 0:
                    if teacher_outputs is not None:
                        t = args.kd_temperature
                        teacher_logits = teacher_outputs.detach()
                        kd_logit_loss = F.kl_div(
                            F.log_softmax(lite_logits / t, dim=1),
                            F.softmax(teacher_logits / t, dim=1),
                            reduction="batchmean",
                        ) * (t * t)
                    elif args.rank == 0 and not warned_no_teacher_logits:
                        _write_local_log(
                            log_f,
                            "[KD] Warning: teacher logits unavailable for current teacher source; kd_logit term is skipped.",
                        )
                        warned_no_teacher_logits = True

                feat_active = args.kd_feat_weight > 0 and epoch >= kd_feat_start_epoch
                struct_active = getattr(args, "kd_struct_weight", 0.0) > 0 and epoch >= kd_struct_start_epoch
                if feat_active or struct_active:
                    feat_s = lite_student_feat
                    feat_t = teacher_feat_for_kd.detach()
                    if kd_feat_proj is not None:
                        feat_s = kd_feat_proj(feat_s)
                    elif feat_s.size(1) != feat_t.size(1):
                        raise ValueError("kd_feat_project is False but feature dims do not match")
                    if feat_active:
                        feat_s_mse = feat_s
                        feat_t_mse = feat_t
                        if args.kd_feat_norm == "l2":
                            feat_s_mse = F.normalize(feat_s_mse, dim=1)
                            feat_t_mse = F.normalize(feat_t_mse, dim=1)
                        kd_feat_loss = F.mse_loss(feat_s_mse, feat_t_mse)
                    if struct_active:
                        struct_norm = getattr(args, "kd_struct_norm", "l2")
                        struct_center = bool(getattr(args, "kd_struct_center", False))
                        if kd_struct_type == "cka":
                            cka_sim = _linear_cka_similarity(feat_s, feat_t, center=struct_center)
                            kd_struct_loss = 1.0 - cka_sim
                        else:
                            gram_s = _batch_gram(feat_s, norm=struct_norm, center=struct_center)
                            gram_t = _batch_gram(feat_t, norm=struct_norm, center=struct_center)
                            kd_struct_loss = F.mse_loss(gram_s, gram_t)

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
            loss = loss + kd_struct_loss * getattr(args, "kd_struct_weight", 0.0)
            loss = loss + lite_recon_loss * args.lite_vae_recon_weight
            loss = loss + lite_kl_loss * args.lite_vae_kl_weight
            loss = loss + lite_ce_loss * args.lite_student_ce_weight

            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            loss_for_backward = loss / float(grad_accum_steps)
            loss_for_backward.backward()

            should_step = ((i + 1) % grad_accum_steps == 0) or is_last_iter
            if should_step:
                if args.grad_clip_enable and args.grad_clip_norm > 0:
                    params = []
                    for group in optimizer.param_groups:
                        params.extend([p for p in group["params"] if p.requires_grad])
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # update ema model
                if not kd_only:
                    update_ema_variables(model, ema_model, args.ema_decay, cur_iters)
                if (
                    args.kd_enable
                    and kd_teacher_source == "lite"
                    and kd_lite_teacher_use_ema
                    and lite_vae_teacher is not None
                    and lite_classifier_teacher is not None
                    and lite_vae is not None
                    and lite_classifier is not None
                ):
                    update_ema_variables(lite_vae, lite_vae_teacher, kd_lite_teacher_ema_decay, cur_iters)
                    update_ema_variables(lite_classifier, lite_classifier_teacher, kd_lite_teacher_ema_decay, cur_iters)

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))


            cur_iters += 1
            if args.rank == 0:
                if cur_iters % 500 == 1 and logger is not None:
                    logger.log({'Strong augmentation': [wandb.Image(item) for item in img.permute(0,2,3,1).detach().cpu().numpy()[:5]]})
                    logger.log({'Weak augmentation': [wandb.Image(item) for item in ema_img.permute(0,2,3,1).detach().cpu().numpy()[:5]]})
                train_log_every_iters = max(1, int(getattr(args, "train_log_every_iters", 10)))
                console_log_every_iters = max(1, int(getattr(args, "console_log_every_iters", 10)))
                should_log_train = ((i + 1) % train_log_every_iters == 0) or is_last_iter
                if should_log_train:
                    _write_local_log(
                        log_f,
                        "epoch={:d} iter={:d} train: total={:.6f}, prob={:.6f}, batch={:.6f}, channel={:.6f}, cls={:.6f}, "
                        "gauss={:.6f}, aux_recon={:.6f}, aux_kl={:.6f}, kd_logit={:.6f}, kd_feat={:.6f}, kd_struct={:.6f}, "
                        "lite_recon={:.6f}, lite_kl={:.6f}, lite_ce={:.6f}, lite_acc={:.6f}, mix_alpha={}".format(
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
                            kd_struct_loss.item(),
                            lite_recon_loss.item(),
                            lite_kl_loss.item(),
                            lite_ce_loss.item(),
                            lite_acc.item(),
                            "None" if mix_alpha is None else float(mix_alpha),
                        ),
                    )
                eval_every_epochs = max(1, int(getattr(args, "eval_every_epochs", 5)))
                should_eval = is_last_iter and (((epoch + 1) % eval_every_epochs == 0) or ((epoch + 1) == args.epochs))
                if should_eval:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    val_acc = val_f1 = val_auc = val_bac = val_sens = val_spec = None
                    test_acc = test_f1 = test_auc = test_bac = test_sens = test_spec = None
                    if show_teacher_metrics:
                        val_acc, val_f1, val_auc, val_bac, val_sens, val_spec = epochVal(model, val_loader)
                        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec = epochVal(model, test_loader)
                    lite_metrics = None
                    if args.lite_eval_enable and args.lite_eval_use_classifier and lite_vae is not None and lite_classifier is not None:
                        lite_val = _epoch_val_lite(lite_vae, lite_classifier, val_loader)
                        lite_test = _epoch_val_lite(lite_vae, lite_classifier, test_loader)
                        lite_metrics = (lite_val, lite_test)
                    mix_metrics = None
                    if args.mix_eval_enable and lite_vae is not None and model is not None:
                        alpha_eval = _compute_mix_alpha(
                            epoch,
                            args.mix_start_epoch,
                            args.mix_end_epoch,
                            args.mix_alpha_start,
                            args.mix_alpha_end,
                            args.mix_schedule,
                        )
                        mix_val = _epoch_val_mix(model, lite_vae, kd_feat_proj, val_loader, alpha_eval)
                        mix_test = _epoch_val_mix(model, lite_vae, kd_feat_proj, test_loader, alpha_eval)
                        mix_metrics = (mix_val, mix_test, alpha_eval)
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
                                                 'kd struct loss': kd_struct_loss.item(),
                                                 'lite recon loss': lite_recon_loss.item(),
                                                 'lite kl loss': lite_kl_loss.item(),
                                                 'lite ce loss': lite_ce_loss.item(),
                                                 'lite acc (batch)': lite_acc.item()}})
                        if show_teacher_metrics:
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
                        if mix_metrics is not None:
                            (mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec), (mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec), alpha_eval = mix_metrics
                            logger.log({'mix_test': {'Accuracy': mtest_acc,
                                                     'F1 score': mtest_f1,
                                                     'AUC': mtest_auc,
                                                     'Balanced Accuracy': mtest_bac,
                                                     'Sensitivity': mtest_sens,
                                                     'Specificity': mtest_spec},
                                        'mix_validation': {'Accuracy': mval_acc,
                                                           'F1 score': mval_f1,
                                                           'AUC': mval_auc,
                                                           'Balanced Accuracy': mval_bac,
                                                           'Sensitivity': mval_sens,
                                                           'Specificity': mval_spec},
                                        'mix_alpha': float(alpha_eval)})
                    if show_teacher_metrics:
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
                    if mix_metrics is not None:
                        (mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec), (mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec), alpha_eval = mix_metrics
                        _write_local_log(
                            log_f,
                            "epoch={:d} mix_test(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                float(alpha_eval),
                                mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec
                            ),
                        )
                        _write_local_log(
                            log_f,
                            "epoch={:d} mix_val(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                float(alpha_eval),
                                mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec
                            ),
                        )
                    if show_teacher_metrics:
                        print(
                            "\nepoch={:d} test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                test_acc, test_f1, test_auc, test_bac, test_sens, test_spec
                            )
                        )
                        print(
                            "epoch={:d} val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                val_acc, val_f1, val_auc, val_bac, val_sens, val_spec
                            )
                        )
                    if lite_metrics is not None:
                        (lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec), (ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec) = lite_metrics
                        print(
                            "epoch={:d} lite_test: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                ltest_acc, ltest_f1, ltest_auc, ltest_bac, ltest_sens, ltest_spec
                            )
                        )
                        print(
                            "epoch={:d} lite_val: acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                lval_acc, lval_f1, lval_auc, lval_bac, lval_sens, lval_spec
                            )
                        )
                    if mix_metrics is not None:
                        (mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec), (mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec), alpha_eval = mix_metrics
                        print(
                            "epoch={:d} mix_test(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                float(alpha_eval),
                                mtest_acc, mtest_f1, mtest_auc, mtest_bac, mtest_sens, mtest_spec
                            )
                        )
                        print(
                            "epoch={:d} mix_val(alpha={:.3f}): acc={:.6f}, f1={:.6f}, auc={:.6f}, bac={:.6f}, sens={:.6f}, spec={:.6f}".format(
                                epoch + 1,
                                float(alpha_eval),
                                mval_acc, mval_f1, mval_auc, mval_bac, mval_sens, mval_spec
                            )
                        )
                    if show_teacher_metrics:
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
                if ((i + 1) % console_log_every_iters == 0) or is_last_iter:
                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True)

        if args.rank == 0:
            if model is not None:
                saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_.pth'.format(epoch + 1))
                state_dict = _get_state_dict(model)
                torch.save(state_dict, saveModelPath)
            if getattr(args, "save_stage1_gaussian_stats", False):
                if getattr(args, "stage1_gaussian_save_every_epoch", False):
                    gp_epoch_path = os.path.join(args.checkpoints, "gaussian_prior_epoch_{:d}_.pth".format(epoch + 1))
                    _save_gaussian_prior_stats(gp_epoch_path, gaussian_prior_loss_func)
                if getattr(args, "stage1_gaussian_save_latest", True):
                    gp_latest_path = os.path.join(args.checkpoints, "gaussian_prior_latest.pth")
                    _save_gaussian_prior_stats(gp_latest_path, gaussian_prior_loss_func)
            if getattr(args, "save_stage1_lite_gaussian_stats", False):
                if getattr(args, "stage1_lite_gaussian_save_every_epoch", False):
                    lgp_epoch_path = os.path.join(args.checkpoints, "lite_gaussian_prior_epoch_{:d}_.pth".format(epoch + 1))
                    _save_gaussian_prior_stats(lgp_epoch_path, lite_gaussian_prior_loss_func)
                if getattr(args, "stage1_lite_gaussian_save_latest", True):
                    lgp_latest_path = os.path.join(args.checkpoints, "lite_gaussian_prior_latest.pth")
                    _save_gaussian_prior_stats(lgp_latest_path, lite_gaussian_prior_loss_func)

            if (args.kd_enable or args.mix_enable) and args.kd_save_lite and lite_vae is not None:
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


        
