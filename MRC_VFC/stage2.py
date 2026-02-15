import os
import glob
import json
import time
import wandb
import argparse
import torch
import numpy as np
from models import CreateModel, Linear, LiteVAE
from data import (
    Transforms,
    ISICDataset,
    compute_virtual_class_sizes,
    fit_class_gaussians,
    sample_virtual_representations,
)
from utils.yaml_config_hook import yaml_config_hook
from torch.utils.data import DataLoader
from utils import epochVal
from utils.loss import GCELoss


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def _extract_feature_batch(feature_model, x, feature_source="resnet", lite_feature_mode="mu"):
    if feature_source == "resnet":
        activations, _ = feature_model(x)
        return activations
    if feature_source == "lite":
        mu, _, z, _ = feature_model(x)
        if lite_feature_mode == "z":
            return z
        if lite_feature_mode == "mu":
            return mu
        raise ValueError(f"Unsupported stage2_lite_feature_mode: {lite_feature_mode}")
    raise ValueError(f"Unsupported stage2_feature_source: {feature_source}")


def inference(loader, feature_model, device, feature_source="resnet", lite_feature_mode="mu"):
    feature_vector = []
    labels_vector = []
    feature_model.eval()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            activations = _extract_feature_batch(feature_model, x, feature_source, lite_feature_mode)

        activations = activations.detach()
        feature_vector.extend(activations.cpu().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector, dtype=np.float32)
    labels_vector = np.array(labels_vector, dtype=np.int64)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(feature_model, train_loader, test_loader, val_loader, device, feature_source="resnet", lite_feature_mode="mu"):
    train_X, train_y = inference(train_loader, feature_model, device, feature_source, lite_feature_mode)
    test_X, test_y = inference(test_loader, feature_model, device, feature_source, lite_feature_mode)
    val_X, val_y = inference(val_loader, feature_model, device, feature_source, lite_feature_mode)
    return train_X, train_y, test_X, test_y, val_X, val_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, X_val, y_val, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, val_loader


def e_step_resnet(backbone, classifier, opt, loader, loss_func, device, logger):
    """
    Update ResNet feature extractor while freezing classifier.
    """
    backbone.train()
    classifier.eval()
    _set_requires_grad(classifier, False)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        activations, _ = backbone(x)
        out = classifier(activations)

        loss = loss_func(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if logger is not None:
            logger.log({"E Step loss": loss.item()})
    _set_requires_grad(classifier, True)


def e_step_lite(lite_vae, classifier, opt, loader, loss_func, device, lite_feature_mode, logger):
    """
    Update LiteVAE feature extractor while freezing classifier.
    """
    lite_vae.train()
    classifier.eval()
    _set_requires_grad(classifier, False)
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        mu, _, z, _ = lite_vae(x)
        features = z if lite_feature_mode == "z" else mu
        out = classifier(features)
        loss = loss_func(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if logger is not None:
            logger.log({"E Step loss": loss.item()})
    _set_requires_grad(classifier, True)


def m_step(classifier, opt, loader, loss_func, device, logger):
    """
    Freeze the backbone and train the classifier with virtual samples,
    i.e., maximize the expectation of the distribution of the features
    :return:
    """
    epoch_loss = 0
    epoch_acc = 0
    classifier.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = classifier(x)
        loss = loss_func(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        predict = out.argmax(1)
        acc = (predict == y).sum().item() / y.size(0)

        epoch_acc += acc
        epoch_loss += loss.item()
        if logger is not None:
            logger.log({"M Step loss": loss.item()})
    return epoch_loss, epoch_acc


def _parse_epoch_from_name(path, prefix):
    name = os.path.basename(path)
    marker = f"{prefix}_epoch_"
    if marker not in name:
        return -1
    try:
        body = name.split(marker, 1)[1]
        return int(body.split("_.pth", 1)[0])
    except Exception:
        return -1


def _auto_find_checkpoint(checkpoints_root, run_name, preferred_name, epoch_prefix):
    run_dir = os.path.join(checkpoints_root, run_name)
    if preferred_name:
        cand = os.path.join(run_dir, preferred_name)
        if os.path.exists(cand):
            return cand
    files = glob.glob(os.path.join(run_dir, f"{epoch_prefix}_epoch_*_.pth"))
    if not files:
        return ""
    files = sorted(files, key=lambda p: _parse_epoch_from_name(p, epoch_prefix))
    return files[-1]


def _resolve_checkpoint_path(path, checkpoints_root, run_name):
    candidates = []
    if path:
        if os.path.isabs(path):
            candidates.append(path)
        else:
            candidates.append(os.path.join(checkpoints_root, run_name, path))
            candidates.append(os.path.join(checkpoints_root, path))
            candidates.append(path)
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _resolve_stats_path(path, checkpoints_dir, default_name):
    if path:
        if os.path.isabs(path):
            out = path
        else:
            out = os.path.join(checkpoints_dir, path)
    else:
        out = os.path.join(checkpoints_dir, default_name)
    if not out.endswith(".npz"):
        out = out + ".npz"
    return out


def _load_virtual_class_sizes(path, class_num, checkpoints_dir=""):
    if not path:
        return None
    load_path = path
    if not os.path.isabs(load_path) and checkpoints_dir:
        cand = os.path.join(checkpoints_dir, load_path)
        if os.path.exists(cand):
            load_path = cand
    with open(load_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        sizes = np.array(obj, dtype=np.int64)
    elif isinstance(obj, dict):
        sizes = np.zeros(class_num, dtype=np.int64)
        for k, v in obj.items():
            sizes[int(k)] = int(v)
    else:
        raise ValueError("stage2_virtual_counts_path must be a JSON list or dict")
    if sizes.shape[0] != class_num:
        raise ValueError(f"virtual class sizes length mismatch: {sizes.shape[0]} != {class_num}")
    return sizes


def _save_gaussian_stats(path, stats):
    payload = {
        "covariance_type": np.array(stats["covariance_type"]),
        "means": stats["means"],
        "var_floor": np.array(stats.get("var_floor", 1e-4), dtype=np.float32),
    }
    if "cov_diag" in stats:
        payload["cov_diag"] = stats["cov_diag"]
    if "cov_full" in stats:
        payload["cov_full"] = stats["cov_full"]
    np.savez(path, **payload)


def _load_gaussian_stats(path):
    raw = np.load(path, allow_pickle=False)
    covariance_type = str(raw["covariance_type"])
    stats = {
        "covariance_type": covariance_type,
        "means": raw["means"].astype(np.float32),
        "var_floor": float(raw["var_floor"]),
    }
    if covariance_type == "diag":
        stats["cov_diag"] = raw["cov_diag"].astype(np.float32)
    elif covariance_type == "full":
        stats["cov_full"] = raw["cov_full"].astype(np.float32)
    else:
        raise ValueError(f"Unsupported covariance_type in stats file: {covariance_type}")
    return stats


def _load_stage1_gaussian_stats(path, expected_classes, expected_feat_dim):
    state = torch.load(path, map_location="cpu")
    means = state.get("means", None)
    vars_ = state.get("vars", None)
    if means is None or vars_ is None:
        raise ValueError(f"Invalid stage1 gaussian stats file: {path}")
    means = means.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(means) else np.asarray(means, dtype=np.float32)
    vars_ = vars_.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(vars_) else np.asarray(vars_, dtype=np.float32)

    if means.shape[0] != expected_classes:
        raise ValueError(f"stage1 gaussian class mismatch: {means.shape[0]} != {expected_classes}")
    if means.shape[1] != expected_feat_dim:
        raise ValueError(f"stage1 gaussian dim mismatch: {means.shape[1]} != {expected_feat_dim}")
    if vars_.shape != means.shape:
        raise ValueError(f"stage1 gaussian vars shape mismatch: {vars_.shape} != {means.shape}")

    var_floor = float(state.get("var_floor", 1e-4))
    vars_ = np.maximum(vars_, var_floor)
    return {
        "covariance_type": "diag",
        "means": means,
        "cov_diag": vars_,
        "var_floor": var_floor,
    }


def _write_local_log(log_f, msg):
    if log_f is None:
        return
    log_f.write(msg + "\n")
    log_f.flush()


def _log_metrics_local(log_f, prefix, metrics):
    parts = [f"{k}={v:.6f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics.items()]
    _write_local_log(log_f, f"{prefix}: " + ", ".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    parser.add_argument('--log_file', type=str, default="", help='write debug logs to a local file')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoints_root = args.checkpoints
    if not args.run_name:
        args.run_name = time.strftime("run_%Y%m%d_%H%M%S")
    args.checkpoints = os.path.join(args.checkpoints_root, args.run_name)
    os.makedirs(args.checkpoints, exist_ok=True)

    log_f = None
    if args.debug and args.log_file:
        log_f = open(args.log_file, "w", encoding="utf-8")
        _write_local_log(log_f, f"Stage2 start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not args.debug:
        wandb.login(key="[Your wandb key here]")

        config = dict()
        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MRC_VFC_on_%s"%args.dataset,
            notes="MICCAI 2023",
            tags=["MICCAI23", "Class imbalance", "Dermoscopy", "Representation Learning"],
            config=config
        )
    else:
        wandb_logger = None

    transforms = Transforms(size=args.image_size)
    train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms.test_transform)
    test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
    val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # Load stage1 feature extractor (ResNet or LiteVAE)
    n_classes = train_dataset.n_class
    feature_source = str(getattr(args, "stage2_feature_source", "resnet")).lower()
    if feature_source not in ("resnet", "lite"):
        raise ValueError("stage2_feature_source must be one of: resnet | lite")
    if feature_source == "lite":
        print(f"[Stage2] feature_source=lite, backbone={args.backbone} is ignored.")

    source_teacher_run = args.teacher_run_name if args.teacher_run_name else args.run_name
    source_student_run = args.student_run_name if args.student_run_name else args.run_name
    stage1_epoch = args.teacher_epoch if args.teacher_epoch > 0 else args.epochs
    lite_feature_mode = str(getattr(args, "stage2_lite_feature_mode", "mu")).lower()

    feature_model = None
    feature_optimizer = None
    feature_criterion = None
    feature_dim = None
    loaded_feature_ckpt = ""

    if feature_source == "resnet":
        feature_model = CreateModel(backbone=args.backbone, out_features=n_classes, pretrained=False).to(args.device)
        backbone_ckpt = _resolve_checkpoint_path(
            getattr(args, "stage2_backbone_resume_path", ""),
            args.checkpoints_root,
            source_teacher_run,
        )
        if not backbone_ckpt:
            run_dir = os.path.join(args.checkpoints_root, source_teacher_run)
            fallback = os.path.join(run_dir, f"epoch_{stage1_epoch}_.pth")
            if os.path.exists(fallback):
                backbone_ckpt = fallback
            else:
                backbone_ckpt = _auto_find_checkpoint(args.checkpoints_root, source_teacher_run, "", "epoch")
        if not backbone_ckpt:
            raise FileNotFoundError("Stage2 could not find ResNet checkpoint for Stage1 output.")
        feature_model.load_state_dict(torch.load(backbone_ckpt, map_location=args.device))
        loaded_feature_ckpt = backbone_ckpt
        feature_dim = feature_model.n_features
    else:
        feature_model = LiteVAE(
            image_size=args.image_size,
            in_channels=3,
            base_channels=args.lite_vae_base_channels,
            latent_dim=args.lite_vae_latent_dim,
            dwt_levels=args.lite_vae_dwt_levels,
            variant=args.lite_vae_variant,
        ).to(args.device)
        lite_ckpt = _resolve_checkpoint_path(args.lite_vae_resume_path, args.checkpoints_root, source_student_run)
        if not lite_ckpt:
            lite_ckpt = _auto_find_checkpoint(args.checkpoints_root, source_student_run, "litevae_latest.pth", "litevae")
        if not lite_ckpt:
            raise FileNotFoundError("Stage2 could not find LiteVAE checkpoint for Stage1 output.")
        feature_model.load_state_dict(torch.load(lite_ckpt, map_location=args.device))
        loaded_feature_ckpt = lite_ckpt
        feature_dim = args.lite_vae_latent_dim

    if bool(getattr(args, "stage2_enable_estep", True)):
        feature_optimizer = torch.optim.SGD(
            feature_model.parameters(), lr=args.backbone_lr, momentum=0.9, weight_decay=1e-4
        )
        feature_criterion = GCELoss(num_classes=n_classes)

    # Classifier
    classifier_model = Linear(feature_dim, n_classes)
    classifier_model = classifier_model.to(args.device)
    classifier_optimizer = torch.optim.SGD(classifier_model.parameters(),
                                           lr=args.classifier_lr, momentum=0.9, weight_decay=1e-4)
    classifier_criterion = torch.nn.CrossEntropyLoss()

    _write_local_log(log_f, f"feature_source={feature_source}, lite_feature_mode={lite_feature_mode}")
    _write_local_log(log_f, f"loaded_feature_ckpt={loaded_feature_ckpt}")

    stats_path = _resolve_stats_path(
        getattr(args, "stage2_gaussian_stats_path", ""),
        args.checkpoints,
        f"stage2_{feature_source}_gaussian_stats.npz",
    )
    use_saved_gaussian = bool(getattr(args, "stage2_use_saved_gaussian", False))
    save_gaussian = bool(getattr(args, "stage2_save_gaussian_stats", True))
    recompute_gaussian_each_epoch = bool(getattr(args, "stage2_recompute_gaussian_each_epoch", False))
    recompute_features_each_epoch = bool(getattr(args, "stage2_recompute_features_each_epoch", True))

    gaussian_stats = None
    class_sizes_override = _load_virtual_class_sizes(
        getattr(args, "stage2_virtual_counts_path", ""),
        n_classes,
        checkpoints_dir=args.checkpoints,
    )
    use_stage1_gaussian_init = bool(getattr(args, "stage2_use_stage1_gaussian_init", False))
    stage1_gaussian_path = _resolve_checkpoint_path(
        getattr(args, "stage2_stage1_gaussian_path", ""),
        args.checkpoints_root,
        source_teacher_run,
    )
    if not stage1_gaussian_path:
        auto_stage1_gaussian = os.path.join(args.checkpoints_root, source_teacher_run, "gaussian_prior_latest.pth")
        if os.path.exists(auto_stage1_gaussian):
            stage1_gaussian_path = auto_stage1_gaussian

    best_val_acc = -1.0
    best_test_acc = -1.0
    train_X = train_y = test_X = test_y = val_X = val_y = None

    for epoch in range(args.stage2_epochs):
        need_refresh_features = (
            train_X is None
            or recompute_features_each_epoch
            or (bool(getattr(args, "stage2_enable_estep", True)) and epoch > 0)
        )
        if need_refresh_features:
            train_X, train_y, test_X, test_y, val_X, val_y = get_features(
                feature_model,
                train_loader,
                test_loader,
                val_loader,
                args.device,
                feature_source=feature_source,
                lite_feature_mode=lite_feature_mode,
            )

        need_refit_gaussian = (
            gaussian_stats is None
            or recompute_gaussian_each_epoch
            or need_refresh_features
        )
        if use_saved_gaussian and epoch == 0 and os.path.exists(stats_path):
            gaussian_stats = _load_gaussian_stats(stats_path)
        elif use_stage1_gaussian_init and epoch == 0 and stage1_gaussian_path:
            try:
                gaussian_stats = _load_stage1_gaussian_stats(stage1_gaussian_path, n_classes, feature_dim)
                _write_local_log(log_f, f"stage1 gaussian init loaded: {stage1_gaussian_path}")
            except Exception as e:
                _write_local_log(log_f, f"stage1 gaussian init skipped: {e}")
                gaussian_stats = None
        elif need_refit_gaussian:
            gaussian_stats = fit_class_gaussians(
                train_X,
                train_y,
                n_classes,
                covariance_type=getattr(args, "stage2_gaussian_covariance", "diag"),
                var_floor=float(getattr(args, "stage2_gaussian_var_floor", 1e-4)),
                full_min_samples=int(getattr(args, "stage2_gaussian_full_min_samples", 32)),
                full_shrinkage=float(getattr(args, "stage2_gaussian_full_shrinkage", 0.1)),
            )
            if save_gaussian:
                _save_gaussian_stats(stats_path, gaussian_stats)

        train_X_for_cls = train_X
        train_y_for_cls = train_y
        virtual_mode = getattr(args, "stage2_virtual_mode", "uniform")
        virtual_enabled = bool(getattr(args, "stage2_virtual_enable", True))
        if virtual_enabled:
            if class_sizes_override is not None:
                class_sizes = class_sizes_override
            else:
                class_sizes = compute_virtual_class_sizes(
                    train_y,
                    n_classes,
                    uniform_size=args.virtual_size,
                    mode=virtual_mode,
                    tail_scale=float(getattr(args, "stage2_tail_scale", 1.0)),
                    tail_target=getattr(args, "stage2_tail_target", "max"),
                    min_size=int(getattr(args, "stage2_virtual_min_per_class", 0)),
                    max_size=int(getattr(args, "stage2_virtual_max_per_class", -1)),
                )
            max_virtual_ratio = float(getattr(args, "stage2_virtual_max_ratio", -1.0))
            if max_virtual_ratio > 0:
                max_total = int(len(train_y) * max_virtual_ratio)
                cur_total = int(class_sizes.sum())
                if cur_total > max_total and cur_total > 0:
                    scale = max_total / float(cur_total)
                    class_sizes = np.floor(class_sizes.astype(np.float64) * scale).astype(np.int64)

            virtual_X, virtual_y = sample_virtual_representations(gaussian_stats, class_sizes)
            merge_real = bool(getattr(args, "stage2_virtual_merge_real", True))
            if merge_real and len(virtual_X) > 0:
                train_X_for_cls = np.concatenate([train_X, virtual_X], axis=0)
                train_y_for_cls = np.concatenate([train_y, virtual_y], axis=0)
            elif len(virtual_X) > 0:
                train_X_for_cls = virtual_X
                train_y_for_cls = virtual_y

            _log_metrics_local(log_f, f"epoch {epoch} virtual", {
                "virtual_total": int(len(virtual_X)),
                "train_total": int(len(train_X_for_cls)),
                "merge_real": int(merge_real),
            })
        else:
            _log_metrics_local(log_f, f"epoch {epoch} virtual", {
                "virtual_total": 0,
                "train_total": int(len(train_X_for_cls)),
                "merge_real": 1,
            })

        arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
            train_X_for_cls, train_y_for_cls, test_X, test_y, val_X, val_y, args.stage2_batch_size
        )

        # m-step: train classifier on real+virtual feature set
        loss_epoch, acc_epoch = m_step(
            classifier_model, classifier_optimizer, arr_train_loader, classifier_criterion, args.device, wandb_logger
        )

        # e-step: optional feature extractor update
        if bool(getattr(args, "stage2_enable_estep", True)):
            if feature_source == "resnet":
                e_step_resnet(
                    feature_model,
                    classifier_model,
                    feature_optimizer,
                    train_loader,
                    feature_criterion,
                    args.device,
                    wandb_logger,
                )
            else:
                e_step_lite(
                    feature_model,
                    classifier_model,
                    feature_optimizer,
                    train_loader,
                    feature_criterion,
                    args.device,
                    lite_feature_mode,
                    wandb_logger,
                )

        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec = epochVal(classifier_model, arr_test_loader)
        val_acc, val_f1, val_auc, val_bac, val_sens, val_spec = epochVal(classifier_model, arr_val_loader)
        if wandb_logger is not None:
            wandb_logger.log({'test': {'Accuracy': test_acc,
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
        _log_metrics_local(log_f, f"epoch {epoch} test", {
            "acc": test_acc, "f1": test_f1, "auc": test_auc, "bac": test_bac, "sens": test_sens, "spec": test_spec
        })
        _log_metrics_local(log_f, f"epoch {epoch} val", {
            "acc": val_acc, "f1": val_f1, "auc": val_auc, "bac": val_bac, "sens": val_sens, "spec": val_spec
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_classifier.pth"))
            torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_feature_extractor.pth"))
            if feature_source == "resnet":
                torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_backbone.pth"))
            else:
                torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_litevae.pth"))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(
            f"Epoch [{epoch}/{args.stage2_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {acc_epoch / len(arr_train_loader)}"
        )
        if log_f is not None:
            _write_local_log(log_f, f"Epoch [{epoch}/{args.stage2_epochs}] Loss={loss_epoch / len(arr_train_loader):.6f} Acc={acc_epoch / len(arr_train_loader):.6f}")

    if log_f is not None:
        _write_local_log(log_f, f"Best val acc={best_val_acc:.6f}, best test acc={best_test_acc:.6f}")
        log_f.close()
