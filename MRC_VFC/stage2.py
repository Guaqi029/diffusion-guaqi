import os
import glob
import json
import time
import argparse
import torch
import numpy as np
from models import CreateModel, Linear, LiteVAE, VAVAEStudentVAE
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

try:
    import wandb
except ImportError:
    wandb = None


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def _arg_type_from_default(v):
    if isinstance(v, bool):
        return _str2bool
    return type(v)


def _sanitize_cuda_alloc_conf():
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        return
    tokens = [x.strip() for x in conf.split(",") if x.strip()]
    kept = [t for t in tokens if not t.startswith("expandable_segments")]
    if kept:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(kept)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    print(
        "[CUDA alloc] Removed unsupported option 'expandable_segments' from "
        "PYTORCH_CUDA_ALLOC_CONF for this run."
    )


def _set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def _extract_feature_batch(feature_model, x, feature_source="resnet", lite_feature_mode="mu"):
    if feature_source == "resnet":
        activations, _ = feature_model(x)
        return activations
    if feature_source in ("lite", "vavae"):
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


def _build_class_weights_np(
    labels,
    num_classes,
    power=1.0,
    min_weight=0.0,
    max_weight=-1.0,
    eps=1e-6,
):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.power(counts + float(eps), -float(power)).astype(np.float32)
    weights = weights / max(float(weights.mean()), float(eps))
    if float(min_weight) > 0:
        weights = np.maximum(weights, float(min_weight))
    if float(max_weight) > 0:
        weights = np.minimum(weights, float(max_weight))
    weights = weights / max(float(weights.mean()), float(eps))
    return weights.astype(np.float32), counts.astype(np.int64)


def _per_class_accuracy(model, data_loader, device, num_classes):
    training = model.training
    model.eval()
    total = np.zeros(num_classes, dtype=np.int64)
    correct = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                _, logits = logits
            pred = logits.argmax(dim=1)

            y_cpu = y.detach().cpu()
            pred_cpu = pred.detach().cpu()
            total += torch.bincount(y_cpu, minlength=num_classes).numpy()
            hit = (pred_cpu == y_cpu)
            if hit.any():
                correct += torch.bincount(y_cpu[hit], minlength=num_classes).numpy()
    model.train(training)
    acc = correct.astype(np.float32) / np.maximum(total, 1).astype(np.float32)
    return acc, total


def _compute_aas_class_sizes(
    base_class_sizes,
    per_class_acc,
    total_target,
    gamma=1.0,
    follow_base_mask=True,
    min_size=0,
    max_size=-1,
    max_virtual_ratio=-1.0,
    real_total=0,
    ema_prev=None,
    ema_momentum=0.0,
):
    base = np.asarray(base_class_sizes, dtype=np.int64)
    acc = np.asarray(per_class_acc, dtype=np.float32)
    acc = np.clip(acc, 0.0, 1.0)
    hard = np.power(1.0 - acc, float(gamma)).astype(np.float32)

    if follow_base_mask:
        mask = (base > 0).astype(np.float32)
        hard = hard * mask
    else:
        mask = np.ones_like(hard, dtype=np.float32)

    if hard.sum() <= 0:
        hard = mask.copy()
        if hard.sum() <= 0:
            hard = np.ones_like(hard, dtype=np.float32)

    weights = hard / hard.sum()
    target = int(max(0, total_target))
    sizes = np.floor(weights * target).astype(np.int64)
    remainder = target - int(sizes.sum())
    if remainder > 0:
        order = np.argsort(-weights)
        for idx in order[:remainder]:
            sizes[idx] += 1

    if int(min_size) > 0:
        if follow_base_mask:
            sizes = np.where(base > 0, np.maximum(sizes, int(min_size)), sizes)
        else:
            sizes = np.maximum(sizes, int(min_size))
    if int(max_size) > 0:
        sizes = np.minimum(sizes, int(max_size))
    if follow_base_mask:
        sizes = np.where(base > 0, sizes, 0)

    if float(max_virtual_ratio) > 0 and int(real_total) > 0:
        max_total = int(real_total * float(max_virtual_ratio))
        cur_total = int(sizes.sum())
        if cur_total > max_total and cur_total > 0:
            scale = max_total / float(cur_total)
            sizes = np.floor(sizes.astype(np.float64) * scale).astype(np.int64)

    sizes_raw = sizes.astype(np.int64).copy()
    ema_active = False
    ema_index = 0.0
    ema_l1_to_raw = 0.0
    ema = float(ema_momentum)
    if ema_prev is not None and 0.0 < ema < 1.0:
        prev = np.asarray(ema_prev, dtype=np.float32)
        smoothed = np.round(prev * ema + sizes.astype(np.float32) * (1.0 - ema)).astype(np.int64)
        denom = float(np.mean(np.abs(prev - sizes_raw.astype(np.float32))))
        numer = float(np.mean(np.abs(smoothed.astype(np.float32) - sizes_raw.astype(np.float32))))
        ema_active = True
        ema_l1_to_raw = numer
        if denom > 1e-8:
            # 该值越接近1，说明EMA平滑影响越大；越接近0，说明接近原始AAS分配。
            ema_index = numer / denom
        else:
            ema_index = 0.0
        sizes = smoothed

    debug = {
        "ema_active": int(ema_active),
        "ema_momentum": float(ema),
        "ema_index": float(ema_index),
        "ema_l1_to_raw": float(ema_l1_to_raw),
        "raw_sizes": sizes_raw.astype(np.int64),
        "smoothed_sizes": sizes.astype(np.int64),
    }

    return sizes.astype(np.int64), hard.astype(np.float32), debug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=_arg_type_from_default(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    parser.add_argument('--log_file', type=str, default="", help='write debug logs to a local file')
    args = parser.parse_args()

    _sanitize_cuda_alloc_conf()
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
        if wandb is None:
            raise ModuleNotFoundError(
                "wandb is not installed. Install it or run with --debug."
            )
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
    if feature_source not in ("resnet", "lite", "vavae"):
        raise ValueError("stage2_feature_source must be one of: resnet | lite | vavae")
    if feature_source in ("lite", "vavae"):
        print(f"[Stage2] feature_source={feature_source}, backbone={args.backbone} is ignored.")

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
    elif feature_source == "lite":
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
    else:
        feature_model = VAVAEStudentVAE(
            in_channels=3,
            ch=int(getattr(args, "vavae_student_ch", getattr(args, "vavae_teacher_ch", 128))),
            ch_mult=getattr(args, "vavae_student_ch_mult", getattr(args, "vavae_teacher_ch_mult", "1,1,2,2,4")),
            num_res_blocks=int(getattr(args, "vavae_student_num_res_blocks", getattr(args, "vavae_teacher_num_res_blocks", 2))),
            z_channels=int(getattr(args, "vavae_student_latent_dim", getattr(args, "vavae_teacher_latent_dim", 32))),
            attn_levels=getattr(args, "vavae_student_attn_levels", getattr(args, "vavae_teacher_attn_levels", "4")),
            input_size=int(getattr(args, "vavae_student_input_size", args.image_size)),
            resize_input=bool(getattr(args, "vavae_student_resize_input", False)),
            pool=str(getattr(args, "vavae_student_pool", "avg")),
            feature_from=str(getattr(args, "vavae_student_feature_from", "mu")),
            enable_decoder=bool(getattr(args, "vavae_student_enable_decoder", False)),
        ).to(args.device)
        vavae_ckpt = _resolve_checkpoint_path(args.lite_vae_resume_path, args.checkpoints_root, source_student_run)
        if not vavae_ckpt:
            vavae_ckpt = _auto_find_checkpoint(args.checkpoints_root, source_student_run, "litevae_latest.pth", "litevae")
        if not vavae_ckpt:
            raise FileNotFoundError("Stage2 could not find VA-VAE student checkpoint for Stage1 output.")
        feature_model.load_state_dict(torch.load(vavae_ckpt, map_location=args.device))
        loaded_feature_ckpt = vavae_ckpt
        feature_dim = int(getattr(args, "vavae_student_latent_dim", getattr(args, "vavae_teacher_latent_dim", 32)))

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
    stage2_use_class_weight = bool(getattr(args, "stage2_use_class_weight", getattr(args, "use_class_weight", False)))
    stage2_class_weight_source = str(getattr(args, "stage2_class_weight_source", "real"))
    if stage2_class_weight_source not in {"real", "train_for_cls"}:
        raise ValueError("stage2_class_weight_source must be one of: real | train_for_cls")
    last_class_weight_signature = None

    _write_local_log(log_f, f"feature_source={feature_source}, lite_feature_mode={lite_feature_mode}")
    _write_local_log(log_f, f"loaded_feature_ckpt={loaded_feature_ckpt}")
    _write_local_log(
        log_f,
        f"stage2_use_class_weight={stage2_use_class_weight}, stage2_class_weight_source={stage2_class_weight_source}",
    )

    stats_path = _resolve_stats_path(
        getattr(args, "stage2_gaussian_stats_path", ""),
        args.checkpoints,
        f"stage2_{feature_source}_gaussian_stats.npz",
    )
    use_saved_gaussian = bool(getattr(args, "stage2_use_saved_gaussian", False))
    save_gaussian = bool(getattr(args, "stage2_save_gaussian_stats", True))
    recompute_gaussian_each_epoch = bool(getattr(args, "stage2_recompute_gaussian_each_epoch", False))
    recompute_features_each_epoch = bool(getattr(args, "stage2_recompute_features_each_epoch", True))
    prioritize_stage1_gaussian = bool(getattr(args, "stage2_prioritize_stage1_gaussian", True))
    refit_after_stage1_gaussian = bool(getattr(args, "stage2_refit_after_stage1_gaussian", False))

    gaussian_stats = None
    gaussian_source = "none"
    stage1_gaussian_loaded = False
    class_sizes_override = _load_virtual_class_sizes(
        getattr(args, "stage2_virtual_counts_path", ""),
        n_classes,
        checkpoints_dir=args.checkpoints,
    )
    use_stage1_gaussian_init = bool(getattr(args, "stage2_use_stage1_gaussian_init", False))
    stage1_gaussian_path = ""
    user_stage1_gaussian = getattr(args, "stage2_stage1_gaussian_path", "")
    if user_stage1_gaussian:
        stage1_gaussian_path = _resolve_checkpoint_path(
            user_stage1_gaussian,
            args.checkpoints_root,
            source_student_run if feature_source == "lite" else source_teacher_run,
        )
        if not stage1_gaussian_path:
            stage1_gaussian_path = _resolve_checkpoint_path(
                user_stage1_gaussian,
                args.checkpoints_root,
                source_teacher_run,
            )
    if not stage1_gaussian_path:
        auto_candidates = []
        if feature_source == "lite":
            auto_candidates.extend([
                os.path.join(args.checkpoints_root, source_student_run, "lite_gaussian_prior_latest.pth"),
                os.path.join(args.checkpoints_root, source_student_run, "gaussian_prior_latest.pth"),
            ])
        auto_candidates.append(os.path.join(args.checkpoints_root, source_teacher_run, "gaussian_prior_latest.pth"))
        for cand in auto_candidates:
            if os.path.exists(cand):
                stage1_gaussian_path = cand
                break

    metric_names = ("acc", "f1", "auc", "bac", "sens", "spec")
    best_val_by_metric = {m: {"value": -1.0, "epoch": -1} for m in metric_names}
    best_test_by_metric = {m: {"value": -1.0, "epoch": -1} for m in metric_names}
    best_val_acc = -1.0
    best_test_acc = -1.0
    train_X = train_y = test_X = test_y = val_X = val_y = None

    stage2_enable_aas = bool(getattr(args, "stage2_enable_aas", False))
    stage2_aas_gamma = float(getattr(args, "stage2_aas_gamma", 1.0))
    stage2_aas_ema = float(getattr(args, "stage2_aas_ema", 0.5))
    stage2_aas_total_source = str(getattr(args, "stage2_aas_total_source", "base")).lower()
    if stage2_aas_total_source not in {"base", "current"}:
        raise ValueError("stage2_aas_total_source must be one of: base | current")
    stage2_aas_follow_base_mask = bool(getattr(args, "stage2_aas_follow_base_mask", True))
    stage2_aas_allow_override_counts = bool(getattr(args, "stage2_aas_allow_override_counts", False))
    stage2_aas_log_per_class = bool(getattr(args, "stage2_aas_log_per_class", True))
    aas_next_class_sizes = None
    aas_prev_applied_sizes = None
    if stage2_enable_aas and class_sizes_override is not None and not stage2_aas_allow_override_counts:
        _write_local_log(
            log_f,
            "AAS disabled because stage2_virtual_counts_path is provided and stage2_aas_allow_override_counts=False.",
        )
        stage2_enable_aas = False
    _write_local_log(
        log_f,
        (
            f"stage2_enable_aas={stage2_enable_aas}, stage2_aas_gamma={stage2_aas_gamma}, "
            f"stage2_aas_ema={stage2_aas_ema}, stage2_aas_total_source={stage2_aas_total_source}, "
            f"stage2_aas_follow_base_mask={stage2_aas_follow_base_mask}"
        ),
    )

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
        if epoch == 0 and gaussian_stats is None:
            if prioritize_stage1_gaussian and stage1_gaussian_path:
                try:
                    gaussian_stats = _load_stage1_gaussian_stats(stage1_gaussian_path, n_classes, feature_dim)
                    stage1_gaussian_loaded = True
                    gaussian_source = "stage1"
                    _write_local_log(log_f, f"gaussian source=stage1, loaded: {stage1_gaussian_path}")
                except Exception as e:
                    _write_local_log(log_f, f"stage1 gaussian priority skipped: {e}")
                    gaussian_stats = None

            if gaussian_stats is None and use_saved_gaussian and os.path.exists(stats_path):
                gaussian_stats = _load_gaussian_stats(stats_path)
                gaussian_source = "saved"
                _write_local_log(log_f, f"gaussian source=saved, loaded: {stats_path}")

            if gaussian_stats is None and use_stage1_gaussian_init and stage1_gaussian_path:
                try:
                    gaussian_stats = _load_stage1_gaussian_stats(stage1_gaussian_path, n_classes, feature_dim)
                    stage1_gaussian_loaded = True
                    gaussian_source = "stage1"
                    _write_local_log(log_f, f"stage1 gaussian init loaded: {stage1_gaussian_path}")
                except Exception as e:
                    _write_local_log(log_f, f"stage1 gaussian init skipped: {e}")
                    gaussian_stats = None

        allow_refit = not stage1_gaussian_loaded or refit_after_stage1_gaussian
        if need_refit_gaussian and allow_refit:
            gaussian_stats = fit_class_gaussians(
                train_X,
                train_y,
                n_classes,
                covariance_type=getattr(args, "stage2_gaussian_covariance", "diag"),
                var_floor=float(getattr(args, "stage2_gaussian_var_floor", 1e-4)),
                full_min_samples=int(getattr(args, "stage2_gaussian_full_min_samples", 32)),
                full_shrinkage=float(getattr(args, "stage2_gaussian_full_shrinkage", 0.1)),
                calib_enable=bool(getattr(args, "stage2_gaussian_calib_enable", False)),
                calib_tau=float(getattr(args, "stage2_gaussian_calib_tau", 100.0)),
                calib_head_min_count=int(getattr(args, "stage2_gaussian_calib_head_min_count", 0)),
            )
            gaussian_source = "fit"
            if save_gaussian:
                _save_gaussian_stats(stats_path, gaussian_stats)
            calib_info = gaussian_stats.get("calibration", {})
            if bool(calib_info.get("enabled", False)):
                alpha = calib_info.get("alpha", None)
                alpha_min = float(np.min(alpha)) if alpha is not None else 0.0
                alpha_max = float(np.max(alpha)) if alpha is not None else 0.0
                _write_local_log(
                    log_f,
                    (
                        "gaussian calibration: enabled=True, tau={:.3f}, head_min_count={}, "
                        "prior_source={}, alpha_min={:.4f}, alpha_max={:.4f}"
                    ).format(
                        float(calib_info.get("tau", 0.0)),
                        int(calib_info.get("head_min_count", 0)),
                        str(calib_info.get("prior_source", "none")),
                        alpha_min,
                        alpha_max,
                    ),
                )

        train_X_for_cls = train_X
        train_y_for_cls = train_y
        virtual_mode = getattr(args, "stage2_virtual_mode", "uniform")
        virtual_enabled = bool(getattr(args, "stage2_virtual_enable", True))
        class_sizes = None
        base_class_sizes = None
        if virtual_enabled:
            if class_sizes_override is not None:
                base_class_sizes = class_sizes_override.copy()
            else:
                base_class_sizes = compute_virtual_class_sizes(
                    train_y,
                    n_classes,
                    uniform_size=args.virtual_size,
                    mode=virtual_mode,
                    tail_scale=float(getattr(args, "stage2_tail_scale", 1.0)),
                    tail_target=getattr(args, "stage2_tail_target", "max"),
                    min_size=int(getattr(args, "stage2_virtual_min_per_class", 0)),
                    max_size=int(getattr(args, "stage2_virtual_max_per_class", -1)),
                )
            if stage2_enable_aas and aas_next_class_sizes is not None:
                class_sizes = aas_next_class_sizes.copy()
                aas_source = "aas_feedback"
            else:
                class_sizes = base_class_sizes.copy()
                aas_source = "base"
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
                "gaussian_source": gaussian_source,
                "class_size_source": aas_source,
            })
        else:
            _log_metrics_local(log_f, f"epoch {epoch} virtual", {
                "virtual_total": 0,
                "train_total": int(len(train_X_for_cls)),
                "merge_real": 1,
                "gaussian_source": gaussian_source,
                "class_size_source": "none",
            })

        arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
            train_X_for_cls, train_y_for_cls, test_X, test_y, val_X, val_y, args.stage2_batch_size
        )

        if stage2_use_class_weight:
            labels_for_weight = train_y if stage2_class_weight_source == "real" else train_y_for_cls
            class_weights_np, class_counts_np = _build_class_weights_np(
                labels_for_weight,
                n_classes,
                power=float(getattr(args, "class_weight_power", 1.0)),
                min_weight=float(getattr(args, "class_weight_min", 0.0)),
                max_weight=float(getattr(args, "class_weight_max", -1.0)),
                eps=float(getattr(args, "class_weight_eps", 1e-6)),
            )
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=args.device)
            classifier_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            signature = (
                tuple(class_counts_np.tolist()),
                tuple(np.round(class_weights_np, 6).tolist()),
                stage2_class_weight_source,
            )
            if signature != last_class_weight_signature:
                _write_local_log(
                    log_f,
                    "stage2 class_weight: source={}, counts={}, weights={}".format(
                        stage2_class_weight_source,
                        class_counts_np.tolist(),
                        [round(float(x), 6) for x in class_weights_np.tolist()],
                    ),
                )
                last_class_weight_signature = signature
        else:
            classifier_criterion = torch.nn.CrossEntropyLoss()

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
        test_metrics = {
            "acc": test_acc, "f1": test_f1, "auc": test_auc, "bac": test_bac, "sens": test_sens, "spec": test_spec
        }
        val_metrics = {
            "acc": val_acc, "f1": val_f1, "auc": val_auc, "bac": val_bac, "sens": val_sens, "spec": val_spec
        }
        _log_metrics_local(log_f, f"epoch {epoch} test", test_metrics)
        _log_metrics_local(log_f, f"epoch {epoch} val", val_metrics)

        for m in metric_names:
            if val_metrics[m] > best_val_by_metric[m]["value"]:
                best_val_by_metric[m]["value"] = float(val_metrics[m])
                best_val_by_metric[m]["epoch"] = int(epoch)
            if test_metrics[m] > best_test_by_metric[m]["value"]:
                best_test_by_metric[m]["value"] = float(test_metrics[m])
                best_test_by_metric[m]["epoch"] = int(epoch)

        if virtual_enabled and stage2_enable_aas and base_class_sizes is not None:
            per_class_acc, per_class_total = _per_class_accuracy(
                classifier_model,
                arr_val_loader,
                args.device,
                n_classes,
            )
            total_target = int(base_class_sizes.sum())
            if stage2_aas_total_source == "current" and class_sizes is not None:
                total_target = int(class_sizes.sum())

            aas_next_class_sizes, aas_hardness, aas_ema_debug = _compute_aas_class_sizes(
                base_class_sizes=base_class_sizes,
                per_class_acc=per_class_acc,
                total_target=total_target,
                gamma=stage2_aas_gamma,
                follow_base_mask=stage2_aas_follow_base_mask,
                min_size=int(getattr(args, "stage2_virtual_min_per_class", 0)),
                max_size=int(getattr(args, "stage2_virtual_max_per_class", -1)),
                max_virtual_ratio=float(getattr(args, "stage2_virtual_max_ratio", -1.0)),
                real_total=int(len(train_y)),
                ema_prev=aas_prev_applied_sizes,
                ema_momentum=stage2_aas_ema,
            )
            aas_prev_applied_sizes = aas_next_class_sizes.copy()
            hardest_cls = int(np.argmax(aas_hardness))
            _log_metrics_local(
                log_f,
                f"epoch {epoch} aas",
                {
                    "hardest_cls": hardest_cls,
                    "hardest_score": float(aas_hardness[hardest_cls]),
                    "next_virtual_total": int(aas_next_class_sizes.sum()),
                    "gamma": float(stage2_aas_gamma),
                    "ema_momentum": float(aas_ema_debug["ema_momentum"]),
                    "ema_active": int(aas_ema_debug["ema_active"]),
                    "ema_index": float(aas_ema_debug["ema_index"]),
                },
            )
            print(
                "epoch {} aas: gamma={:.3f}, ema_momentum={:.3f}, ema_active={}, ema_index={:.4f}".format(
                    epoch,
                    float(stage2_aas_gamma),
                    float(aas_ema_debug["ema_momentum"]),
                    int(aas_ema_debug["ema_active"]),
                    float(aas_ema_debug["ema_index"]),
                )
            )
            if stage2_aas_log_per_class:
                _write_local_log(
                    log_f,
                    "epoch {} aas_per_class_acc={}".format(
                        epoch,
                        [round(float(x), 4) for x in per_class_acc.tolist()],
                    ),
                )
                _write_local_log(
                    log_f,
                    "epoch {} aas_per_class_val_count={}".format(
                        epoch,
                        [int(x) for x in per_class_total.tolist()],
                    ),
                )
                _write_local_log(
                    log_f,
                    "epoch {} aas_raw_class_sizes={}".format(
                        epoch,
                        [int(x) for x in aas_ema_debug["raw_sizes"].tolist()],
                    ),
                )
                _write_local_log(
                    log_f,
                    "epoch {} aas_next_class_sizes={}".format(
                        epoch,
                        [int(x) for x in aas_next_class_sizes.tolist()],
                    ),
                )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_classifier.pth"))
            torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_feature_extractor.pth"))
            if feature_source == "resnet":
                torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_backbone.pth"))
            elif feature_source == "lite":
                torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_litevae.pth"))
            else:
                torch.save(feature_model.state_dict(), os.path.join(args.checkpoints, "stage2_best_vavae.pth"))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(
            f"Epoch [{epoch}/{args.stage2_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {acc_epoch / len(arr_train_loader)}"
        )
        if log_f is not None:
            _write_local_log(log_f, f"Epoch [{epoch}/{args.stage2_epochs}] Loss={loss_epoch / len(arr_train_loader):.6f} Acc={acc_epoch / len(arr_train_loader):.6f}")

    summary_val = ", ".join(
        [f"{m}={best_val_by_metric[m]['value']:.6f}@epoch{best_val_by_metric[m]['epoch']}" for m in metric_names]
    )
    summary_test = ", ".join(
        [f"{m}={best_test_by_metric[m]['value']:.6f}@epoch{best_test_by_metric[m]['epoch']}" for m in metric_names]
    )
    print(f"Best validation metrics: {summary_val}")
    print(f"Best test metrics: {summary_test}")

    if log_f is not None:
        _write_local_log(log_f, f"Best val acc={best_val_acc:.6f}, best test acc={best_test_acc:.6f}")
        _write_local_log(log_f, f"Best validation metrics: {summary_val}")
        _write_local_log(log_f, f"Best test metrics: {summary_test}")
        log_f.close()
