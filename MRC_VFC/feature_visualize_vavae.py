import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ISICDataset, Transforms
from models import VAVAEStudentVAE
from utils.lt_split import resolve_isic2019lt_split_paths, isic2019lt_split_files_exist
from utils.yaml_config_hook import yaml_config_hook


REPO_ROOT = Path(__file__).resolve().parent


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def _parse_args():
    yaml_config = yaml_config_hook(str(REPO_ROOT / "config" / "configs.yaml"))

    parser = argparse.ArgumentParser(
        description="Visualize Stage1 VA-VAE student features with t-SNE / UMAP."
    )
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=_str2bool if isinstance(v, bool) else type(v))

    parser.add_argument("--checkpoints_root", type=str, default="", help="Overrides config checkpoints root.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--split_dir",
        type=str,
        default="",
        help="Optional directory containing training.csv / validation.csv / testing.csv.",
    )
    parser.add_argument("--feature_mode", type=str, default="mu", choices=["mu", "z"])
    parser.add_argument("--viz_batch_size", type=int, default=64)
    parser.add_argument("--max_samples_total", type=int, default=3000)
    parser.add_argument("--max_samples_per_class", type=int, default=300)
    parser.add_argument("--normalize_features", type=_str2bool, default=True)
    parser.add_argument("--methods", type=str, default="tsne,umap")
    parser.add_argument("--pca_dim_before_reduce", type=int, default=32)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=25)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--point_alpha", type=float, default=0.75)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--force_cpu", type=_str2bool, default=False)
    return parser.parse_args()


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("This script requires matplotlib. Install it with: pip install matplotlib") from exc
    return plt


def _set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _auto_find_checkpoint(run_dir, preferred_name, epoch_prefix):
    if preferred_name:
        preferred = os.path.join(run_dir, preferred_name)
        if os.path.exists(preferred):
            return preferred
    candidates = sorted(Path(run_dir).glob(f"{epoch_prefix}_epoch_*_.pth"))
    if not candidates:
        return ""
    return str(candidates[-1])


def _resolve_repo_path(path):
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def _resolve_split_csv(args):
    if str(getattr(args, "dataset", "")).strip() == "ISIC2019LT":
        if args.split_dir:
            split_dir = Path(_resolve_repo_path(args.split_dir))
            if split_dir.name.startswith("shared_eval_seed"):
                args.csv_file_train = str(split_dir / f"training_if{int(args.imbalance_factor)}.csv")
                args.csv_file_val = str(split_dir / "validation.csv")
                args.csv_file_test = str(split_dir / "testing.csv")
            else:
                args.lt_split_root = str(split_dir)
                resolve_isic2019lt_split_paths(args)
        else:
            resolve_isic2019lt_split_paths(args)

        if not isic2019lt_split_files_exist(args):
            raise FileNotFoundError(
                "ISIC2019LT split files not found. Expected training_if{factor}.csv, validation.csv, testing.csv."
            )
        mapping = {
            "train": _resolve_repo_path(args.csv_file_train),
            "val": _resolve_repo_path(args.csv_file_val),
            "test": _resolve_repo_path(args.csv_file_test),
        }
        return mapping[args.split]

    if args.split_dir:
        split_dir = Path(_resolve_repo_path(args.split_dir))
        mapping = {
            "train": split_dir / "training.csv",
            "val": split_dir / "validation.csv",
            "test": split_dir / "testing.csv",
        }
        return str(mapping[args.split])
    mapping = {
        "train": _resolve_repo_path(args.csv_file_train),
        "val": _resolve_repo_path(args.csv_file_val),
        "test": _resolve_repo_path(args.csv_file_test),
    }
    return mapping[args.split]


def _resolve_device(args):
    if args.force_cpu:
        return torch.device("cpu")
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(args, device):
    model = VAVAEStudentVAE(
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
    ).to(device)

    checkpoints_root = _resolve_repo_path(args.checkpoints_root if args.checkpoints_root else args.checkpoints)
    if not args.run_name:
        raise ValueError("--run_name is required.")
    run_dir = os.path.join(checkpoints_root, args.run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpt_path = args.lite_vae_resume_path
    if ckpt_path:
        candidate = os.path.join(run_dir, ckpt_path)
        if os.path.exists(candidate):
            ckpt_path = candidate
        else:
            ckpt_path = _resolve_repo_path(ckpt_path)
    if not ckpt_path or not os.path.exists(ckpt_path):
        ckpt_path = _auto_find_checkpoint(run_dir, "litevae_latest.pth", "litevae")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find litevae checkpoint under: {run_dir}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, run_dir, ckpt_path


def _collect_features(loader, model, device, feature_mode):
    feats = []
    labels = []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            mu, logvar, z, _ = model(images)
            feat = z if feature_mode == "z" else mu
            feats.append(feat.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    features = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    return features, labels


def _subsample(features, labels, max_total, max_per_class, seed):
    n = features.shape[0]
    if (max_total is None or max_total <= 0 or n <= max_total) and (max_per_class is None or max_per_class <= 0):
        return features, labels, np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    keep = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if max_per_class is not None and max_per_class > 0 and cls_idx.size > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        keep.append(np.sort(cls_idx))
    keep_idx = np.concatenate(keep, axis=0) if keep else np.arange(n, dtype=np.int64)

    if max_total is not None and max_total > 0 and keep_idx.size > max_total:
        chosen = []
        remaining = max_total
        classes = np.unique(labels[keep_idx])
        base = max(1, max_total // max(1, len(classes)))
        for cls in classes:
            cls_idx = keep_idx[labels[keep_idx] == cls]
            take = min(base, cls_idx.size, remaining)
            if take > 0:
                chosen.append(rng.choice(cls_idx, size=take, replace=False))
                remaining -= take
        chosen_idx = np.concatenate(chosen, axis=0) if chosen else np.empty((0,), dtype=np.int64)
        leftover_pool = np.setdiff1d(keep_idx, chosen_idx, assume_unique=False)
        if remaining > 0 and leftover_pool.size > 0:
            extra = rng.choice(leftover_pool, size=min(remaining, leftover_pool.size), replace=False)
            chosen_idx = np.concatenate([chosen_idx, extra], axis=0)
        keep_idx = np.sort(chosen_idx)
    else:
        keep_idx = np.sort(keep_idx)

    return features[keep_idx], labels[keep_idx], keep_idx


def _prepare_features(features, normalize_features):
    out = features.astype(np.float32, copy=True)
    if normalize_features:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.clip(norms, 1e-12, None)
    return out


def _pca_reduce(features, out_dim):
    if out_dim <= 0 or features.shape[1] <= out_dim:
        return features
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:out_dim].T


def _run_tsne(features, seed, perplexity):
    from sklearn.manifold import TSNE

    p = min(float(perplexity), max(2.0, features.shape[0] - 1.0))
    reducer = TSNE(
        n_components=2,
        perplexity=p,
        init="pca",
        random_state=int(seed),
        learning_rate="auto",
    )
    return reducer.fit_transform(features)


def _run_umap(features, seed, n_neighbors, min_dist):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP requires 'umap-learn'. Install it with: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
    )
    return reducer.fit_transform(features)


def _build_colors(n_classes):
    plt = _get_plt()
    if n_classes <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n_classes)]
    cmap = plt.get_cmap("gist_ncar")
    return [cmap(i / max(1, n_classes - 1)) for i in range(n_classes)]


def _plot_embedding(embedding, labels, class_names, title, out_path, point_size, point_alpha):
    plt = _get_plt()
    uniq = np.unique(labels)
    colors = _build_colors(len(uniq))
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(uniq):
        mask = labels == cls
        name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=float(point_size),
            alpha=float(point_alpha),
            c=[colors[i]],
            label=name,
            edgecolors="none",
        )
    plt.title(title)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    args = _parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args)

    split_csv = _resolve_split_csv(args)
    transforms = Transforms(size=args.image_size)
    dataset = ISICDataset(_resolve_repo_path(args.data_path), split_csv, transform=transforms.test_transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.viz_batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
    )

    model, run_dir, ckpt_path = _build_model(args, device)
    raw_features, raw_labels = _collect_features(loader, model, device, args.feature_mode)
    features, labels, keep_idx = _subsample(
        raw_features,
        raw_labels,
        args.max_samples_total,
        args.max_samples_per_class,
        args.seed,
    )
    features = _prepare_features(features, args.normalize_features)
    features_for_reduce = _pca_reduce(features, int(args.pca_dim_before_reduce))

    requested_methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    if not requested_methods:
        requested_methods = ["tsne", "umap"]

    if args.out_dir:
        out_dir = Path(_resolve_repo_path(args.out_dir))
    else:
        out_dir = Path(run_dir) / "feature_vis" / f"{args.split}_{args.feature_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "feature_mode": args.feature_mode,
        "checkpoint": ckpt_path,
        "raw_feature_shape": list(raw_features.shape),
        "viz_feature_shape": list(features.shape),
        "split_csv": split_csv,
        "normalize_features": bool(args.normalize_features),
        "pca_dim_before_reduce": int(args.pca_dim_before_reduce),
        "methods": requested_methods,
        "class_names": [str(x) for x in dataset.class_names],
        "kept_indices_count": int(keep_idx.size),
    }

    np.savez_compressed(
        out_dir / "features_labels.npz",
        features=features,
        labels=labels,
        class_names=np.array(dataset.class_names, dtype=object),
        kept_indices=keep_idx,
    )

    for method in requested_methods:
        if method == "tsne":
            embedding = _run_tsne(features_for_reduce, args.seed, args.tsne_perplexity)
        elif method == "umap":
            embedding = _run_umap(features_for_reduce, args.seed, args.umap_n_neighbors, args.umap_min_dist)
        elif method == "pca":
            embedding = _pca_reduce(features, 2)
        else:
            raise ValueError(f"Unsupported method: {method}")

        np.savez_compressed(
            out_dir / f"{method}_embedding.npz",
            embedding=embedding,
            labels=labels,
            class_names=np.array(dataset.class_names, dtype=object),
        )
        _plot_embedding(
            embedding,
            labels,
            dataset.class_names,
            title=f"{method.upper()} | split={args.split} | feature={args.feature_mode}",
            out_path=out_dir / f"{method}_{args.split}_{args.feature_mode}.png",
            point_size=args.point_size,
            point_alpha=args.point_alpha,
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] checkpoint={ckpt_path}")
    print(f"[done] split_csv={split_csv}")
    print(f"[done] raw_features={raw_features.shape}, viz_features={features.shape}")
    print(f"[done] outputs={out_dir}")


if __name__ == "__main__":
    main()
