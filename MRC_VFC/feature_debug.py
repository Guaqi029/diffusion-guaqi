import os
import glob
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models import CreateModel, LiteVAE
from data import ISICDataset, Transforms
from utils.yaml_config_hook import yaml_config_hook


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


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
    print("[CUDA alloc] Removed unsupported option 'expandable_segments'.")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _auto_find_checkpoint(run_dir, preferred_name, epoch_prefix):
    if preferred_name:
        cand = os.path.join(run_dir, preferred_name)
        if os.path.exists(cand):
            return cand
    files = glob.glob(os.path.join(run_dir, f"{epoch_prefix}_epoch_*_.pth"))
    if not files:
        return ""
    files = sorted(files, key=lambda p: _parse_epoch_from_name(p, epoch_prefix))
    return files[-1]


def _resolve_path(path, run_dir):
    if not path:
        return ""
    cands = []
    if os.path.isabs(path):
        cands.append(path)
    else:
        cands.append(os.path.join(run_dir, path))
        cands.append(path)
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def _build_kd_proj(args, in_dim, out_dim, device):
    if not _str2bool(getattr(args, "kd_feat_project", True)):
        return None
    if _str2bool(getattr(args, "kd_feat_project_mlp", False)):
        hidden_dim = int(getattr(args, "kd_feat_proj_hidden_dim", 0))
        if hidden_dim <= 0:
            hidden_dim = max(in_dim, out_dim)
        depth = int(getattr(args, "kd_feat_proj_depth", 2))
        dropout = float(getattr(args, "kd_feat_proj_dropout", 0.0))
        if depth <= 1:
            return torch.nn.Linear(in_dim, out_dim).to(device)
        layers = []
        cur = in_dim
        for i in range(depth - 1):
            is_last = i == (depth - 2)
            nxt = out_dim if is_last else hidden_dim
            layers.append(torch.nn.Linear(cur, nxt))
            if not is_last:
                layers.append(torch.nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(torch.nn.Dropout(p=dropout))
            cur = nxt
        return torch.nn.Sequential(*layers).to(device)
    return torch.nn.Linear(in_dim, out_dim).to(device)


def _reduce_2d(feats, method="tsne", seed=42, perplexity=30):
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            p = min(float(perplexity), max(2.0, feats.shape[0] - 1.0))
            tsne = TSNE(
                n_components=2,
                perplexity=p,
                init="pca",
                random_state=seed,
                learning_rate="auto",
            )
            return tsne.fit_transform(feats), "tsne"
        except Exception as e:
            print(f"[warn] TSNE failed ({e}), fallback to PCA.")
    x = feats - feats.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    out = x @ vh[:2].T
    return out, "pca"


def _plot_embedding(feat2d, labels, title, out_path):
    plt.figure(figsize=(8, 6))
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(uniq):
        mask = labels == cls
        plt.scatter(
            feat2d[mask, 0],
            feat2d[mask, 1],
            s=12,
            alpha=0.7,
            c=[cmap(i % 10)],
            label=f"class {int(cls)}",
        )
    plt.title(title)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_overlay(t2d, s2d, labels, out_path):
    plt.figure(figsize=(8, 6))
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(uniq):
        m = labels == cls
        color = cmap(i % 10)
        plt.scatter(t2d[m, 0], t2d[m, 1], s=14, alpha=0.55, c=[color], marker="o")
        plt.scatter(s2d[m, 0], s2d[m, 1], s=14, alpha=0.55, c=[color], marker="x")
    plt.title("Teacher(o) vs Student(x) in shared 2D")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _compute_gram(x):
    x = F.normalize(x, dim=1)
    g = torch.matmul(x, x.t())
    return g / max(1, x.size(1))


def _linear_cka(x, y, eps=1e-8):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = torch.matmul(x.t(), y)
    xtx = torch.matmul(x.t(), x)
    yty = torch.matmul(y.t(), y)
    hsic = torch.norm(xty, p="fro").pow(2)
    denom = torch.norm(xtx, p="fro") * torch.norm(yty, p="fro")
    return hsic / (denom + float(eps))


def _plot_gram(g, title, out_path):
    g = g.detach().cpu().numpy()
    plt.figure(figsize=(5, 4.5))
    plt.imshow(g, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return torch.clamp(img * std + mean, 0.0, 1.0)


def _save_recon(input_img, recon_img, out_path):
    n = input_img.size(0)
    show_in = _denorm(input_img)
    show_re = _denorm(recon_img)
    grid = make_grid(torch.cat([show_in, show_re], dim=0), nrow=n, padding=2)
    arr = grid.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(2.2 * n, 4.8))
    plt.imshow(arr)
    plt.axis("off")
    plt.title("Top: Input | Bottom: Lite Recon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _collect_features(loader, teacher, lite_vae, kd_proj, device, feature_mode="mu", max_samples=1200):
    teacher.eval()
    lite_vae.eval()
    if kd_proj is not None:
        kd_proj.eval()

    t_all = []
    s_all = []
    sp_all = []
    y_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ft, _ = teacher(x)
            mu, _, z, _ = lite_vae(x)
            fs = z if feature_mode == "z" else mu
            if kd_proj is not None:
                fsp = kd_proj(fs)
                sp_all.append(fsp.detach().cpu().numpy())

            t_all.append(ft.detach().cpu().numpy())
            s_all.append(fs.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())
            if sum(v.shape[0] for v in y_all) >= max_samples:
                break

    t_np = np.concatenate(t_all, axis=0)[:max_samples]
    s_np = np.concatenate(s_all, axis=0)[:max_samples]
    y_np = np.concatenate(y_all, axis=0)[:max_samples]
    sp_np = np.concatenate(sp_all, axis=0)[:max_samples] if len(sp_all) > 0 else None
    return t_np, s_np, sp_np, y_np


def main(args):
    _sanitize_cuda_alloc_conf()
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"[info] device={device}")

    checkpoints_root = args.checkpoints
    teacher_run = args.teacher_run_name if args.teacher_run_name else args.run_name
    student_run = args.student_run_name if args.student_run_name else args.run_name
    teacher_dir = os.path.join(checkpoints_root, teacher_run)
    student_dir = os.path.join(checkpoints_root, student_run)
    if not os.path.isdir(teacher_dir):
        raise FileNotFoundError(f"Teacher run dir not found: {teacher_dir}")
    if not os.path.isdir(student_dir):
        raise FileNotFoundError(f"Student run dir not found: {student_dir}")

    out_dir = args.out_dir
    if not out_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("log", "visual_debug", f"{student_run}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] output_dir={out_dir}")

    transforms = Transforms(size=args.image_size)
    split_csv = args.csv_file_val
    if args.split == "test":
        split_csv = args.csv_file_test
    elif args.split == "train":
        split_csv = args.csv_file_train
    dataset = ISICDataset(args.data_path, split_csv, transform=transforms.test_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.viz_batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    n_classes = dataset.n_class

    teacher = CreateModel(backbone=args.backbone, ema=False, out_features=n_classes, pretrained=False).to(device)
    if args.teacher_model_path:
        teacher_ckpt = _resolve_path(args.teacher_model_path, teacher_dir)
    else:
        if int(args.teacher_epoch) > 0:
            teacher_ckpt = os.path.join(teacher_dir, f"epoch_{int(args.teacher_epoch)}_.pth")
        else:
            teacher_ckpt = _auto_find_checkpoint(teacher_dir, "", "epoch")
    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise FileNotFoundError("Teacher checkpoint not found.")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    print(f"[info] loaded teacher ckpt: {teacher_ckpt}")

    lite_vae = LiteVAE(
        image_size=args.image_size,
        in_channels=3,
        base_channels=args.lite_vae_base_channels,
        latent_dim=args.lite_vae_latent_dim,
        dwt_levels=args.lite_vae_dwt_levels,
        variant=args.lite_vae_variant,
    ).to(device)
    lite_ckpt = _resolve_path(args.lite_vae_resume_path, student_dir)
    if not lite_ckpt:
        lite_ckpt = _auto_find_checkpoint(student_dir, "litevae_latest.pth", "litevae")
    if not lite_ckpt:
        raise FileNotFoundError("LiteVAE checkpoint not found.")
    lite_vae.load_state_dict(torch.load(lite_ckpt, map_location=device))
    print(f"[info] loaded lite ckpt: {lite_ckpt}")

    kd_proj = None
    proj_ckpt = _resolve_path(args.kd_feat_proj_resume_path, student_dir)
    if not proj_ckpt:
        proj_ckpt = _auto_find_checkpoint(student_dir, "kd_feat_proj_latest.pth", "kd_feat_proj")
    if proj_ckpt:
        kd_proj = _build_kd_proj(args, args.lite_vae_latent_dim, teacher.n_features, device)
        kd_proj.load_state_dict(torch.load(proj_ckpt, map_location=device))
        print(f"[info] loaded kd proj ckpt: {proj_ckpt}")
    else:
        print("[warn] no kd_feat_proj checkpoint found; projected-student plots will be skipped.")

    t_np, s_np, sp_np, y_np = _collect_features(
        loader,
        teacher,
        lite_vae,
        kd_proj,
        device,
        feature_mode=args.lite_feature_mode,
        max_samples=args.max_samples,
    )
    print(f"[info] features collected: teacher={t_np.shape}, student={s_np.shape}, labels={y_np.shape}")
    if sp_np is not None:
        print(f"[info] projected student={sp_np.shape}")

    emb_t, used_t = _reduce_2d(t_np, method=args.reduce_method, seed=args.seed, perplexity=args.tsne_perplexity)
    emb_s, used_s = _reduce_2d(s_np, method=args.reduce_method, seed=args.seed, perplexity=args.tsne_perplexity)
    _plot_embedding(emb_t, y_np, f"Teacher Feature ({used_t})", os.path.join(out_dir, "tsne_teacher.png"))
    _plot_embedding(emb_s, y_np, f"Student Feature ({used_s})", os.path.join(out_dir, "tsne_student.png"))

    overlay_done = False
    if sp_np is not None and sp_np.shape[1] == t_np.shape[1]:
        both = np.concatenate([t_np, sp_np], axis=0)
        emb_both, used_b = _reduce_2d(
            both, method=args.reduce_method, seed=args.seed, perplexity=args.tsne_perplexity
        )
        n = t_np.shape[0]
        _plot_overlay(
            emb_both[:n],
            emb_both[n:],
            y_np,
            os.path.join(out_dir, "tsne_overlay_teacher_vs_student_proj.png"),
        )
        overlay_done = True
        print(f"[info] overlay saved (method={used_b})")
    else:
        print("[warn] skip overlay: projected student unavailable or dim mismatch.")

    # Gram/CKA on one batch
    batch = next(iter(loader))
    xg, yg = batch
    xg = xg[: args.gram_batch_size].to(device)
    with torch.no_grad():
        ftg, _ = teacher(xg)
        mug, _, zg, rg = lite_vae(xg)
        fsg = zg if args.lite_feature_mode == "z" else mug
        if kd_proj is not None:
            fsg_proj = kd_proj(fsg)
        else:
            fsg_proj = None

    target_student = fsg_proj if (fsg_proj is not None and _str2bool(args.use_proj_for_gram)) else fsg
    gram_t = _compute_gram(ftg)
    gram_s = _compute_gram(target_student)
    gram_mse = F.mse_loss(gram_t, gram_s).item()
    cka = _linear_cka(ftg, target_student).item()

    _plot_gram(gram_t, "Teacher Gram", os.path.join(out_dir, "gram_teacher.png"))
    _plot_gram(gram_s, "Student Gram", os.path.join(out_dir, "gram_student.png"))
    _plot_gram(torch.abs(gram_t - gram_s), "|Teacher-Student| Gram", os.path.join(out_dir, "gram_abs_diff.png"))

    # Reconstruction
    xr = xg[: args.recon_samples]
    rr = rg[: args.recon_samples]
    _save_recon(xr, rr, os.path.join(out_dir, "reconstruction.png"))

    summary = {
        "teacher_ckpt": teacher_ckpt,
        "lite_ckpt": lite_ckpt,
        "kd_proj_ckpt": proj_ckpt,
        "split": args.split,
        "max_samples": int(args.max_samples),
        "teacher_shape": list(t_np.shape),
        "student_shape": list(s_np.shape),
        "student_proj_shape": list(sp_np.shape) if sp_np is not None else None,
        "overlay_done": overlay_done,
        "gram_mse": float(gram_mse),
        "cka": float(cka),
        "lite_feature_mode": args.lite_feature_mode,
        "use_proj_for_gram": _str2bool(args.use_proj_for_gram),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] gram_mse={gram_mse:.6f}, cka={cka:.6f}")
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--viz_batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=1200)
    parser.add_argument("--gram_batch_size", type=int, default=64)
    parser.add_argument("--recon_samples", type=int, default=8)
    parser.add_argument("--reduce_method", type=str, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--lite_feature_mode", type=str, default="mu", choices=["mu", "z"])
    parser.add_argument("--use_proj_for_gram", type=_str2bool, default=True)
    parser.add_argument("--teacher_model_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--force_cpu", type=_str2bool, default=False)

    args = parser.parse_args()
    main(args)
