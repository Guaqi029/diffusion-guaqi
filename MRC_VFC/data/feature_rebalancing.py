import numpy as np


def compute_virtual_class_sizes(
    y,
    class_num,
    uniform_size=1000,
    mode="uniform",
    tail_scale=1.0,
    tail_target="max",
    min_size=0,
    max_size=-1,
):
    """
    Compute generated sample size per class.

    mode = "uniform": keep old behavior, each class has `uniform_size` synthetic samples.
    mode = "tail_to_target": only generate for under-represented classes to approach target count.
    """
    if mode == "uniform":
        sizes = np.full(class_num, int(uniform_size), dtype=np.int64)
    elif mode == "tail_to_target":
        counts = np.bincount(y, minlength=class_num).astype(np.int64)
        if tail_target == "max":
            target = int(np.max(counts))
        elif tail_target == "median":
            target = int(np.median(counts))
        elif tail_target == "mean":
            target = int(np.mean(counts))
        else:
            raise ValueError(f"Unsupported tail_target: {tail_target}")

        deficits = np.maximum(target - counts, 0)
        sizes = np.round(deficits * float(tail_scale)).astype(np.int64)
        if min_size > 0:
            sizes = np.maximum(sizes, int(min_size))
        if max_size > 0:
            sizes = np.minimum(sizes, int(max_size))
    else:
        raise ValueError(f"Unsupported virtual size mode: {mode}")

    return sizes


def fit_class_gaussians(
    x,
    y,
    class_num,
    covariance_type="full",
    var_floor=1e-4,
    full_min_samples=32,
    full_shrinkage=0.1,
    calib_enable=False,
    calib_tau=100.0,
    calib_head_min_count=0,
):
    """
    Fit class-conditional Gaussian stats from feature matrix x.
    Returns a dict with arrays to simplify save/load with np.savez.
    """
    assert len(set(y)) == class_num, "Training set must include samples from all classes"
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    feat_dim = x.shape[1]

    means = np.zeros((class_num, feat_dim), dtype=np.float32)
    counts = np.bincount(y, minlength=class_num).astype(np.int64)
    if covariance_type == "diag":
        cov_diag = np.zeros((class_num, feat_dim), dtype=np.float32)
        cov_full = None
    elif covariance_type == "full":
        cov_diag = None
        cov_full = np.zeros((class_num, feat_dim, feat_dim), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    global_var = np.var(x, axis=0).astype(np.float32)
    global_var = np.maximum(global_var, var_floor)

    for cls in range(class_num):
        class_samples = x[y == cls]
        mean = np.mean(class_samples, axis=0).astype(np.float32)
        means[cls] = mean

        if covariance_type == "diag":
            if len(class_samples) > 1:
                var = np.var(class_samples, axis=0).astype(np.float32)
            else:
                var = global_var.copy()
            cov_diag[cls] = np.maximum(var, var_floor)
        else:
            n = len(class_samples)
            class_var = np.var(class_samples, axis=0).astype(np.float32) if n > 1 else global_var.copy()
            class_var = np.maximum(class_var, var_floor)
            diag_cov = np.diag(class_var)

            # For extreme long-tail classes (n << d), full covariance is unstable.
            # Fall back to diagonal covariance in matrix form to avoid singular sampling.
            if n <= max(2, feat_dim) or n < int(full_min_samples):
                covariance = diag_cov
            else:
                normed = class_samples - mean
                empirical = np.matmul(normed.T, normed) / (n - 1)
                shrink = float(np.clip(full_shrinkage, 0.0, 1.0))
                covariance = (1.0 - shrink) * empirical + shrink * diag_cov

            covariance = covariance + np.eye(feat_dim, dtype=np.float32) * var_floor
            cov_full[cls] = covariance.astype(np.float32)

    calibration = {
        "enabled": bool(calib_enable),
        "tau": float(calib_tau),
        "head_min_count": int(calib_head_min_count) if int(calib_head_min_count) > 0 else int(max(1, calib_tau)),
        "prior_source": "none",
    }
    if bool(calib_enable):
        tau = float(max(calib_tau, 1.0))
        head_min_count = int(calib_head_min_count)
        if head_min_count <= 0:
            head_min_count = int(round(tau))
        head_mask = counts >= head_min_count
        if covariance_type == "diag":
            prior = np.mean(cov_diag[head_mask], axis=0) if np.any(head_mask) else np.mean(cov_diag, axis=0)
            prior = np.maximum(prior.astype(np.float32), var_floor)
            alpha = np.clip(counts.astype(np.float32) / tau, 0.0, 1.0)
            cov_diag = alpha[:, None] * cov_diag + (1.0 - alpha)[:, None] * prior[None, :]
            cov_diag = np.maximum(cov_diag.astype(np.float32), var_floor)
        else:
            prior = np.mean(cov_full[head_mask], axis=0) if np.any(head_mask) else np.mean(cov_full, axis=0)
            prior = prior.astype(np.float32)
            alpha = np.clip(counts.astype(np.float32) / tau, 0.0, 1.0)
            for cls in range(class_num):
                a = float(alpha[cls])
                cov_full[cls] = (
                    a * cov_full[cls].astype(np.float32) + (1.0 - a) * prior
                ).astype(np.float32)
                # keep diagonal numerically safe after calibration
                diag = np.diag(cov_full[cls]).copy()
                diag = np.maximum(diag, var_floor)
                np.fill_diagonal(cov_full[cls], diag)

        calibration["prior_source"] = "head" if np.any(head_mask) else "all_fallback"
        calibration["alpha"] = alpha.astype(np.float32)

    stats = {
        "covariance_type": covariance_type,
        "means": means,
        "var_floor": float(var_floor),
        "class_counts": counts,
        "calibration": calibration,
    }
    if cov_diag is not None:
        stats["cov_diag"] = cov_diag
    if cov_full is not None:
        stats["cov_full"] = cov_full
    return stats


def sample_virtual_representations(stats, class_sizes, rng=None):
    """
    Sample synthetic features from pre-fitted class-conditional Gaussian stats.
    """
    if rng is None:
        rng = np.random.default_rng()

    covariance_type = stats["covariance_type"]
    means = stats["means"]
    class_sizes = np.asarray(class_sizes, dtype=np.int64)
    class_num = means.shape[0]

    virtual_samples = []
    virtual_labels = []
    for cls in range(class_num):
        n = int(class_sizes[cls])
        if n <= 0:
            continue
        mean = means[cls]
        if covariance_type == "diag":
            std = np.sqrt(stats["cov_diag"][cls])
            gaussian_samples = mean + rng.standard_normal((n, mean.shape[0])) * std
        elif covariance_type == "full":
            gaussian_samples = rng.multivariate_normal(mean, stats["cov_full"][cls], size=n)
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        gaussian_labels = np.full((n,), cls, dtype=np.int64)
        virtual_samples.append(gaussian_samples.astype(np.float32))
        virtual_labels.append(gaussian_labels)

    if not virtual_samples:
        feat_dim = means.shape[1]
        return np.zeros((0, feat_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.concatenate(virtual_samples, axis=0), np.concatenate(virtual_labels, axis=0)


def virtual_representations(x, y, class_num, size=1000):
    """
    Backward-compatible API used by the original Stage2 code.
    """
    stats = fit_class_gaussians(x, y, class_num, covariance_type="diag", var_floor=1e-4)
    class_sizes = np.full(class_num, int(size), dtype=np.int64)
    return sample_virtual_representations(stats, class_sizes)
