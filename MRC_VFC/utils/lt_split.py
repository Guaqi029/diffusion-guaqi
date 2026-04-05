import os


def resolve_isic2019lt_split_paths(args):
    """Map ISIC2019LT csv paths to shared-eval split files for the given factor."""
    if getattr(args, "dataset", "") != "ISIC2019LT":
        return ""
    if not bool(getattr(args, "lt_split_use_factor_seed_dir", True)):
        return ""

    split_root = str(getattr(args, "lt_split_root", "")).strip()
    if not split_root:
        split_root = os.path.dirname(str(getattr(args, "csv_file_train", "")).strip())
    if not split_root:
        split_root = "./split/ISIC2019LT"

    imbalance_factor = int(getattr(args, "imbalance_factor", 500))
    seed = int(getattr(args, "seed", 42))
    split_dir = os.path.join(split_root, f"shared_eval_seed{seed}")

    args.csv_file_train = os.path.join(split_dir, f"training_if{imbalance_factor}.csv")
    args.csv_file_val = os.path.join(split_dir, "validation.csv")
    args.csv_file_test = os.path.join(split_dir, "testing.csv")
    return split_dir


def isic2019lt_split_files_exist(args):
    paths = (
        getattr(args, "csv_file_train", ""),
        getattr(args, "csv_file_val", ""),
        getattr(args, "csv_file_test", ""),
    )
    return all(path and os.path.exists(path) for path in paths)
