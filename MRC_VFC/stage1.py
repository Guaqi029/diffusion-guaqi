import os
import sys
import subprocess
import time
import random
import torch
import argparse
import torch.distributed as dist
from models import (
    CreateModel,
    LiteVAE,
    Linear,
    VAVAETeacherEncoder,
    VAVAEStudentVAE,
)
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data import ISICDataset, Transforms
from torch.utils.data import DataLoader
from train import trainEncoder
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from utils.lt_split import resolve_isic2019lt_split_paths, isic2019lt_split_files_exist

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


def _set_seed(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(gpu, args):
    wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    # Initialize wandb only on rank0 child process to avoid mp.spawn pickling issues.
    if rank == 0 and not args.debug:
        if wandb is None:
            raise ModuleNotFoundError(
                "wandb is not installed. Install it or run with --debug."
            )
        wandb.login(key="[Your wandb key here]")
        wandb_logger = wandb.init(
            project="MRC_VFC_on_%s" % args.dataset,
            notes="MICCAI 2023",
            tags=["MICCAI23", "Class imbalance", "Dermoscopy", "Representation Learning"],
            config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
        )

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    base_seed = int(args.seed)
    process_seed = base_seed + int(rank)
    _set_seed(process_seed, deterministic=True)
    if rank == 0:
        print(f"[Seed] base_seed={base_seed}, rank0_process_seed={process_seed}, cudnn_deterministic=True")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(process_seed)

    # training set
    transforms = Transforms(size=args.image_size)
    train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms)
    

    # set sampler for parallel training
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    if rank == 0:
        test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
        val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)
        eval_generator = torch.Generator()
        eval_generator.manual_seed(base_seed + 99991)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            worker_init_fn=_seed_worker,
            generator=eval_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            worker_init_fn=_seed_worker,
            generator=eval_generator,
        )
    else:
        test_loader = None
        val_loader = None

    loaders = (train_loader, val_loader, test_loader)

    num_class = train_dataset.n_class
    args.num_classes = num_class

    kd_teacher_source = str(getattr(args, "kd_teacher_source", "resnet")).lower()
    student_source = str(getattr(args, "student_source", "lite")).lower()
    if student_source not in ("lite", "vavae"):
        raise ValueError("student_source must be one of: lite | vavae")
    args.student_source = student_source
    show_teacher_metrics = bool(getattr(args, "show_teacher_metrics", False))
    skip_resnet_backbone = (
        args.kd_enable
        and args.kd_only
        and kd_teacher_source in ("lite", "vavae")
        and not show_teacher_metrics
    )

    # model init (resnet path can be skipped in pure KD-only lite/vavae runs)
    model = None
    ema_model = None
    if not skip_resnet_backbone:
        model = CreateModel(backbone=args.backbone, ema=False, out_features=num_class, pretrained=args.pretrained)
        ema_model = CreateModel(backbone=args.backbone, ema=True, out_features=num_class, pretrained=args.pretrained)
    elif rank == 0:
        print(f"[Init] Skip ResNet backbone creation (kd_teacher_source={kd_teacher_source}, kd_only=True).")
    def _get_map_location():
        if isinstance(args.device, torch.device):
            return args.device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")

    need_resnet_teacher = not (
        args.kd_enable
        and args.kd_only
        and kd_teacher_source in ("lite", "vavae")
    )
    need_teacher_reload = bool(args.reload) and need_resnet_teacher
    if args.lite_eval_only:
        need_teacher_reload = False

    if need_teacher_reload:
        if model is None:
            raise RuntimeError("need_teacher_reload=True but ResNet model is not initialized.")
        checkpoints_root = getattr(args, "checkpoints_root", os.path.dirname(args.checkpoints))
        teacher_run = args.teacher_run_name if args.teacher_run_name else os.path.basename(args.checkpoints)
        teacher_epoch = args.teacher_epoch if args.teacher_epoch > 0 else args.epochs
        teacher_ckpt_dir = os.path.join(checkpoints_root, teacher_run)
        model_fp = os.path.join(
            teacher_ckpt_dir, "epoch_{}_.pth".format(teacher_epoch)
        )
        model.load_state_dict(torch.load(model_fp, map_location=_get_map_location()))
    elif args.reload and rank == 0 and args.lite_eval_only:
        print("[Reload] Skipped teacher checkpoint loading (lite_eval_only=True).")
    elif args.reload and rank == 0 and not need_resnet_teacher:
        print(
            f"[Reload] Skipped ResNet teacher checkpoint loading "
            f"(kd_teacher_source={kd_teacher_source}, kd_only=True)."
        )

    if model is not None:
        model = model.to(args.device)
    if ema_model is not None:
        ema_model = ema_model.to(args.device)

    if args.kd_enable and args.kd_freeze_teacher and model is not None and ema_model is not None:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in ema_model.parameters():
            param.requires_grad_(False)
    lite_vae = None
    lite_classifier = None
    lite_vae_teacher = None
    lite_classifier_teacher = None
    vavae_teacher = None
    kd_feat_proj = None
    if args.kd_enable or args.lite_eval_enable:
        def _build_student_vae():
            if student_source == "vavae":
                student_latent_dim = int(
                    getattr(
                        args,
                        "vavae_student_latent_dim",
                        getattr(args, "vavae_teacher_latent_dim", 32),
                    )
                )
                student = VAVAEStudentVAE(
                    in_channels=3,
                    ch=int(getattr(args, "vavae_student_ch", getattr(args, "vavae_teacher_ch", 128))),
                    ch_mult=getattr(args, "vavae_student_ch_mult", getattr(args, "vavae_teacher_ch_mult", "1,1,2,2,4")),
                    num_res_blocks=int(
                        getattr(args, "vavae_student_num_res_blocks", getattr(args, "vavae_teacher_num_res_blocks", 2))
                    ),
                    z_channels=student_latent_dim,
                    attn_levels=getattr(
                        args,
                        "vavae_student_attn_levels",
                        getattr(args, "vavae_teacher_attn_levels", "4"),
                    ),
                    input_size=int(getattr(args, "vavae_student_input_size", args.image_size)),
                    resize_input=bool(getattr(args, "vavae_student_resize_input", False)),
                    pool=str(getattr(args, "vavae_student_pool", "avg")),
                    feature_from=str(getattr(args, "vavae_student_feature_from", "mu")),
                    enable_decoder=bool(getattr(args, "vavae_student_enable_decoder", False)),
                )
                return student, student_latent_dim

            student = LiteVAE(
                image_size=args.image_size,
                in_channels=3,
                base_channels=args.lite_vae_base_channels,
                latent_dim=args.lite_vae_latent_dim,
                dwt_levels=args.lite_vae_dwt_levels,
                variant=args.lite_vae_variant,
            )
            return student, int(args.lite_vae_latent_dim)

        lite_vae, student_latent_dim = _build_student_vae()
        lite_vae = lite_vae.to(args.device)
        args.student_latent_dim = int(student_latent_dim)
        lite_classifier = Linear(args.student_latent_dim, num_class).to(args.device)
        if kd_teacher_source == "resnet":
            if model is None:
                raise RuntimeError("kd_teacher_source=resnet requires ResNet backbone, but it is disabled.")
            kd_teacher_feat_dim = model.n_features
        elif kd_teacher_source == "vavae":
            kd_teacher_feat_dim = int(getattr(args, "vavae_teacher_latent_dim", 32))
        else:
            kd_teacher_feat_dim = args.student_latent_dim

        need_feat_proj = bool(args.kd_feat_project) and (kd_teacher_feat_dim != args.student_latent_dim)
        if need_feat_proj:
            if getattr(args, "kd_feat_project_mlp", False):
                hidden_dim = int(getattr(args, "kd_feat_proj_hidden_dim", 0))
                if hidden_dim <= 0:
                    hidden_dim = max(args.student_latent_dim, kd_teacher_feat_dim)
                dropout = float(getattr(args, "kd_feat_proj_dropout", 0.0))
                use_bn = bool(getattr(args, "kd_feat_proj_use_bn", True))
                depth = int(getattr(args, "kd_feat_proj_depth", 2))
                if depth <= 1:
                    kd_feat_proj = torch.nn.Linear(args.student_latent_dim, kd_teacher_feat_dim).to(args.device)
                else:
                    layers = []
                    in_dim = args.student_latent_dim
                    for layer_idx in range(depth - 1):
                        is_last = layer_idx == (depth - 2)
                        out_dim = kd_teacher_feat_dim if is_last else hidden_dim
                        layers.append(torch.nn.Linear(in_dim, out_dim))
                        if not is_last:
                            if use_bn:
                                layers.append(torch.nn.BatchNorm1d(out_dim))
                            layers.append(torch.nn.ReLU(inplace=True))
                            if dropout > 0:
                                layers.append(torch.nn.Dropout(p=dropout))
                        in_dim = out_dim
                    kd_feat_proj = torch.nn.Sequential(*layers).to(args.device)
            else:
                kd_feat_proj = torch.nn.Linear(args.student_latent_dim, kd_teacher_feat_dim).to(args.device)

        def _resolve_existing_path(path):
            if not path:
                return ""
            candidates = []
            if os.path.isabs(path) or str(path).startswith("./"):
                candidates.append(path)
            else:
                checkpoints_root = getattr(args, "checkpoints_root", os.path.dirname(args.checkpoints))
                candidates.extend([
                    path,  # e.g. ./checkpoints/run_x/xxx.pth
                    os.path.join(args.checkpoints, path),  # current run dir
                    os.path.join(checkpoints_root, path),  # checkpoints root
                ])
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
            return ""

        def _maybe_load(module, path, name):
            if module is None or not path:
                return
            load_path = _resolve_existing_path(path)
            if not load_path:
                raise FileNotFoundError(f"{name} checkpoint not found: {path}")
            state = torch.load(load_path, map_location=_get_map_location())
            module.load_state_dict(state)
            if rank == 0:
                print(f"[Resume] Loaded {name} from {load_path}")

        _maybe_load(lite_vae, args.lite_vae_resume_path, "lite_vae")
        _maybe_load(lite_classifier, args.lite_classifier_resume_path, "lite_classifier")
        _maybe_load(kd_feat_proj, args.kd_feat_proj_resume_path, "kd_feat_proj")
        if student_source == "vavae" and not args.lite_vae_resume_path:
            vavae_student_init_path = _resolve_existing_path(getattr(args, "vavae_student_init_path", ""))
            if vavae_student_init_path:
                load_info = lite_vae.load_pretrained(
                    vavae_student_init_path,
                    strict=bool(getattr(args, "vavae_student_load_strict", False)),
                    partial=bool(getattr(args, "vavae_student_partial_load", True)),
                    map_location=_get_map_location(),
                )
                if rank == 0:
                    print(f"[Init] Loaded VA-VAE student init from: {vavae_student_init_path}")
                    print(f"[Init] VA-VAE student load stats: {load_info}")

        if args.kd_enable and kd_teacher_source == "lite":
            lite_vae_teacher, _ = _build_student_vae()
            lite_vae_teacher = lite_vae_teacher.to(args.device)
            lite_classifier_teacher = Linear(args.student_latent_dim, num_class).to(args.device)

            lite_vae_teacher.load_state_dict(lite_vae.state_dict())
            lite_classifier_teacher.load_state_dict(lite_classifier.state_dict())
            for p in lite_vae_teacher.parameters():
                p.requires_grad_(False)
            for p in lite_classifier_teacher.parameters():
                p.requires_grad_(False)
            lite_vae_teacher.eval()
            lite_classifier_teacher.eval()
            if rank == 0:
                print(f"[KD] Using {student_source.upper()} self-distillation teacher branch.")
                if not args.lite_vae_resume_path:
                    print(
                        "[KD] Warning: self-distill teacher is cold-started (no lite_vae_resume_path). "
                        "This can reinforce early mistakes."
                    )
        elif args.kd_enable and kd_teacher_source == "vavae":
            vavae_teacher = VAVAETeacherEncoder(
                in_channels=3,
                ch=int(getattr(args, "vavae_teacher_ch", 128)),
                ch_mult=getattr(args, "vavae_teacher_ch_mult", "1,1,2,2,4"),
                num_res_blocks=int(getattr(args, "vavae_teacher_num_res_blocks", 2)),
                z_channels=int(getattr(args, "vavae_teacher_latent_dim", 32)),
                attn_levels=getattr(args, "vavae_teacher_attn_levels", "4"),
                input_size=int(getattr(args, "vavae_teacher_input_size", args.image_size)),
                resize_input=bool(getattr(args, "vavae_teacher_resize_input", False)),
                pool=str(getattr(args, "vavae_teacher_pool", "avg")),
                feature_from=str(getattr(args, "vavae_teacher_feature_from", "mu")),
            ).to(args.device)
            vavae_ckpt_path = _resolve_existing_path(getattr(args, "vavae_ckpt_path", ""))
            if not vavae_ckpt_path:
                raise FileNotFoundError(
                    "VA-VAE checkpoint not found. Set --vavae_ckpt_path to a valid file path."
                )
            load_info = vavae_teacher.load_pretrained(
                vavae_ckpt_path,
                strict=bool(getattr(args, "vavae_teacher_load_strict", False)),
                partial=bool(getattr(args, "vavae_teacher_partial_load", True)),
                map_location=_get_map_location(),
            )
            for p in vavae_teacher.parameters():
                p.requires_grad_(False)
            vavae_teacher.eval()
            if rank == 0:
                print(f"[KD] Using VA-VAE teacher from: {vavae_ckpt_path}")
                print(f"[KD] VA-VAE load stats: {load_info}")

    optim_params = []
    if model is not None and not (args.kd_enable and args.kd_freeze_teacher):
        optim_params += list(model.parameters())
    if lite_vae is not None:
        optim_params += list(lite_vae.parameters())
    if lite_classifier is not None:
        optim_params += list(lite_classifier.parameters())
    if kd_feat_proj is not None:
        optim_params += list(kd_feat_proj.parameters())

    optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9)

    if args.dataparallel:
        if model is not None:
            model = convert_model(model)
            model = DataParallel(model)
        if ema_model is not None:
            ema_model = convert_model(ema_model)
            ema_model = DataLoader(ema_model)
        if lite_vae is not None:
            lite_vae = DataParallel(lite_vae)
        if lite_classifier is not None:
            lite_classifier = DataParallel(lite_classifier)
        if kd_feat_proj is not None:
            kd_feat_proj = DataParallel(kd_feat_proj)
    else:
        if args.world_size > 1:
            ddp_find_unused = bool(getattr(args, "ddp_find_unused_parameters", True))
            if model is not None and any(p.requires_grad for p in model.parameters()):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu], find_unused_parameters=ddp_find_unused)
            if lite_vae is not None:
                lite_vae = DDP(lite_vae, device_ids=[gpu], find_unused_parameters=ddp_find_unused)
            if lite_classifier is not None:
                lite_classifier = DDP(lite_classifier, device_ids=[gpu], find_unused_parameters=ddp_find_unused)
            if kd_feat_proj is not None:
                kd_feat_proj = DDP(kd_feat_proj, device_ids=[gpu], find_unused_parameters=ddp_find_unused)

    log_f = None
    if args.debug and args.log_file and rank == 0:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
    trainEncoder(
        model,
        ema_model,
        loaders,
        optimizer,
        wandb_logger,
        args,
        lite_vae=lite_vae,
        lite_classifier=lite_classifier,
        lite_vae_teacher=lite_vae_teacher,
        lite_classifier_teacher=lite_classifier_teacher,
        vavae_teacher=vavae_teacher,
        kd_feat_proj=kd_feat_proj,
        log_f=log_f,
    )
    if log_f is not None:
        log_f.close()


def run_stage2(args):
    cmd = [sys.executable, "stage2.py"]
    passthrough = [a for a in sys.argv[1:] if a not in ("--auto_run_stage2",)]
    # Remove stage1-only args
    passthrough = [a for a in passthrough if a not in ("--stage2_log", "--stage2_debug")]
    if args.stage2_debug and "--debug" not in passthrough:
        passthrough.append("--debug")
    cmd.extend(passthrough)

    if args.stage2_log:
        with open(args.stage2_log, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=_arg_type_from_default(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    parser.add_argument('--log_file', type=str, default="", help='write debug logs to a local file')
    parser.add_argument('--auto_run_stage2', action="store_true", help='run stage2 after stage1 finishes')
    parser.add_argument('--stage2_debug', action="store_true", help='force stage2 to run in debug mode')
    parser.add_argument('--stage2_log', type=str, default="", help='log file path for stage2 output')
    args = parser.parse_args()

    _sanitize_cuda_alloc_conf()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    checkpoints_root = args.checkpoints
    args.checkpoints_root = checkpoints_root
    if args.student_run_name:
        args.run_name = args.student_run_name
    if not args.run_name:
        args.run_name = time.strftime("run_%Y%m%d_%H%M%S")
    args.checkpoints = os.path.join(checkpoints_root, args.run_name)

    # Respect externally provided visible devices and allow caller to override rendezvous config.
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12345')

    split_dir = resolve_isic2019lt_split_paths(args)

    # For ISIC2019LT, stage1 only resolves which csv files to use.
    if args.dataset == "ISIC2019LT":
        if split_dir:
            print(f"ISIC2019LT split dir: {split_dir}")
        if not isic2019lt_split_files_exist(args):
            raise FileNotFoundError(
                "Resolved ISIC2019LT split files are missing. "
                "Run the standalone split builder first, or copy the saved csv files into: "
                f"{split_dir}"
            )
        print("Using existing ISIC2019LT split files.")

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)

    # Run stage2 once after stage1 finishes (only in the launcher process)
    if args.auto_run_stage2:
        run_stage2(args)
