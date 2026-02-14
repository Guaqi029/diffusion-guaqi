import os
import sys
import subprocess
import time
import torch
import wandb
import argparse
import torch.distributed as dist
from models import CreateModel, AuxVAE, LiteAuxVAE, LiteVAE, Linear
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data import ISICDataset, Transforms
from torch.utils.data import DataLoader
from train import trainEncoder
from utils.yaml_config_hook import yaml_config_hook
from utils.sync_batchnorm import convert_model
from prepare_datasets import construct_ISIC2019LT


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # training set
    transforms = Transforms(size=args.image_size)
    train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms)
    

    # set sampler for parallel training
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
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
    )
    if rank == 0:
        test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
        val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = None
        val_loader = None

    loaders = (train_loader, val_loader, test_loader)

    num_class = train_dataset.n_class
    args.num_classes = num_class

    # model init
    model = CreateModel(backbone=args.backbone, ema=False, out_features=num_class, pretrained=args.pretrained)
    ema_model = CreateModel(backbone=args.backbone, ema=True, out_features=num_class, pretrained=args.pretrained)
    def _get_map_location():
        if isinstance(args.device, torch.device):
            return args.device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")

    if args.reload:
        checkpoints_root = getattr(args, "checkpoints_root", os.path.dirname(args.checkpoints))
        teacher_run = args.teacher_run_name if args.teacher_run_name else os.path.basename(args.checkpoints)
        teacher_epoch = args.teacher_epoch if args.teacher_epoch > 0 else args.epochs
        teacher_ckpt_dir = os.path.join(checkpoints_root, teacher_run)
        model_fp = os.path.join(
            teacher_ckpt_dir, "epoch_{}_.pth".format(teacher_epoch)
        )
        model.load_state_dict(torch.load(model_fp, map_location=_get_map_location()))

    model = model.to(args.device)
    ema_model = ema_model.to(args.device)

    if args.kd_enable and args.kd_freeze_teacher:
        for param in model.parameters():
            param.requires_grad_(False)
        for param in ema_model.parameters():
            param.requires_grad_(False)
    if args.mix_enable and args.mix_freeze_teacher:
        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad_(False)

    aux_vae = None
    lite_vae = None
    lite_classifier = None
    kd_feat_proj = None
    if args.use_aux_vae:
        if args.aux_vae_type == "lite":
            aux_vae = LiteAuxVAE(
                image_size=args.image_size,
                in_channels=3,
                base_channels=args.aux_vae_base_channels,
                latent_dim=args.aux_vae_latent_dim,
                dwt_levels=args.aux_vae_dwt_levels,
            ).to(args.device)
        else:
            aux_vae = AuxVAE(
                in_features=model.n_features,
                latent_dim=args.aux_vae_latent_dim,
                image_size=args.image_size,
            ).to(args.device)

    if args.kd_enable or args.mix_enable or args.lite_eval_enable:
        lite_vae = LiteVAE(
            image_size=args.image_size,
            in_channels=3,
            base_channels=args.lite_vae_base_channels,
            latent_dim=args.lite_vae_latent_dim,
            dwt_levels=args.lite_vae_dwt_levels,
            variant=args.lite_vae_variant,
        ).to(args.device)
        lite_classifier = Linear(args.lite_vae_latent_dim, num_class).to(args.device)
        if args.kd_feat_project:
            kd_feat_proj = torch.nn.Linear(args.lite_vae_latent_dim, model.n_features).to(args.device)

        def _maybe_load(module, path, name):
            if module is None or not path:
                return
            load_path = path
            if not os.path.isabs(load_path):
                load_path = os.path.join(args.checkpoints, load_path)
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"{name} checkpoint not found: {load_path}")
            state = torch.load(load_path, map_location=_get_map_location())
            module.load_state_dict(state)
            if rank == 0:
                print(f"[Resume] Loaded {name} from {load_path}")

        _maybe_load(lite_vae, args.lite_vae_resume_path, "lite_vae")
        _maybe_load(lite_classifier, args.lite_classifier_resume_path, "lite_classifier")
        _maybe_load(kd_feat_proj, args.kd_feat_proj_resume_path, "kd_feat_proj")

    optim_params = []
    if not (args.kd_enable and args.kd_freeze_teacher):
        optim_params += list(model.parameters())
    if aux_vae is not None:
        optim_params += list(aux_vae.parameters())
    if lite_vae is not None:
        optim_params += list(lite_vae.parameters())
    if lite_classifier is not None:
        optim_params += list(lite_classifier.parameters())
    if kd_feat_proj is not None:
        optim_params += list(kd_feat_proj.parameters())

    optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9)

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
        ema_model = convert_model(ema_model)
        ema_model = DataLoader(ema_model)
        if aux_vae is not None:
            aux_vae = DataParallel(aux_vae)
        if lite_vae is not None:
            lite_vae = DataParallel(lite_vae)
        if lite_classifier is not None:
            lite_classifier = DataParallel(lite_classifier)
        if kd_feat_proj is not None:
            kd_feat_proj = DataParallel(kd_feat_proj)
    else:
        if args.world_size > 1:
            if any(p.requires_grad for p in model.parameters()):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu])
            if lite_vae is not None:
                lite_vae = DDP(lite_vae, device_ids=[gpu])
            if lite_classifier is not None:
                lite_classifier = DDP(lite_classifier, device_ids=[gpu])
            if kd_feat_proj is not None:
                kd_feat_proj = DDP(kd_feat_proj, device_ids=[gpu])

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
        aux_vae=aux_vae,
        lite_vae=lite_vae,
        lite_classifier=lite_classifier,
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
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    parser.add_argument('--log_file', type=str, default="", help='write debug logs to a local file')
    parser.add_argument('--auto_run_stage2', action="store_true", help='run stage2 after stage1 finishes')
    parser.add_argument('--stage2_debug', action="store_true", help='force stage2 to run in debug mode')
    parser.add_argument('--stage2_log', type=str, default="", help='log file path for stage2 output')
    args = parser.parse_args()

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

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # if the dataset is 2019LT, construct a new dataset split
    # with imbalance factor=args.imbalance_factor
    if args.dataset == "ISIC2019LT":
        print("Constructing ISIC2019LT Dataset with imbalance factor=%d" % args.imbalance_factor)
        construct_ISIC2019LT(imbalance_factor=args.imbalance_factor, data_root=args.data_path,
        csv_file_root=os.path.dirname(args.csv_file_train), random_seed=args.seed)

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb if not in debug mode
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


    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)

    # Run stage2 once after stage1 finishes (only in the launcher process)
    if args.auto_run_stage2:
        run_stage2(args)

