"""
Example script for training SPHARM-Reg.

If you use this code, please cite the following paper.

    Seungeun Lee, Seunghwan Lee, Sunghwa Ryu, and Ilwoo Lyu
    SPHARM-Reg: Unsupervised Cortical Surface Registration using Spherical Harmonics.
    IEEE Transactions on Medical Imaging

Copyright 2023 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack
from packaging import version

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spharmdnns import SPHARMReg
from spharmdnns.utils import SphericalDataset, Logger, parse_args, save_config
from spharmdnns.loss import CCLoss, ArcLoss, AreaLoss
from spharmdnns.io import read_feat, read_tif
from spharmdnns.sphere import Icosphere


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for learning", required=True)
    group.add_argument("--target", type=str, nargs="+", help="Target geometry", required=True)
    group.add_argument("--data-dir", type=str, help="Path to re-tessellated data", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--gpu", type=int, default=0, help="GPU ID for training (normally, starting with 0)")
    group.add_argument("--no-cuda", action="store_true", help="No CUDA")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads")
    group.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Dataset
    group = parser.add_argument_group("dataset")
    group.add_argument("--preload", type=str, choices=["none", "cpu", "device"], default="device", help="Data preloading")
    group.add_argument("--in-ch", type=str, nargs="+", default=["curv"], help="Geometry for similarity metric")
    group.add_argument("--no-data-norm", action="store_true", help="No Z-score+prctile data normalization")
    group.add_argument("--n-splits", type=int, default=5, help="A total of cross-validation folds")
    group.add_argument("--fold", type=int, default=1, help="Cross-validation fold")
    group.add_argument("--aug", type=int, default=0, help="Level of data augmentation")

    # Training
    group = parser.add_argument_group("training")
    group.add_argument("--epochs", type=int, default=80, help="Max epoch")
    group.add_argument("--batch-size", type=int, default=1, help="Batch size")
    group.add_argument("--lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    group.add_argument("--decay", type=int, default=4, help="Decay after every # epochs if no progress (0: no decay)")
    group.add_argument("--det", action="store_true", help="Enable deterministic algorithms for reproducibility")
    group.add_argument("--resume", type=str, default=None, help="Checkpoint (pt) to resume training")

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--log-dir", type=str, default="log", help="Path to the log files (output)")
    group.add_argument("--log-format", type=str, default="12.6", help="Log format specification")
    group.add_argument("--ckpt-dir", type=str, default="log", help="Path to the checkpoint file (output)")

    # SPHARM-Net settings
    group = parser.add_argument_group("backbone")
    group.add_argument("-D", "--depth", type=int, default=3, help="Depth of SPHARM-Net")
    group.add_argument("-C", "--channel", type=int, default=32, help="# of channels in the entry layer of SPHARM-Net")
    group.add_argument("-L", "--bandwidth", type=int, default=80, help="Bandwidth of SPHARM-Net")
    group.add_argument("--interval", type=int, default=5, help="Anchor interval of hamonic coefficients")

    # SPHARM-Reg settings
    group = parser.add_argument_group("registration")
    group.add_argument("--composite", type=int, default=2, help="# of warp field compositions")
    group.add_argument("--DL", type=int, default=40, help="SHT bandwidth for velocity field decomposition")
    group.add_argument("-k", type=int, default=6, help="Numerical integration step for scaling and squaring")
    group.add_argument("--loss-mse", type=float, default=10, help="Weight for MSE loss (similarity)")
    group.add_argument("--loss-cc", type=float, default=0, help="Weight for CC loss (similarity)")
    group.add_argument("--loss-arc", type=float, default=1, help="Weight for arc loss (regularity)")
    group.add_argument("--loss-arc-max", type=float, default=0, help="Weight for max arc loss (regularity)")
    group.add_argument("--loss-area", type=float, default=0, help="Weight for area loss (regularity)")
    group.add_argument("--loss-area-max", type=float, default=0, help="Weight for max area loss (regularity)")
    group.add_argument("--in-weight", type=float, nargs="+", default=None, help="Weight for each input geometry")
    group.add_argument("--ico", type=int, default=6, help="Level of icosahedral subdivision")

    args = parse_args(parser, section_prefix="reg.train")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="reg.train")

    return args


def step(model, loader, device, criterion, weight, epoch, logger, log_format, optimizer=None, pbar=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    progress = lambda x: tqdm(x, desc=f"Epoch {str(pbar)}") if pbar is not None else x

    running_mse = 0.0
    running_cc = 0.0
    running_arc = 0.0
    running_arc_max = 0.0
    running_area = 0.0
    running_area_max = 0.0
    running_loss = 0.0

    iter = 0
    with torch.no_grad() if optimizer is None else ExitStack():
        for input, target, _ in progress(loader):
            input = input.to(device)
            target = target.to(device)

            if optimizer is not None:
                optimizer.zero_grad()
            v_new, target = model(input, target)

            loss_cc = criterion["cc"](input, target)
            loss_mse = criterion["mse"](input, target).mean(-1)
            loss_arc = criterion["arc"](v_new)
            loss_area = criterion["area"](v_new)

            loss_arc_sum = loss_arc.sum(-1).mean()
            loss_arc_max = loss_arc.max(-1)[0].mean()
            loss_log_area = loss_area.log2()

            loss = (
                (loss_mse * weight["feat"]).sum(-1).mean() * weight["mse"]
                + (loss_cc * weight["feat"]).sum(-1).mean() * weight["cc"]
                + loss_arc_sum * weight["arc"]
                + loss_arc_max * weight["arc_max"]
                + loss_log_area.mean(-1).mean() * weight["area"]
                + loss_log_area.max(-1)[0].mean() * weight["area_max"]
            )

            running_mse += loss_mse.mean(0)
            running_cc += loss_cc.mean(0)
            running_arc += loss_arc_sum
            running_arc_max += loss_arc_max
            running_area += loss_area.mean(-1).mean()
            running_area_max += loss_area.max(-1)[0].mean()
            running_loss += loss

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            iter += 1

    logger.write(
        [
            epoch,
            running_loss.item() / iter,
            *(running_mse / iter).ravel().tolist(),
            *(1 - running_cc / iter).ravel().tolist(),
            running_arc.item() / iter,
            running_arc_max.item() / iter,
            running_area.item() / iter,
            running_area_max.item() / iter,
        ],
        log_format,
    )

    return running_loss / iter


def main(args):
    if args.resume:
        checkpoint = torch.load(args.resume, **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {}))

        if checkpoint.get("epoch") >= args.epochs:
            print("Training has already been completed.")
            return

        for key, value in checkpoint["args"].items():
            if key not in ["preload", "log_dir", "log_format", "ckpt_dir", "resume", "det", "epochs", "threads", "no_cuda"]:
                setattr(args, key, value)

    args.no_cuda |= not torch.cuda.is_available()
    device = torch.device("cpu" if args.no_cuda else f"cuda:{args.gpu}")
    preload = None if args.preload == "none" else device if args.preload == "device" else args.preload

    torch.set_num_threads(args.threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    if args.det:
        if version.parse(torch.__version__) >= version.parse("1.13.0") or (
            args.composite == 0 and version.parse(torch.__version__) >= version.parse("1.9.0")
        ):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        else:
            print("Warning: SPHARM-Reg uses non-deterministic algorithms.")

    print("Loading data...")

    # target
    target = args.target
    if len(target) == 1 and os.path.splitext(target[0])[1] == ".tif":
        rigid_sphere = Icosphere(args.ico).sphere
        target, _, _ = read_tif(target[0], rigid_sphere[-1][0])
        target = [target[feat] for feat in args.in_ch]
    else:
        target = [read_feat(feat) for feat in target]
    target = np.asarray(target, dtype=np.float32)

    # dataset partition
    partition = ["train", "val", "test"]
    dataset = dict()
    for partition_type in partition:
        dataset[partition_type] = SphericalDataset(
            data_dir=args.data_dir,
            partition=partition_type,
            fold=args.fold,
            n_vert=4**args.ico * 10 + 2,
            classes=target,
            in_ch=args.in_ch,
            seed=args.seed,
            aug=args.aug,
            n_splits=args.n_splits,
            hemi=args.hemi,
            data_norm=not args.no_data_norm,
            preload=preload,
        )

    # dataset loader
    loader = dict()
    loader["train"] = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, drop_last=False)
    loader["val"] = DataLoader(dataset["val"], batch_size=args.batch_size, shuffle=False, drop_last=False)
    loader["test"] = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False, drop_last=False)

    # logger
    os.makedirs(args.log_dir, exist_ok=True)
    logger = dict()
    for partition_type in partition:
        logger[partition_type] = Logger(os.path.join(args.log_dir, partition_type + ".log"))

    # SPHARM-Reg
    print("Loading model...")
    start_epoch = 0
    model = SPHARMReg(
        device=device,
        ico=args.ico,
        composite=args.composite,
        DL=args.DL,
        k=args.k,
        C=args.channel,
        L=args.bandwidth,
        D=args.depth,
        in_ch=len(args.in_ch) + target.shape[0],
        interval=args.interval,
        threads=args.threads,
        verbose=False,
    )
    model.to(device)

    # model parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of params {params}")
    arguments = vars(args)
    arguments["num_params"] = params

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.decay - 1, threshold=1e-4, threshold_mode="abs", min_lr=1e-8)
        if args.decay > 0
        else None
    )

    # training loss
    criterion = {
        "mse": nn.MSELoss(reduction="none"),
        "cc": CCLoss(reduction="none"),
        "arc": ArcLoss(model.v, model.f, reduction="none"),
        "area": AreaLoss(model.v, model.f, reduction="none"),
    }

    # weight adjustment
    weight = {
        "mse": args.loss_mse,
        "cc": args.loss_cc,
        "arc": args.loss_arc,
        "arc_max": args.loss_arc_max,
        "area": args.loss_area,
        "area_max": args.loss_area_max,
    }
    total_weight = sum(weight.values())
    weight = {key: value / total_weight for key, value in weight.items()}
    if args.in_weight is None:
        args.in_weight = [1] * len(args.in_ch)
    elif len(args.in_ch) != len(args.in_weight):
        raise Exception("len(args.in_ch) != len(args.in_weight)")
    weight["feat"] = torch.as_tensor(args.in_weight, dtype=torch.float32, device=device)[None]
    weight["feat"] /= weight["feat"].sum()

    # resume if past training is available
    if args.resume:
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        if not args.no_cuda and "cuda" in checkpoint["rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

    # ckpt path
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # rigid alignment configurations for inference
    with open(os.path.join(args.data_dir, "targets", f"{args.hemi}.config.json"), "r") as fd:
        rigid = json.load(fd)
    rigid["target"] = {feat: read_feat(os.path.join(args.data_dir, "targets", f"{args.hemi}.{feat}.dat")) for feat in rigid["feat"]}
    rigid["vert"] = read_feat(os.path.join(args.data_dir, "targets", f"{args.hemi}.vert.dat")).reshape(-1, 3)

    # logging the current configurations
    for partition_type in partition:
        if partition_type == "train":
            logger[partition_type].write(arguments)
        logger[partition_type].write({"fold_data": dataset[partition_type].subj_list})
        logger[partition_type].write(
            [
                "Epoch",
                "Loss",
                *[f"MSE_{ch}" for ch in args.in_ch],
                *[f"CC_{ch}" for ch in args.in_ch],
                "Arc_sum",
                "Arc_max",
                "Area_mean",
                "Area_max",
            ],
            args.log_format,
        )

    # main loop
    best_loss = 1e8
    for epoch in range(start_epoch, args.epochs):
        epoch += 1
        step(model, loader["train"], device, criterion, weight, epoch, logger["train"], args.log_format, optimizer, epoch)
        val_loss = step(model, loader["val"], device, criterion, weight, epoch, logger["val"], args.log_format)

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "acc": best_loss,
                    "args": arguments,
                    "target": target,
                    "rigid": rigid,
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                        **({"cuda": torch.cuda.get_rng_state_all()} if not args.no_cuda else {}),
                    },
                },
                os.path.join(args.ckpt_dir, "best_model_fold{}.pt".format(args.fold)),
            )

    # test
    test_ckpt = torch.load(
        os.path.join(args.ckpt_dir, "best_model_fold{}.pt".format(args.fold)),
        **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {}),
    )
    model.load_state_dict(test_ckpt["model_state_dict"])
    model.to(device)
    step(model, loader["test"], device, criterion, weight, test_ckpt["epoch"], logger["test"], args.log_format)


if __name__ == "__main__":
    main(args())
