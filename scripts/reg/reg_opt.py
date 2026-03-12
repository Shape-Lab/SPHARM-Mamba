"""
Example script for diffeomorphic spherical registration without learning a.k.a. classical registration.

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
import sys
import re
import glob
import time
import platform
import argparse
import multiprocessing
import numpy as np
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib._parallel_backends import LokyBackend
from packaging import version

import torch
import torch.nn as nn
import torch.optim as optim

import spharmdnns
from spharmdnns import SPHARMReg
from spharmdnns.utils import Logger, parse_args, save_config, normalize_data
from spharmdnns.loss import CCLoss, ArcLoss, AreaLoss
from spharmdnns.io import read_feat, read_mesh, write_mesh, read_tif
from spharmdnns.sphere import Icosphere, TriangleSearch, rigid_alignment, retess, composite


def args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SPHARM-Reg: diffeomorphic spherical registration using classical optimization.",
        epilog="Contact: Ilwoo Lyu <ilwoolyu@postech.ac.kr>",
    )

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for registration", required=True)
    group.add_argument("--subj-dir", type=str, nargs="+", help="List of FreeSurfer's subject paths", required=True)
    group.add_argument("--target", type=str, nargs="+", help="Target geometry (tif or list of features tessellated by the ShapeLab icospheres)", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--version", action="version", version=f"%(prog)s {spharmdnns.__version__}")
    group.add_argument("--gpu", type=int, default=0, help="GPU ID for optimization (-1: all)")
    group.add_argument("--no-cuda", action="store_true", help="No CUDA")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads")
    group.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    group.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    group.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh")
    group.add_argument("--in-ch", type=str, nargs="+", default=["curv"], help="Geometry for similarity metric for tif")
    group.add_argument("--no-data-norm", action="store_true", help="No Z-score+prctile data normalization")

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--out-dir", type=str, default="output", help="Path to the registered sphere")
    group.add_argument("--out-postfix", type=str, default="sphere.reg", help="Postfix of the registered sphere")
    group.add_argument("--out-to-subj", action="store_true", help="Set 'subject' as output's root")
    group.add_argument("--fs", action="store_true", help="Enable FreeSurfer's mesh format")
    group.add_argument("--log-dir", type=str, help="Path to the log files")
    group.add_argument("--log-format", type=str, default="12.6", help="Log format specification")
    group.add_argument("--show-metrics", action="store_true", help="Show error metrics rather than progress bar")

    # Optimizer
    group = parser.add_argument_group("optimizer")
    group.add_argument("--epochs", type=int, default=200, help="Max epoch (0: rigid alignment only)")
    group.add_argument("--lr", "--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    group.add_argument("--decay", type=int, default=4, help="Decay after every # epochs if no progress (0: no decay)")

    # SPHARM-Net settings
    group = parser.add_argument_group("backbone")
    group.add_argument("-D", "--depth", type=int, default=2, help="Depth of SPHARM-Net")
    group.add_argument("-C", "--channel", type=int, default=8, help="# of channels in the entry layer of SPHARM-Net")
    group.add_argument("-L", "--bandwidth", type=int, default=20, help="Bandwidth of SPHARM-Net")
    group.add_argument("--interval", type=int, default=5, help="Anchor interval of hamonic coefficients")

    # Rigid alignment
    group = parser.add_argument_group("rigid alignment")
    group.add_argument("--rigid-feat", type=str, nargs="+", default=["inflated.H", "sulc"], help="List of geometry for rigid alignment for tif")
    group.add_argument("--rigid-target", type=str, nargs="+", help="Target geometry for rigid alignment other than 'target'")
    group.add_argument("--rigid-ico", type=int, nargs="+", default=[4, 5], help="List of icospheres for rigid alignment")
    group.add_argument("--rigid-interval", type=int, default=4, help="Rotation interval (π/interval) of global search for rigid alignment")
    group.add_argument("--rigid-axis-ico", type=int, default=1, help="Icospheral rotation axes of global search for rigid alignment")
    group.add_argument("--rigid-cand", type=int, default=4, help="# of optimizable candidates of global search for rigid alignment")

    # SPHARM-Reg settings
    group = parser.add_argument_group("registration")
    group.add_argument("--composite", type=int, default=0, help="# of warp field compositions")
    group.add_argument("--DL", type=int, default=40, help="SHT bandwidth for velocity field decomposition")
    group.add_argument("-k", type=int, default=6, help="Numerical integration step for scaling and squaring")
    group.add_argument("--loss-mse", type=float, default=6, help="Weight for MSE loss (similarity)")
    group.add_argument("--loss-cc", type=float, default=0, help="Weight for CC loss (similarity)")
    group.add_argument("--loss-arc", type=float, default=1, help="Weight for arc loss (regularity)")
    group.add_argument("--loss-arc-max", type=float, default=1, help="Weight for max arc loss (regularity)")
    group.add_argument("--loss-area", type=float, default=0, help="Weight for area loss (regularity)")
    group.add_argument("--loss-area-max", type=float, default=0.05, help="Weight for max area loss (regularity)")
    group.add_argument("--in-weight", type=float, nargs="+", help="Weight for each input geometry")
    group.add_argument("--ico", type=int, default=6, help="Level of icosahedral subdivision")

    args = parse_args(parser)

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"])

    return args


def tag(cmd, args):
    tag = cmd
    tag += f"\nProgramName: {os.path.splitext(os.path.basename(sys.argv[0]))[0]}"
    tag += f"\nProgramArguments: {vars(args)}"
    tag += f"\nProgramVersion: {spharmdnns.__version__}"
    tag += f"\nTimeStamp: {datetime.now().astimezone().strftime('%a %b %d %H:%M:%S %Y %z')}"
    tag += f"\nUser: {os.environ.get('USER')}"
    tag += f"\nPlatform: {platform.system()}"
    tag += f"\nPlatformVersion: {platform.release()}"

    return tag


def read_subj(subj_dir, feat_dir, hemi, in_ch, data_norm):
    feat_dir = os.path.join(subj_dir, feat_dir)

    data = [read_feat(os.path.join(feat_dir, hemi + "." + ch)) for ch in in_ch]
    data = np.asarray(data, dtype=np.float32).reshape((len(in_ch), -1))
    data = normalize_data(data) if data_norm else data

    return data


def eval_performance(start_time, epoch, total, width):
    elapsed = time.time() - start_time
    speed = epoch / elapsed if elapsed > 0 else 0.0
    remaining = (total - epoch) / speed if speed > 0 else float("inf")

    elapsed = f"{int(elapsed // 60):02}:{int(elapsed % 60):02}"
    remaining = f"{int(remaining // 60):02}:{int(remaining % 60):02}"
    speed = f"{speed:.2f}it/s" if speed >= 1 else f"{(1/speed):.2f}s/it" if speed > 0 else "--"

    return f"{f'{elapsed}<{remaining}':^{max(width, 13)}}{speed:^{max(width, 10)}}"


def eval_metric(v_new, input, target, criterion, weight, epoch, logger, log_format):
    loss_cc = criterion["cc"](input, target)
    loss_mse = criterion["mse"](input, target).mean(-1)
    loss_arc = criterion["arc"](v_new) if "arc" in criterion else torch.tensor(0.0)
    loss_area = criterion["area"](v_new) if "area" in criterion else torch.tensor(1.0)

    loss_arc_sum = loss_arc.sum()
    loss_arc_max = loss_arc.max()
    loss_log_area = loss_area.log2()

    loss = (
        (loss_mse * weight["feat"]).sum() * weight["mse"]
        + (loss_cc * weight["feat"]).sum() * weight["cc"]
        + loss_arc_sum * weight["arc"]
        + loss_arc_max * weight["arc_max"]
        + loss_log_area.mean() * weight["area"]
        + loss_log_area.max() * weight["area_max"]
    )

    metric = (
        logger.write(
            [
                epoch,
                loss.item(),
                *loss_mse.ravel().tolist(),
                *(1 - loss_cc).ravel().tolist(),
                loss_arc_sum.item(),
                loss_arc_max.item(),
                loss_area.mean().item(),
                loss_area.max().item(),
            ],
            log_format,
        )
        if logger is not None
        else ""
    )

    return loss, metric


def step(model, input, target, criterion, weight, epoch, logger, optimizer, cache, log_format):
    optimizer.zero_grad()
    model.set_cache(cache)
    v_new, target = model(input, target)

    loss, metric = eval_metric(v_new, input, target, criterion, weight, epoch, logger, log_format)

    loss.backward()
    optimizer.step()

    return loss, v_new, metric


def reg_sphere(args, subj_dir, in_target, in_target_v, rigid_sphere, rigid_target, rigid_feat, verbose=True):
    device = torch.device("cpu" if args.no_cuda else f"cuda:{pid_to_gid[os.getpid()] if args.gpu == -1 else args.gpu}")
    in_target = in_target.to(device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
    if version.parse(torch.__version__) >= version.parse("1.13.0") or (
        args.composite == 0 and version.parse(torch.__version__) >= version.parse("1.9.0")
    ):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    elif verbose:
        print("Warning: SPHARM-Reg uses non-deterministic algorithms.")

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

    criterion = {
        "mse": nn.MSELoss(reduction="none"),
        "cc": CCLoss(reduction="none"),
    }

    # subj info
    args.subj_dir = subj_dir
    out_prefix = list(filter(None, subj_dir.split("/")))
    out_prefix = out_prefix[-1] + "." + args.hemi if len(out_prefix) > 1 else args.hemi

    if verbose:
        print(f"Loading data from {subj_dir}")
    native_data = read_subj(subj_dir, args.feat_dir, args.hemi, args.in_ch, not args.no_data_norm)
    sphere = os.path.join(subj_dir, args.native_sphere_dir, args.hemi + "." + args.native_sphere)
    native_v, native_f = read_mesh(sphere)
    native_v = native_v.astype(float)
    native_f = native_f.astype(int)
    native_v /= np.linalg.norm(native_v, axis=1, keepdims=True)

    # logger
    log_width = int(args.log_format.split(".")[0])
    write = lambda x: sys.stdout.write(x) if args.show_metrics else x
    if args.log_dir or args.show_metrics:
        logger = Logger(os.path.join(args.log_dir, out_prefix + ".log") if args.log_dir else None)

        # logging the current configurations
        logger.write({"arguments": vars(args)})
        metric = logger.write(
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
        write(metric + f"{'Elapsed':^{max(log_width, 13)}}{'Speed':^{max(log_width, 10)}}" + "\n")

        target_v = torch.from_numpy(in_target_v).to(device, dtype=torch.float32)
        in_data = retess(in_target_v, native_data, TriangleSearch(native_v, native_f))
        in_data = torch.from_numpy(in_data).to(device, dtype=torch.float32)
        write(eval_metric(target_v, in_data, in_target, criterion, weight, "(Initial)", logger, args.log_format)[1] + "\n")
    else:
        logger = None

    # rigid alignment
    start_time = time.time()
    native_v_rigid = native_v
    for i, (v, target, feat) in enumerate(zip(rigid_sphere, rigid_target, rigid_feat)):
        feat = read_subj(subj_dir, args.feat_dir, args.hemi, [feat], False)
        rot_mat = rigid_alignment(
            v,
            native_v_rigid,
            native_f,
            target[: v.shape[0]],
            feat,
            search_intv=args.rigid_interval if i == 0 else 0,
            search_ico=args.rigid_axis_ico,
            search_topk=args.rigid_cand,
        )
        native_v_rigid = native_v_rigid @ rot_mat
        native_v_rigid /= np.linalg.norm(native_v_rigid, axis=1, keepdims=True)

        if args.log_dir or args.show_metrics:
            in_data = retess(in_target_v, native_data, TriangleSearch(native_v_rigid, native_f))
            in_data = torch.from_numpy(in_data).to(device, dtype=torch.float32)
            write(
                eval_metric(target_v, in_data, in_target, criterion, weight, "(Rigid)", logger, args.log_format)[1]
                + eval_performance(start_time, i + 1, len(rigid_sphere), log_width)
                + "\r"
            )
    v = native_v_rigid
    write("\n")

    if args.epochs > 0:
        model = SPHARMReg(
            device=device,
            ico=args.ico,
            composite=args.composite,
            in_ch=len(args.in_ch) + in_target.shape[0],
            DL=args.DL,
            k=args.k,
            C=args.channel,
            L=args.bandwidth,
            D=args.depth,
            interval=args.interval,
            threads=1,
            verbose=False,
        )
        model.to(device)
        model.train()

        # training loss
        criterion["arc"] = ArcLoss(model.v, model.f, reduction="none")
        criterion["area"] = AreaLoss(model.v, model.f, reduction="none")

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.decay - 1, threshold=1e-4, threshold_mode="abs", min_lr=1e-8)
            if args.decay > 0
            else None
        )

        # channel data
        in_data = retess(model.v.cpu().numpy(), native_data, TriangleSearch(native_v_rigid, native_f))
        in_data = torch.from_numpy(in_data).to(device, dtype=torch.float32)[None]

        # cache
        cache = (
            {
                "composite": [torch.zeros((1, model.v.shape[0]), dtype=torch.long) for _ in range(args.composite)],
                "retess": [torch.zeros((1, model.v.shape[0]), dtype=torch.long) for _ in range(args.composite + 1)],
                "rblock": [torch.zeros((1, model.v.shape[0]), dtype=torch.long) for _ in range(args.composite + 1)],
                "wblock": [torch.zeros((args.k, 1, model.v.shape[0]), dtype=torch.long) for _ in range(args.composite + 1)],
            }
            if args.no_cuda
            else None
        )

        in_target = in_target[None]
        start_time = time.time()
        for epoch in tqdm(
            range(args.epochs),
            desc=f"({device.type}{f':{device.index}' if device.index is not None else ''}) {out_prefix}",
            disable=args.show_metrics,
        ):
            epoch += 1
            loss, v, metric = step(model, in_data, in_target, criterion, weight, epoch, logger, optimizer, cache, args.log_format)
            if scheduler:
                scheduler.step(loss.detach())
            write(metric + eval_performance(start_time, epoch, args.epochs, log_width) + "\r")

        v = composite(torch.from_numpy(native_v_rigid).to(device, dtype=torch.float), v, model.tree)
        v = v.squeeze().cpu().detach().numpy()

        del model
        if not args.no_cuda:
            torch.cuda.empty_cache()
        write("\n")

    out_dir = args.out_dir
    if args.out_to_subj:
        out_dir = os.path.join(subj_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.out_to_subj:
        out_file = ".".join([args.hemi, args.out_postfix])
    else:
        out_file = ".".join([out_prefix, args.out_postfix])
    out_file = os.path.join(out_dir, out_file)
    if not args.fs:
        out_file += ".vtk"
    cmd = re.sub(r"--subj-dir(?:\s+(?!--\w)\S+)+", f"--subj-dir {subj_dir}", " ".join(sys.argv))

    if verbose:
        print(f"Saving {out_file}")
    write_mesh(out_file, v if not args.fs else v * 100, native_f, tag=tag(cmd, args))


def main(args):
    args.no_cuda |= not torch.cuda.is_available()

    args.subj_dir = sorted({f for p in args.subj_dir for f in glob.glob(p)})
    torch.set_num_threads(max(args.threads // len(args.subj_dir), 1))
    args.show_metrics &= len(args.subj_dir) == 1 or args.threads == 1

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    print("Loading template(s)...")
    ico_v, _ = Icosphere(args.ico).sphere[-1]

    target = args.target
    if len(target) == 1 and os.path.splitext(target[0])[1].lower() == ".tif":
        target, _, _ = read_tif(target[0], ico_v)
        target = [target[feat] for feat in args.in_ch]
    else:
        target = [read_feat(feat) for feat in target]
    target = np.asarray(target, dtype=np.float32)
    target = target[..., : ico_v.shape[0]]
    if not args.no_data_norm:
        target = normalize_data(target)
    target = torch.from_numpy(target)

    rigid_target = args.rigid_target if args.rigid_target is not None else args.target
    rigid_sphere = Icosphere(max(args.rigid_ico)).sphere
    rigid_sphere = [rigid_sphere[i][0] for i in args.rigid_ico]

    if len(rigid_target) == 1 and os.path.splitext(rigid_target[0])[1].lower() == ".tif":
        rigid_target, _, _ = read_tif(rigid_target[0], rigid_sphere[-1])
        rigid_target = [rigid_target[feat] for feat in args.rigid_feat]
    else:
        rigid_target = [read_feat(fn) for fn in rigid_target]
    rigid_target += [rigid_target[-1]] * (len(args.rigid_ico) - len(rigid_target))
    rigid_feat = args.rigid_feat.copy()
    rigid_feat += [rigid_feat[-1]] * (len(rigid_sphere) - len(rigid_feat))

    for i in range(args.threads):
        gid.put(i % torch.cuda.device_count() if not args.no_cuda else 0)

    init_worker = lambda: pid_to_gid.setdefault(os.getpid(), gid.get_nowait())
    LokyBackend.configure = (lambda orig: lambda self, *a, **k: orig(self, *a, **{**k, "initializer": init_worker}))(LokyBackend.configure)
    Parallel(n_jobs=args.threads, backend="loky")(
        delayed(reg_sphere)(args, subj_dir, target, ico_v, rigid_sphere, rigid_target, rigid_feat, len(args.subj_dir) == 1 or args.threads == 1)
        for subj_dir in args.subj_dir
    )


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    pid_to_gid = manager.dict()
    gid = manager.Queue()

    main(args())
