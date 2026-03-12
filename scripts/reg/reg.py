"""
Example script for diffeomorphic spherical registration.

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
import platform
import argparse
import multiprocessing
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
from joblib._parallel_backends import LokyBackend
from packaging import version

import torch
import torch.nn as nn

import spharmdnns
from spharmdnns import SPHARMReg
from spharmdnns.utils import Logger, parse_args, save_config, normalize_data
from spharmdnns.loss import CCLoss, ArcLoss, AreaLoss
from spharmdnns.io import read_feat, read_mesh, write_mesh
from spharmdnns.sphere import Icosphere, TriangleSearch, rigid_alignment, retess, composite


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for inference", required=True)
    group.add_argument("--subj-dir", type=str, nargs="+", help="Path to subject dir", required=True)
    group.add_argument("--ckpt", type=str, help="Trained model for inference", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--gpu", type=int, default=0, help="GPU ID for inference (-1: all)")
    group.add_argument("--no-cuda", action="store_true", help="No CUDA")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    group.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    group.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh")

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--out-dir", type=str, default="output", help="Path to deformed sphere (output)")
    group.add_argument("--out-postfix", type=str, default="sphere.reg", help="Path to deformed sphere (output)")
    group.add_argument("--out-to-subj", action="store_true", help="Set output's root as the subject dir")
    group.add_argument("--fs", action="store_true", help="Enable FreeSurfer's mesh format")
    group.add_argument("--log-dir", type=str, help="Path to the log files")
    group.add_argument("--log-format", type=str, default="12.6", help="Log format specification")

    args = parse_args(parser, section_prefix="reg.reg")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="reg.reg")

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


def eval_metric(v_new, input, target, criterion, epoch, logger, log_format):
    loss_cc = criterion["cc"](input, target)
    loss_mse = criterion["mse"](input, target).mean(-1)
    loss_arc = criterion["arc"](v_new)
    loss_area = criterion["area"](v_new)

    loss_arc_sum = loss_arc.sum()
    loss_arc_max = loss_arc.max()

    logger.write(
        [
            epoch,
            *loss_mse.ravel().tolist(),
            *(1 - loss_cc).ravel().tolist(),
            loss_arc_sum.item(),
            loss_arc_max.item(),
            loss_area.mean().item(),
            loss_area.max().item(),
        ],
        log_format,
    )


def reg_sphere(args, ckpt_args, checkpoint, subj_dir, in_target, rigid):
    device = torch.device("cpu" if args.no_cuda else f"cuda:{pid_to_gid[os.getpid()] if args.gpu == -1 else args.gpu}")
    in_target = in_target.to(device)

    args.subj_dir = subj_dir

    print(f"Loading data from {subj_dir}")
    native_data = read_subj(subj_dir, args.feat_dir, args.hemi, ckpt_args["in_ch"], not ckpt_args["no_data_norm"])
    sphere = os.path.join(subj_dir, args.native_sphere_dir, args.hemi + "." + args.native_sphere)
    native_v, native_f = read_mesh(sphere)
    native_v = native_v.astype(float)
    native_f = native_f.astype(int)
    native_v /= np.linalg.norm(native_v, axis=1, keepdims=True)

    print("Rigid alignment...")
    native_v_rigid = native_v
    for i, (res, feat) in enumerate(zip(rigid["res"], rigid["order"])):
        target = rigid["target"][feat]
        feat = read_subj(subj_dir, args.feat_dir, args.hemi, [feat], False)
        rot_mat = rigid_alignment(
            rigid["vert"][:res, :],
            native_v_rigid,
            native_f,
            target[:res],
            feat,
            search_intv=rigid["intv"] if i == 0 else 0,
            search_ico=rigid["axis"],
            search_topk=rigid["cand"],
        )
        native_v_rigid = native_v_rigid @ rot_mat
        native_v_rigid /= np.linalg.norm(native_v_rigid, axis=1, keepdims=True)

    print("Diffeomorphic non-rigid registration...")
    model = SPHARMReg(
        device=device,
        ico=ckpt_args["ico"],
        composite=ckpt_args["composite"],
        DL=ckpt_args["DL"],
        k=ckpt_args["k"],
        C=ckpt_args["channel"],
        L=ckpt_args["bandwidth"],
        D=ckpt_args["depth"],
        interval=ckpt_args["interval"],
        in_ch=len(ckpt_args["in_ch"]) + checkpoint["target"].shape[0],
        threads=1,
        verbose=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # logger
    write = eval_metric if args.log_dir else (lambda *args, **kwargs: None)
    logger = criterion = None
    if args.log_dir:
        out_prefix = list(filter(None, subj_dir.split("/")))
        out_prefix = out_prefix[-1] + "." + args.hemi if len(out_prefix) > 1 else args.hemi
        logger = Logger(os.path.join(args.log_dir, out_prefix + ".log"))
        logger.write(
            [
                "Type",
                *[f"MSE_{ch}" for ch in ckpt_args["in_ch"]],
                *[f"CC_{ch}" for ch in ckpt_args["in_ch"]],
                "Arc_sum",
                "Arc_max",
                "Area_mean",
                "Area_max",
            ],
            args.log_format,
        )

        criterion = {
            "mse": nn.MSELoss(reduction="none"),
            "cc": CCLoss(reduction="none"),
            "arc": ArcLoss(model.v, model.f, reduction="none"),
            "area": AreaLoss(model.v, model.f, reduction="none"),
        }

        in_data = retess(model.v.cpu().numpy(), native_data, TriangleSearch(native_v, native_f))
        in_data = torch.from_numpy(in_data).to(device, dtype=torch.float)[None]
        write(model.v, in_data, in_target, criterion, "Initial", logger, args.log_format)

    in_data = retess(model.v.cpu().numpy(), native_data, TriangleSearch(native_v_rigid, native_f))
    in_data = torch.from_numpy(in_data).to(device, dtype=torch.float)[None]
    write(model.v, in_data, in_target, criterion, "Rigid", logger, args.log_format)

    with torch.no_grad():
        v, target = model(in_data, in_target)
    write(v, in_data, target, criterion, "Final", logger, args.log_format)

    v = composite(torch.from_numpy(native_v_rigid).to(device, dtype=torch.float), v, model.tree)
    v = v.squeeze().cpu().detach().numpy()

    del model
    if not args.no_cuda:
        torch.cuda.empty_cache()

    out_dir = args.out_dir
    if args.out_to_subj:
        out_dir = os.path.join(subj_dir, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.out_to_subj:
        out_file = ".".join([args.hemi, args.out_postfix])
    else:
        out_prefix = list(filter(None, subj_dir.split("/")))
        out_prefix = out_prefix[-1] + "." + args.hemi if len(out_prefix) > 1 else args.hemi
        out_file = ".".join([out_prefix, args.out_postfix])
    out_file = os.path.join(out_dir, out_file)
    if not args.fs:
        out_file += ".vtk"
    cmd = re.sub(r"--subj-dir(?:\s+(?!--\w)\S+)+", f"--subj-dir {subj_dir}", " ".join(sys.argv))

    print(f"Saving {out_file}")
    write_mesh(out_file, v if not args.fs else v * 100, native_f, tag=tag(cmd, args))


def main(args):
    args.no_cuda |= not torch.cuda.is_available()
    args.subj_dir = sorted({f for p in args.subj_dir for f in glob.glob(p)})
    if args.no_cuda:
        torch.set_num_threads(max(args.threads // len(args.subj_dir), 1))

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    print(f"Checkpoint: {args.ckpt}")
    checkpoint = torch.load(
        args.ckpt,
        map_location=torch.device("cpu"),
        **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {}),
    )
    ckpt_args = checkpoint["args"]

    print("Loading template(s)...")
    ico_v, _ = Icosphere(ckpt_args["ico"]).sphere[-1]
    in_target = checkpoint["target"][..., : ico_v.shape[0]]
    if not ckpt_args["no_data_norm"]:
        in_target = normalize_data(in_target)
    in_target = torch.from_numpy(in_target).type(torch.float)
    in_target = in_target.reshape(1, -1, ico_v.shape[0])

    for i in range(args.threads):
        gid.put(i % torch.cuda.device_count() if not args.no_cuda else 0)

    init_worker = lambda: pid_to_gid.setdefault(os.getpid(), gid.get_nowait())
    LokyBackend.configure = (lambda orig: lambda self, *a, **k: orig(self, *a, **{**k, "initializer": init_worker}))(LokyBackend.configure)
    Parallel(n_jobs=args.threads, backend="loky")(
        delayed(reg_sphere)(args, ckpt_args, checkpoint, subj_dir, in_target, checkpoint["rigid"]) for subj_dir in args.subj_dir
    )


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    pid_to_gid = manager.dict()
    gid = manager.Queue()

    main(args())
