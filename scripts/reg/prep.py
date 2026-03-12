"""
Example script for data preparation.

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
from joblib import Parallel, delayed

from spharmdnns.utils import parse_args, save_config
from spharmdnns.io import read_feat, read_mesh, read_tif
from spharmdnns.sphere import Icosphere, TriangleSearch, rigid_alignment, retess


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for data generation", required=True)
    group.add_argument("--target", type=str, default=None, nargs="+", help="Target geometry", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads for parallel data generation")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--data-dir", type=str, help="Path to FreeSurfer home (default: $SUBJECTS_DIR)")
    group.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    group.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    group.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh")

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--in-ch", type=str, default=["curv"], nargs="+", help="List of geometry")
    group.add_argument("--out-dir", type=str, default="dataset", help="Path to re-tessellated data (output)")
    group.add_argument("--ico", type=int, default=6, help="Icosaheral subdivision of reference sphere for re-tessellation")

    # Rigid alignment
    group = parser.add_argument_group("rigid alignment")
    group.add_argument("--rigid-feat", type=str, nargs="+", default=["inflated.H", "sulc"], help="List of geometry for rigid alignment")
    group.add_argument("--rigid-ico", type=int, nargs="+", default=[4, 5], help="List of spheres for rigid alignment")
    group.add_argument("--rigid-interval", type=int, default=4, help="Rotation interval (π/interval) for global search for rigid alignment")
    group.add_argument("--rigid-axis-ico", type=int, default=1, help="Icospheral rotation axes of global search for rigid alignment")
    group.add_argument("--rigid-cand", type=int, default=4, help="# of optimizable candidates of global search for rigid alignment")

    # Group-wise alignment
    group = parser.add_argument_group("group-wise alignment")
    group.add_argument("--group-reg", action="store_true", help="Enable groupwise registration to templates")
    group.add_argument("--iter", type=int, default=3, help="# of iterations for template creation")

    args = parse_args(parser, section_prefix="reg.prep")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="reg.prep")

    return args


def gen_data(data_dir, out_dir, feat_dir, native_sphere_dir, subj_name, hemi, native_sphere, mat, in_ch, ico_v):
    print(f"Processing {subj_name}...")

    feat_dir = os.path.join(data_dir, subj_name, feat_dir)
    native_sphere_dir = os.path.join(data_dir, subj_name, native_sphere_dir)

    feat_out_dir = os.path.join(out_dir, "features")

    native_v, native_f = read_mesh(os.path.join(native_sphere_dir, hemi + "." + native_sphere))
    native_v = native_v @ mat

    feat = np.empty((len(in_ch), native_v.shape[0]))
    for i, feat_name in enumerate(in_ch):
        feat[i] = read_feat(os.path.join(feat_dir, hemi + "." + feat_name))

    feat = retess(ico_v, feat, TriangleSearch(native_v, native_f))

    for i, feat_name in enumerate(in_ch):
        feat[i].tofile(os.path.join(feat_out_dir, f"{subj_name}.{hemi}.aug0.{feat_name}.dat"))


def target_creation(data_dir, feat_dir, native_sphere_dir, subj_name, hemi, native_sphere, rot_mat, feat_name, ico_v):
    native_v, native_f = read_mesh(os.path.join(data_dir, subj_name, native_sphere_dir, hemi + "." + native_sphere))
    feat = read_feat(os.path.join(data_dir, subj_name, feat_dir, hemi + "." + feat_name))

    return retess(ico_v, feat[None], TriangleSearch(native_v @ rot_mat, native_f)).squeeze()


def rigid_registration(data_dir, feat_dir, native_sphere_dir, target_list, sphere_list, subj_name, hemi, native_sphere, mat, rigid_config):
    feat_dir = os.path.join(data_dir, subj_name, feat_dir)
    native_sphere_dir = os.path.join(data_dir, subj_name, native_sphere_dir)

    native_v, native_f = read_mesh(os.path.join(native_sphere_dir, hemi + "." + native_sphere))
    native_v /= np.linalg.norm(native_v, axis=1, keepdims=True)
    native_v = native_v @ mat
    for i, (feat, sphere, target) in enumerate(zip(rigid_config["order"], sphere_list, target_list)):
        feat = read_feat(os.path.join(feat_dir, hemi + "." + feat))
        ico_v, _ = sphere
        rot_mat = rigid_alignment(
            ico_v,
            native_v,
            native_f,
            target[: ico_v.shape[0]],
            feat,
            search_intv=rigid_config["intv"] if i == 0 else 0,
            search_ico=rigid_config["axis"],
            search_topk=rigid_config["cand"],
        )
        native_v = native_v @ rot_mat
        mat = mat @ rot_mat
        native_v /= np.linalg.norm(native_v, axis=1, keepdims=True)

    return mat


def iterative_mean(data_dir, feat_dir, native_sphere_dir, sphere_list, subj_list, hemi, native_sphere, rot_mat, rigid_config, threads):
    target_list = []
    for sphere, feat in zip(sphere_list, rigid_config["order"]):
        ico_v, _ = sphere
        mean_feat = Parallel(n_jobs=threads, backend="loky")(
            delayed(target_creation)(data_dir, feat_dir, native_sphere_dir, subj_name, hemi, native_sphere, mat, feat, ico_v)
            for subj_name, mat in zip(subj_list, rot_mat)
        )
        target_list.append(np.array(mean_feat).mean(axis=0))

    return (
        Parallel(n_jobs=threads, backend="loky")(
            delayed(rigid_registration)(
                data_dir, feat_dir, native_sphere_dir, target_list, sphere_list, subj_name, hemi, native_sphere, mat, rigid_config
            )
            for subj_name, mat in zip(subj_list, rot_mat)
        ),
        target_list,
    )


def main(args):
    feat_out_dir = os.path.join(args.out_dir, "features")
    rigid_dir = os.path.join(args.out_dir, "targets")
    rigid_sphere = Icosphere(max(args.rigid_ico)).sphere
    rigid_sphere = [rigid_sphere[i] for i in args.rigid_ico]

    os.makedirs(feat_out_dir, exist_ok=True)
    os.makedirs(rigid_dir, exist_ok=True)

    if len(args.target) == 1 and os.path.splitext(args.target[0])[1].lower() == ".tif":
        target_list, _, _ = read_tif(args.target[0], rigid_sphere[-1][0])
        target_list = [target_list[feat] for feat in args.rigid_feat]
    else:
        target_list = [read_feat(fn) for fn in args.target]

    target_list += [target_list[-1]] * (len(rigid_sphere) - len(target_list))
    rigid_feat = args.rigid_feat.copy()
    rigid_feat += [rigid_feat[-1]] * (len(rigid_sphere) - len(rigid_feat))
    data_dir = os.environ.get("SUBJECTS_DIR") if args.data_dir is None else args.data_dir
    print(f"Subject dir: {data_dir}")

    subj_list = sorted(next(os.walk(data_dir))[1])
    rot_mat = [np.eye(3)] * len(subj_list)

    rigid_config = {
        "feat": args.rigid_feat,
        "order": rigid_feat,
        "res": [rigid_sphere[i][0].shape[0] for i in range(len(rigid_sphere))],
        "intv": args.rigid_interval,
        "axis": args.rigid_axis_ico,
        "cand": args.rigid_cand,
    }
    rigid_target = []

    if not args.group_reg:
        print("Alignment to the template(s)...")
        rot_mat = Parallel(n_jobs=args.threads, backend="loky")(
            delayed(rigid_registration)(
                data_dir,
                args.feat_dir,
                args.native_sphere_dir,
                target_list,
                rigid_sphere,
                subj_name,
                args.hemi,
                args.native_sphere,
                mat,
                rigid_config,
            )
            for subj_name, mat in zip(subj_list, rot_mat)
        )
        for i in range(len(args.rigid_feat)):
            feat = target_list[i][: rigid_sphere[i if i < len(args.rigid_feat) - 1 else -1][0].shape[0]]
            rigid_target.append(feat)

    else:
        print("Template cration...")
        for i in range(args.iter):
            print(f"Rigid registration #{i}")
            rot_mat, target = iterative_mean(
                data_dir,
                args.feat_dir,
                args.native_sphere_dir,
                rigid_sphere,
                subj_list,
                args.hemi,
                args.native_sphere,
                rot_mat,
                rigid_config,
                args.threads,
            )

        print("Final alignment...")
        q_v, q_f = rigid_sphere[-1]
        rigid_mat = rigid_alignment(
            q_v,
            q_v,
            q_f,
            target[-1][: q_v.shape[0]],
            target_list[-1][: q_v.shape[0]],
            search_intv=args.rigid_interval,
            search_ico=args.rigid_axis_ico,
            search_topk=args.rigid_cand,
        )
        rot_mat = [mat @ rigid_mat.T for mat in rot_mat]

        print("Update template(s)...")
        for i in range(len(args.rigid_feat)):
            idx = i if i < len(args.rigid_feat) - 1 else -1
            q_v, q_f = rigid_sphere[idx]
            feat = retess(q_v, target[idx][None], TriangleSearch(q_v @ rigid_mat.T, q_f)).squeeze()
            rigid_target.append(feat)

    print("Retessellation...")
    Parallel(n_jobs=args.threads, backend="loky")(
        delayed(gen_data)(
            data_dir,
            args.out_dir,
            args.feat_dir,
            args.native_sphere_dir,
            subj_name,
            args.hemi,
            args.native_sphere,
            mat,
            args.in_ch,
            Icosphere(args.ico).sphere[-1][0],
        )
        for subj_name, mat in zip(subj_list, rot_mat)
    )

    rigid_sphere[-1][0].tofile(os.path.join(rigid_dir, f"{args.hemi}.vert.dat"))
    for target, feat in zip(rigid_target, args.rigid_feat):
        target.astype(float).tofile(os.path.join(rigid_dir, f"{args.hemi}.{feat}.dat"))

    with open(os.path.join(rigid_dir, f"{args.hemi}.config.json"), "w") as fd:
        json.dump(rigid_config, fd, indent=2)


if __name__ == "__main__":
    main(args())
