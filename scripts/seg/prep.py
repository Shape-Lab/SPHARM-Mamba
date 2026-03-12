"""
Example script for data preparation.

If you use this code, please cite the following paper.

    Seungbo Ha and Ilwoo Lyu
    SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation.
    IEEE Transactions on Medical Imaging, 41(10), 2739-2751, 2022

Copyright 2022 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import argparse
import numpy as np
from joblib import Parallel, delayed

from spharmdnns.utils import parse_args, save_config
from spharmdnns.sphere import Icosphere, TriangleSearch
from spharmdnns.io import read_feat, read_mesh, read_annot


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for data generation", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads for parallel data generation")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--data-dir", type=str, help="Path to FreeSurfer home (default: $SUBJECTS_DIR)")
    group.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for parcellation")
    group.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    group.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh (sphere, sphere.reg, etc.)")
    group.add_argument("--annot", type=str, default="aparc", help="Manual labels (e.g. aparc for ?h.aparc.annot)")

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H","thickness"], nargs="+", help="List of geometry")
    group.add_argument("--out-dir", type=str, default="dataset", help="Path to re-tessellated data (output)")
    group.add_argument("--ico", type=int, default=6, help="Icosaheral subdivision of reference sphere for re-tessellation")
    group.add_argument("--sphere", type=str, default=None, help="Reference sphere mesh for re-tessellation (vtk or FreeSurfer format)")
    group.add_argument("--label-dir", type=str, default="label", help="Path to target labels")

    args = parse_args(parser, section_prefix="seg.prep")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="seg.prep")

    return args


def gen_data(data_dir, feat_out_dir, label_out_dir, csv_out_dir, feat_dir, label_dir, native_sphere_dir, subj_name, hemisphere, native_sphere, in_ch, ico_v, annot_file):
    print(f"Processing {subj_name}...")

    for hemi in hemisphere:
        native_v, native_f = read_mesh(os.path.join(data_dir, subj_name, native_sphere_dir, hemi + "." + native_sphere))
        tree = TriangleSearch(native_v, native_f)
        fid, bary = tree.query(ico_v)

        # Generating features
        for feat_name in in_ch:
            feat = read_feat(os.path.join(data_dir, subj_name, feat_dir, hemi + "." + feat_name))
            feat = (feat[native_f[fid]] * bary).sum(-1)
            feat.tofile(os.path.join(feat_out_dir, f"{subj_name}.{hemi}.aug0.{feat_name}.dat"))

        # Generating labels
        # vID, label, struct, structID = read_annot(os.path.join(data_dir, subj_name, label_dir, hemi + "." + annot_file + ".annot"))

        # # If 'label' contains unidenfied label, map those elements to 0
        # structID_to_index = {val: idx for idx, val in enumerate(structID)}
        # annot = np.zeros(native_v.shape[0], dtype=np.int16)
        # annot[vID] = np.vectorize(lambda x: structID_to_index.get(x, 0))(label)
        # annot = annot[native_f[fid, bary.argmax(axis=1)]]
        # annot.tofile(os.path.join(label_out_dir, f"{subj_name}.{hemi}.aug0.label.dat"))

        # # write csv file
        # struct = np.column_stack((struct, np.arange(len(struct))))
        # np.savetxt(os.path.join(csv_out_dir, f"{subj_name}.{hemi}.csv"), struct, delimiter=",", header="label,ID", comments="", fmt="%s")


def main(args):
    feat_out_dir = os.path.join(args.out_dir, "features")
    label_out_dir = os.path.join(args.out_dir, "labels")
    csv_out_dir = os.path.join(args.out_dir, "labelmaps")

    os.makedirs(feat_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(csv_out_dir, exist_ok=True)

    data_dir = os.environ.get("SUBJECTS_DIR") if args.data_dir is None else args.data_dir
    print(f"Subject dir: {data_dir}")

    subj_list = sorted(next(os.walk(data_dir))[1])

    if args.sphere:
        ico_v, _ = read_mesh(args.sphere)
    else:
        ico_v, _ = Icosphere(args.ico).sphere[-1]

    Parallel(n_jobs=args.threads, backend="loky")(
        delayed(gen_data)(
            data_dir,
            feat_out_dir,
            label_out_dir,
            csv_out_dir,
            args.feat_dir,
            args.label_dir,
            args.native_sphere_dir,
            subj_name,
            args.hemi,
            args.native_sphere,
            args.in_ch,
            ico_v,
            args.annot,
        )
        for subj_name in subj_list
    )


if __name__ == "__main__":
    main(args())
