"""
Example script for evaluation of spherical registration based on cortical parecllation labels.

If you use this code, please cite the following paper.

    Seungeun Lee, Seunghwan Lee, Sunghwa Ryu, and Ilwoo Lyu
    SPHARM-Reg: Unsupervised Cortical Surface Registration using Spherical Harmonics.
    IEEE Transactions on Medical Imaging

Copyright 2025 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import glob
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
from joblib import Parallel, delayed

from spharmdnns.io import read_annot, read_mesh
from spharmdnns.utils import parse_args, save_config, eval_dice, squeeze_label
from spharmdnns.sphere import TriangleSearch, vertex_area, retess


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for registration", required=True)
    group.add_argument("--subj-dir", type=str, nargs="+", help="List of FreeSurfer's subject paths", required=True)
    group.add_argument("--sphere-reg", type=str, help="Registered sphere mesh", required=True)
    group.add_argument("--annot", type=str, help="Manual labels (e.g. aparc for ?h.aparc.annot)", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--classes", type=int, nargs="+", help="List of regions of interest, automatically detected when unspecified")
    group.add_argument("--individual", action="store_true", help="Report statistics for each subject individually")
    group.add_argument("--precision", type=int, default=4, help="Number of decimal places to display in outputs")
    group.add_argument("--column-per-row", type=int, default=18, help="Column per row")
    group.add_argument("--verbose", action="store_true", help="Print a detailed summary to the console")
    group.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    group.add_argument("--out-file", type=str, help="Save ROI-wise Dice scores per subject as csv")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--label-dir", type=str, default="label", help="Path to annotation")
    group.add_argument("--sphere-dir", type=str, default="surf", help="Path to sphere")
    group.add_argument("--sphere", type=str, default="sphere", help="Native sphere mesh")

    args = parse_args(parser, section_prefix="eval.reg_parc")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="eval.reg_parc")

    return args


def read_data(subj, hemi, sphere_dir, sphere, sphere_reg, label_dir, annot_prefix):
    vID, label, _, structID = read_annot(f"{subj}/{label_dir}/{hemi}.{annot_prefix}.annot")
    structID_to_index = {val: idx for idx, val in enumerate(structID)}
    annot = np.zeros(vID.shape[0], dtype=int)
    annot[vID] = np.vectorize(lambda x: structID_to_index.get(x, 0))(label)

    v, f = read_mesh(f"{subj}/{sphere_dir}/{hemi}.{sphere}")
    area = vertex_area(v, f, norm=True)

    v, f = read_mesh(f"{subj}/{sphere_dir}/{hemi}.{sphere_reg}")
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    return annot, v, f, area


def subj_dice(subj_list, data, id):
    target, q, _, area = data[id]
    resample = []
    for k, (input, v, f, _) in enumerate(tqdm(data, desc=f"{subj_list[id]}")):
        if id == k:
            continue
        resample.append(retess(q, input, TriangleSearch(v, f), nearest=True))

    input = mode(np.stack(resample), keepdims=False)[0]
    dice = eval_dice(input, target, area=area)

    return dice


def summary(subj_list, classes, lut_old2new, dice, show_subj=True, precision=4, column_per_row=18):
    log = ""
    if show_subj:
        log += "Subject-wise scores:\n"
        indentation = max(len(subj) for subj in subj_list)
        n_subj = len(subj_list)

        for start in range(0, len(classes), column_per_row):
            end = start + column_per_row
            class_chunk = classes[start:end]
            col_indices = [lut_old2new[c] for c in class_chunk]

            header = " " * (indentation + 6 - precision // 2) + " ".join(f"{c:>{precision+2}}" for c in class_chunk)
            log += f"{header}\n"

            for i in range(n_subj):
                row_chunk = dice[i, col_indices]
                chunk_vals = " ".join(f"{v:.{precision}f}" if not np.isnan(v) else f"{'nan':>{precision+2}}" for v in row_chunk)

                if start == 0:
                    row_mean = np.nanmean(dice[i])
                    row_std = np.nanstd(dice[i])
                    prefix = " └─" if i == n_subj - 1 else " ├─"
                    log += f"{prefix} {subj_list[i]:<{indentation}}: {chunk_vals} | {row_mean:.{precision}f} ± {row_std:.{precision}f}\n"
                else:
                    log += " " * (indentation + 6) + chunk_vals + "\n"
            log += "\n"

    class_mean = np.nanmean(dice, axis=0)
    class_std = np.nanstd(dice, axis=0)

    log += "ROI-wise scores:\n"
    for start in range(0, len(classes), column_per_row):
        end = start + column_per_row
        class_chunk = classes[start:end]
        mean_chunk = class_mean[start:end]
        std_chunk = class_std[start:end]

        header = " " * (8 - precision // 2) + " ".join(f"{c:>{precision+2}}" for c in class_chunk)
        means = "  Mean: " + " ".join(f"{m:.{precision}f}" if not np.isnan(m) else f"{'nan':>{precision+2}}" for m in mean_chunk)
        stds = "   Std: " + " ".join(f"{s:.{precision}f}" if not np.isnan(s) else f"{'nan':>{precision+2}}" for s in std_chunk)

        if start == 0:
            total_mean = np.nanmean(dice)
            total_std = np.nanstd(dice)
            log += f"{header}\n{means} | {total_mean:.{precision}f}\n{stds} | {total_std:.{precision}f}\n"
        else:
            log += f"{header}\n{means}\n{stds}\n"

    return log


def main(args):
    hemi = args.hemi
    sphere_dir = args.sphere_dir
    label_dir = args.label_dir
    sphere = args.sphere
    sphere_reg = args.sphere_reg
    threads = args.threads
    annot_prefix = args.annot
    precision = args.precision
    column_per_row = args.column_per_row

    subj_list = list(itertools.chain.from_iterable(glob.glob(fn) for fn in args.subj_dir))

    if not subj_list or len(subj_list) == 1:
        raise FileNotFoundError("At least two subject folders needed to evaluate.")

    print_individual = args.individual

    data = Parallel(n_jobs=min(threads, len(subj_list)))(
        delayed(read_data)(subj, hemi, sphere_dir, sphere, sphere_reg, label_dir, annot_prefix) for subj in subj_list
    )
    if args.classes is not None:
        classes = args.classes
    else:
        classes = np.unique(data[0][0])
        for i in range(1, len(data)):
            classes = np.intersect1d(classes, np.unique(data[i][0]))

    lut_old2new, _ = squeeze_label(classes)
    for i in range(len(data)):
        data[i] = (np.vectorize(lambda x: lut_old2new.get(x, 0))(data[i][0]),) + data[i][1:]

    subj_list = [subj.split("/")[-1] for subj in subj_list]
    dice = Parallel(n_jobs=min(threads, len(subj_list)))(delayed(subj_dice)(subj_list, data, i) for i in range(len(subj_list)))
    dice = np.stack(dice)

    if args.out_file is not None:
        np.savetxt(
            args.out_file, np.column_stack((subj_list, dice)), delimiter=",", fmt="%s", header="ID," + ",".join(map(str, classes)), comments=""
        )

    log = summary(subj_list, classes, lut_old2new, dice, show_subj=print_individual, precision=precision, column_per_row=column_per_row)

    if args.verbose or args.out_file is None:
        print(log)


if __name__ == "__main__":
    main(args())
