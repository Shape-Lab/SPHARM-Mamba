"""
Example script for evaluation of spherical registration based on similarity and distortion metrics.

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

import os
import glob
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

from spharmdnns.io import read_mesh, read_feat, read_tif
from spharmdnns.utils import parse_args, save_config, eval_distortion
from spharmdnns.sphere import TriangleSearch, Icosphere, retess
from spharmdnns.sphere import vertex_area, face_area, edge_length


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Requirements
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, choices=["lh", "rh"], help="Hemisphere for registration", required=True)
    group.add_argument("--subj-dir", type=str, nargs="+", help="List of FreeSurfer's subject paths", required=True)
    group.add_argument("--target", type=str, help="Target geometry (tif or list of features tessellated by the ShapeLab icospheres)", required=True)
    group.add_argument("--feature", type=str, nargs="+", help="Features to be evaluated", required=True)
    group.add_argument("--sphere-reg", type=str, nargs="+", help="Registered sphere mesh", required=True)

    # Options
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--area-type", type=str, default="vert", choices=["vert", "face"], help="vert (per-vertex) or face (per-triangle)")
    group.add_argument("--area-eps", type=float, default=0, help="Threshold below which computed areas are treated as zero")
    group.add_argument("--ico", type=int, default=7, help="Level of icosahedral subdivision")
    group.add_argument("--individual", action="store_true", help="Report statistics for each subject individually")
    group.add_argument("--precision", type=int, default=4, help="Number of decimal places to display in outputs")
    group.add_argument("--verbose", action="store_true", help="Print a detailed summary to the console")
    group.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    group.add_argument("--out-file", type=str, help="Save console output to file")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")

    # Input
    group = parser.add_argument_group("input")
    group.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for registration metrics")
    group.add_argument("--sphere-dir", type=str, default="surf", help="Path to sphere")
    group.add_argument("--sphere", type=str, default="sphere", help="Native sphere mesh")
    group.add_argument("--title", type=str, nargs="*", help="Custom titles corresponding to each file in --sphere-reg (order-sensitive)")

    args = parse_args(parser, section_prefix="eval.reg_metric")

    if args.save_config:
        save_config(args, parser, args.save_config, exclude_keys=["config", "save_config"], section_prefix="eval.reg_metric")

    return args


def metric_line(label, value, std=None, log_value=None, log_std=None, raw_data=None, precision=4, last=False):
    branch = "    └─" if last else "    ├─"
    line = f"  {branch} {label:<7}: {value:.{precision}f}"

    if std is not None:
        line += f" ± {std:.{precision}f}"

    if log_value is not None:
        line += f"  (log₂: {log_value:.{precision}f}"
        if log_std is not None:
            line += f" ± {log_std:.{precision}f}"
        line += ")"

    if raw_data is not None:
        line += f"  (intv: {raw_data.min():.{precision}f} - {raw_data.max():.{precision}f})"

    return line + "\n"


def eval_metrics(
    id, subj, hemi, sphere_dir, sphere, sphere_reg, feat_dir, feat, geom, q, x, x0, x0_norm, area_func, title, metric_names, eps, precision
):
    log = f"== {id}. {subj.split('/')[-1]} ==\n"

    fn = f"{subj}/{sphere_dir}/{hemi}.{sphere}"
    if os.path.exists(fn):
        v0, f = read_mesh(fn)
        v0 /= np.linalg.norm(v0, axis=-1, keepdims=True)
    else:
        print(f"{fn} is not found!")
        return (np.nan,) * len(metric_names) + (None,)

    area0 = area_func(v0, f)
    edge0 = edge_length(v0, f)
    idx = np.where(area0 >= eps)
    area0 = area0[idx]

    metrics = {name: {method: np.nan for method in title} for name in metric_names}

    for fn, method in [(f"{subj}/{sphere_dir}/{hemi}.{reg}", method) for reg, method in zip(sphere_reg, title)]:
        log += f"- {method}\n"
        y = np.vstack([read_feat(f"{subj}/{feat_dir}/{hemi}.{f}") for f in feat])

        if os.path.exists(fn):
            v, _ = read_mesh(fn)
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
        else:
            print(f"{fn} is not found!")
            continue

        y = retess(q, y, TriangleSearch(v, f))
        y0 = y - y.mean(-1, keepdims=True)
        metrics["ncc"][method] = np.sum(x0 * y0, -1) / (x0_norm * np.linalg.norm(y0, axis=-1))
        metrics["mse"][method] = np.mean((x - y) ** 2, -1)
        metrics["mae"][method] = np.mean(np.abs(x - y), -1)

        area = area_func(v, f)[idx]
        area_all = area / area0
        area_all[area_all < 1] = 1 / area_all[area_all < 1]

        edge = edge_length(v, f)
        edge_all = edge / edge0
        edge_all[edge_all < 1] = 1 / edge_all[edge_all < 1]

        _, shape_all = eval_distortion(v0, v, f)

        arrays = {
            "area": area_all,
            "area_log": np.log2(area_all),
            "shape": shape_all,
            "shape_log": np.log2(shape_all),
            "edge": edge_all,
            "edge_log": np.log2(edge_all),
        }

        for cat in geom:
            metrics[f"{cat}_mean"][method] = np.mean(arrays[cat])
            metrics[f"{cat}_median"][method] = np.median(arrays[cat])
            metrics[f"{cat}_pr"][method] = np.percentile(arrays[cat], 99.73)
            metrics[f"{cat}_max"][method] = np.max(arrays[cat])

            metrics[f"{cat}_log_mean"][method] = np.mean(arrays[f"{cat}_log"])
            metrics[f"{cat}_log_median"][method] = np.median(arrays[f"{cat}_log"])
            metrics[f"{cat}_log_pr"][method] = np.percentile(arrays[f"{cat}_log"], 99.73)
            metrics[f"{cat}_log_max"][method] = np.max(arrays[f"{cat}_log"])

        log += f"  1) Similarity Metrics\n"
        for i, name in enumerate(feat):
            log += f"    • {name}\n"
            log += metric_line("NCC", metrics["ncc"][method][i], precision=precision)
            log += metric_line("MSE", metrics["mse"][method][i], precision=precision)
            log += metric_line("MAE", metrics["mae"][method][i], precision=precision, last=True)

        log += f"  2) Distortion Metrics\n"
        for name in geom:
            log += f"    • {name.capitalize()}\n"
            log += metric_line("Mean", metrics[f"{name}_mean"][method], log_value=metrics[f"{name}_log_mean"][method], precision=precision)
            log += metric_line("Median", metrics[f"{name}_median"][method], log_value=metrics[f"{name}_log_median"][method], precision=precision)
            log += metric_line("99.73%", metrics[f"{name}_pr"][method], log_value=metrics[f"{name}_log_pr"][method], precision=precision)
            log += metric_line("Max", metrics[f"{name}_max"][method], log_value=metrics[f"{name}_log_max"][method], precision=precision, last=True)
        log += "\n"

    return metrics, log


def summary(batches, feat, geom, title, precision):
    log = ""
    for method in title:
        log += f"- {method}\n"
        log += f"  1) Similarity Metrics\n"
        for i, name in enumerate(feat):
            log += f"    • {name}\n"
            log += metric_line(
                "NCC",
                batches["ncc"][method][:, i].mean(),
                std=batches["ncc"][method][:, i].std(),
                raw_data=batches["ncc"][method][:, i],
                precision=precision,
            )
            log += metric_line(
                "MSE",
                batches["mse"][method][:, i].mean(),
                std=batches["mse"][method][:, i].std(),
                raw_data=batches["mse"][method][:, i],
                precision=precision,
            )
            log += metric_line(
                "MAE",
                batches["mae"][method][:, i].mean(),
                std=batches["mae"][method][:, i].std(),
                raw_data=batches["mae"][method][:, i],
                precision=precision,
                last=True,
            )

        log += f"  2) Distortion Metrics\n"
        for i, name in enumerate(geom):
            log += f"    • {name.capitalize()}\n"
            log += metric_line(
                "Mean",
                np.nanmean(batches[f"{name}_mean"][method]),
                np.nanstd(batches[f"{name}_mean"][method]),
                np.nanmean(batches[f"{name}_log_mean"][method]),
                np.nanstd(batches[f"{name}_log_mean"][method]),
                precision=precision,
            )
            log += metric_line(
                "Median",
                np.nanmean(batches[f"{name}_median"][method]),
                np.nanstd(batches[f"{name}_median"][method]),
                np.nanmean(batches[f"{name}_log_median"][method]),
                np.nanstd(batches[f"{name}_log_median"][method]),
                precision=precision,
            )
            log += metric_line(
                "99.73%",
                np.nanmean(batches[f"{name}_pr"][method]),
                np.nanstd(batches[f"{name}_pr"][method]),
                np.nanmean(batches[f"{name}_log_pr"][method]),
                np.nanstd(batches[f"{name}_log_pr"][method]),
                precision=precision,
            )
            log += metric_line(
                "Max",
                np.nanmean(batches[f"{name}_max"][method]),
                np.nanstd(batches[f"{name}_max"][method]),
                np.nanmean(batches[f"{name}_log_max"][method]),
                np.nanstd(batches[f"{name}_log_max"][method]),
                precision=precision,
                last=True,
            )
        log += "\n"

    return log


def main(args):
    sphere_dir = args.sphere_dir
    feat_dir = args.feat_dir
    sphere = args.sphere
    sphere_reg = args.sphere_reg
    threads = args.threads
    hemi = args.hemi
    target = args.target
    title = args.title if args.title is not None else args.sphere_reg
    eps = args.area_eps

    feat = args.feature
    precision = args.precision
    sim = ["ncc", "mse", "mae"]
    geom = ["area", "shape", "edge"]
    suffixes = ["mean", "median", "pr", "max", "log_mean", "log_median", "log_pr", "log_max"]
    distor = [f"{cat}_{suffix}" for cat in geom for suffix in suffixes]
    metric_names = sim + distor

    area_func = vertex_area if args.area_type == "vert" else face_area

    q = Icosphere(args.ico).sphere[-1][0]
    x, _, _ = read_tif(target, q)
    x = np.vstack([x[f] for f in feat])
    x0 = x - x.mean(-1, keepdims=True)
    x0_norm = np.linalg.norm(x0, axis=-1)

    subj_list = list(itertools.chain.from_iterable(glob.glob(fn) for fn in args.subj_dir))
    if not subj_list:
        raise FileNotFoundError("No subject folders found to evaluate.")

    print_individual = len(subj_list) == 1 or args.individual

    metrics = Parallel(n_jobs=min(threads, len(subj_list)))(
        delayed(eval_metrics)(
            i, subj, hemi, sphere_dir, sphere, sphere_reg, feat_dir, feat, geom, q, x, x0, x0_norm, area_func, title, metric_names, eps, precision
        )
        for i, subj in enumerate(subj_list)
    )

    batches = {}
    for name in sim:
        batches[name] = {method: np.full((len(subj_list), len(feat)), np.nan, dtype=float) for method in title}
    for name in distor:
        batches[name] = {method: np.full((len(subj_list), 1), np.nan, dtype=float) for method in title}
    log = ""
    for i, (metric_dict, log_line) in enumerate(metrics):
        if print_individual:
            log += log_line
        for name in metric_names:
            for method in title:
                batches[name][method][i, :] = metric_dict[name][method]

    if len(subj_list) > 1:
        log += f"== Summary (N={len(subj_list)}) ==\n"
        log += summary(batches, feat, geom, title, precision)

    if args.out_file is not None:
        with open(args.out_file, "w") as f:
            f.write(log)

    if args.verbose or args.out_file is None:
        print(log, end="")


if __name__ == "__main__":
    main(args())
