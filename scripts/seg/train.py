"""
Example script for training SPHARM-Net.

If you use this code, please cite the following paper.

    Seungbo Ha and Ilwoo Lyu
    SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation.
    IEEE Transactions on Medical Imaging, 41(10), 2739-2751, 2022
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack
from packaging import version
import math  
import time  


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_
from spharmdnns.utils import Logger, parse_args, save_config
from spharmdnns.io import read_mesh
from spharmdnns.sphere import Icosphere


from spharmdnns.utils.loader import SphericalDataset


from spharmdnns.core.spharm_mamba import SPHARMMambaAge


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset & dataloader
    group = parser.add_argument_group("required arguments")
    group.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for learning", required=True)
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
    group.add_argument("--in-ch", type=str, default=["curv", "sulc"], nargs="+", help="List of geometry")
    group.add_argument("--data-norm", action="store_true", help="Z-score+prctile data normalization (x features)")
    group.add_argument("--n-splits", type=int, default=5, help="A total of cross-validation folds")
    group.add_argument("--fold", type=int, default=1, help="Cross-validation fold")
    group.add_argument("--aug", type=int, default=0, help="Level of data augmentation")

    # Training
    group = parser.add_argument_group("training")
    group.add_argument("--epochs", type=int, default=20, help="Max epoch")
    group.add_argument("--batch-size", type=int, default=1, help="Batch size")
    group.add_argument("--lr", "--learning-rate", type=float, default=0.0001, help="Initial learning rate")
    group.add_argument("--decay", type=int, default=2, help="Decay after every # epochs if no progress (0: no decay)")
    group.add_argument("--det", action="store_true", help="Enable deterministic algorithms for reproducibility")

    # Loss & target space
    group = parser.add_argument_group("loss/target")
    group.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "smoothl1"], help="mse: MSELoss, l1: L1Loss, smoothl1: SmoothL1 (Huber)")
    group.add_argument("--smoothl1-beta", type=float, default=1.0, help="beta for SmoothL1Loss (Huber)")
    group.add_argument(
        "--true-age",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="True이면 실제 나이(year)로 학습. False이면 age z-score로 학습."
    )

    # Output
    group = parser.add_argument_group("output")
    group.add_argument("--log-dir", type=str, default="log_age_enc", help="Path to the log files (output)")
    group.add_argument("--log-format", type=str, default="12.6", help="Log format specification")
    group.add_argument("--ckpt-dir", type=str, default="log_age_enc", help="Path to the checkpoint file (output)")
    group.add_argument("--resume", type=str, default=None, help="Checkpoint (pt) to resume training")

    # SPHARM-Net settings
    group = parser.add_argument_group("backbone")
    group.add_argument("-D", "--depth", type=int, default=3, help="Depth of SPHARM-Net encoder")
    group.add_argument("-C", "--channel", type=int, default=128, help="# of channels in the entry layer")
    group.add_argument("-L", "--bandwidth", type=int, default=80, help="Bandwidth")
    group.add_argument("--interval", type=int, default=5, help="Anchor interval of harmonic coefficients")
    group.add_argument("--ico", type=int, default=6, help="Level of icosahedral subdivision")
    group.add_argument("--sphere", type=str, default=None, help="Sphere mesh (vtk or FreeSurfer format)")

    # keep section prefix as seg.train to reuse parse_args behavior in your repo
    args_ = parse_args(parser, section_prefix="seg.train")

    if args_.save_config:
        save_config(args_, parser, args_.save_config, exclude_keys=["config", "save_config"], section_prefix="seg.train")

    return args_


@torch.no_grad()
def _metrics(pred, y):
    # pred, y: [B]
    mae = torch.mean(torch.abs(pred - y)).item()
    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()
    return mae, rmse


def step(model, loader, device, criterion, epoch, logger, log_format, optimizer=None, pbar=None,
         true_age=True, age_mean=None, age_std=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    progress = lambda x: tqdm(x, desc=f"Epoch {str(pbar)}") if pbar is not None else x

    running_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_iter = 0

    with torch.no_grad() if optimizer is None else ExitStack():
        for x, y_z, _ in progress(loader):
            x = x.to(device)              # [B, in_ch, V]
            y_z = y_z.to(device).float()  # [B, 1]  (loader는 항상 z를 리턴)

            
            if true_age:
                if age_mean is None or age_std is None:
                    raise RuntimeError("true-age=True인데 dataset에서 age_mean/std를 못 받았습니다.")
                y = y_z * age_std + age_mean   # [B,1] year
            else:
                y = y_z                        # [B,1] z

            if optimizer is not None:
                optimizer.zero_grad()

            
            out = model(x)            # [B,1] 또는 [B]
            if out.dim() == 2 and out.size(1) == 1:
                pred = out.squeeze(1)     # [B]
            else:
                pred = out               # [B]
            target = y.squeeze(1)        # [B]

            loss = criterion(pred, target)

            if optimizer is not None:
                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1.0)      ####이걸로 gradient clipping 조절
                optimizer.step()

            running_loss += loss.item()
            mae, rmse = _metrics(pred.detach(), target.detach())
            total_mae += mae
            total_rmse += rmse
            n_iter += 1

    avg_loss = running_loss / max(n_iter, 1)
    avg_mae = total_mae / max(n_iter, 1)
    avg_rmse = total_rmse / max(n_iter, 1)

    logger.write([epoch, avg_loss, avg_mae, avg_rmse], log_format)
    return avg_mae


def write_predict_age_log_subjectwise(model, dataset_train, device, out_path, batch_size,
                                      true_age, age_mean, age_std):
    """
    predict_age.log 형식:
      subj<TAB>pred_age<TAB>true_age

    - dataset_train에는 aug가 포함되어 subj가 중복될 수 있으므로,
      subj별로 pred_age를 평균내서 1줄만 저장.
    - true_age는 항상 year 단위로 기록.
    """
    model.eval()
    infer_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=False)

    agg = {}

    with torch.no_grad():
        for x, y_z, subj in tqdm(infer_loader, desc="Predict train (subjectwise)"):
            x = x.to(device)
            y_z = y_z.to(device).float()  # [B,1] z

            true_age_year = (y_z * age_std + age_mean).squeeze(1)  # [B]

            out = model(x)  # [B,1] 또는 [B]
            if out.dim() == 2 and out.size(1) == 1:
                out = out.squeeze(1)  # [B]

            if true_age:
                pred_age_year = out
            else:
                pred_age_year = out * age_std + age_mean

            B = pred_age_year.shape[0]
            for i in range(B):
                s = subj[i]
                pa = float(pred_age_year[i].item())
                ta = float(true_age_year[i].item())

                if s not in agg:
                    agg[s] = {"sum_pred": 0.0, "sum_true": 0.0, "cnt": 0}
                agg[s]["sum_pred"] += pa
                agg[s]["sum_true"] += ta
                agg[s]["cnt"] += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("subj\tpred_age\ttrue_age\n")
        for s in sorted(agg.keys()):
            cnt = max(agg[s]["cnt"], 1)
            pred_mean = agg[s]["sum_pred"] / cnt
            true_mean = agg[s]["sum_true"] / cnt
            f.write(f"{s}\t{pred_mean:.6f}\t{true_mean:.6f}\n")


@torch.no_grad()
def compute_pearson_r_and_r2(model, loader, device, true_age, age_mean, age_std):
    model.eval()

    preds = []
    trues = []

    for x, y_z, _ in loader:
        x = x.to(device)
        y_z = y_z.to(device).float()  # [B,1] z

        true_year = (y_z * age_std + age_mean).squeeze(1)  # [B]

        out = model(x)  # [B,1] or [B]
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)

        if true_age:
            pred_year = out
        else:
            pred_year = out * age_std + age_mean

        preds.append(pred_year.detach().cpu())
        trues.append(true_year.detach().cpu())

    pred_all = torch.cat(preds, dim=0).float()
    true_all = torch.cat(trues, dim=0).float()

    pred_centered = pred_all - pred_all.mean()
    true_centered = true_all - true_all.mean()

    denom = torch.sqrt(torch.sum(pred_centered ** 2)) * torch.sqrt(torch.sum(true_centered ** 2))
    if float(denom.item()) == 0.0:
        r = float("nan")
    else:
        r = float((torch.sum(pred_centered * true_centered) / denom).item())

    r2 = r * r
    return r, r2


def append_final_r_to_log(log_path, r, r2):
    with open(log_path, "a") as f:
        f.write(f"Final(best_ckpt)\tPearsonR\t{r:.6f}\tR2\t{r2:.6f}\n")



@torch.no_grad()
def collect_pred_true_year(model, loader, device, true_age, age_mean, age_std):
    model.eval()
    preds = []
    trues = []

    for x, y_z, _ in loader:
        x = x.to(device)
        y_z = y_z.to(device).float()  # [B,1] z

        true_year = (y_z * age_std + age_mean).squeeze(1)  # [B]

        out = model(x)  # [B,1] or [B]
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)  # [B]

        if true_age:
            pred_year = out
        else:
            pred_year = out * age_std + age_mean

        preds.append(pred_year.detach().cpu())
        trues.append(true_year.detach().cpu())

    pred_all = torch.cat(preds, dim=0).float().numpy()
    true_all = torch.cat(trues, dim=0).float().numpy()
    return pred_all, true_all


def save_pred_vs_true_hexbin(pred, true, out_png_path, title="Predicted vs Chronological Age", gridsize=60):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    if pred.size == 0 or true.size == 0:
        return

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    true_c = true - true.mean()
    pred_c = pred - pred.mean()
    denom = (np.sqrt(np.sum(true_c ** 2)) * np.sqrt(np.sum(pred_c ** 2)))
    r = float(np.sum(true_c * pred_c) / denom) if denom > 0 else float("nan")
    r2 = r * r

    slope, intercept = np.polyfit(true, pred, 1)

    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    pad = 0.02 * (hi - lo + 1e-6)
    lo -= pad
    hi += pad

    fig = plt.figure(figsize=(7.2, 5.6), dpi=200)
    ax = fig.add_subplot(111)

    hb = ax.hexbin(true, pred, gridsize=gridsize, mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Counts")

    # equality line
    ax.plot([lo, hi], [lo, hi], linewidth=2, label="Equality line")

    # best-fit line
    xs = np.array([lo, hi], dtype=np.float32)
    ys = slope * xs + intercept
    ax.plot(xs, ys, linestyle="--", linewidth=2, label="Linear line of best fit")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title(title)
    ax.legend(loc="lower right")

    txt = (
        f"MAE={mae:.3f}\n"
        f"RMSE={rmse:.3f}\n"
        f"Pearson r={r:.3f}\n"
        f"r^2={r2:.3f}\n"
        f"slope={slope:.3f}, intercept={intercept:.3f}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.8),
    )

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png_path)
    plt.close(fig)



@torch.no_grad()
def benchmark_inference_forward_only(
    model,
    sample_x,               
    device,
    warmup=50,
    iters=200,
):
    """
    Forward-only benchmark:
    - DataLoader/CPU->GPU 전송 제외하고, GPU 상에서 model(x) 순수 forward 시간 측정
    - latency(ms/sample), throughput(samples/s), peak memory(MB) 반환

    NOTE:
    - GPU일 때만 cuda.Event / peak mem 사용. CPU면 perf_counter로만 측정.
    """
    model.eval()

    
    x = sample_x.to(device, non_blocking=True)

    is_cuda = (device.type == "cuda")

    
    if is_cuda:
        torch.cuda.synchronize()
    for _ in range(int(warmup)):
        _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    
    peak_mb = float("nan")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # 4) timing
    bs = int(x.shape[0])

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        for _ in range(int(iters)):
            _ = model(x)
        ender.record()

        torch.cuda.synchronize()
        total_ms = float(starter.elapsed_time(ender))  

        
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = float(peak_bytes) / (1024.0 ** 2)

    else:
        t0 = time.perf_counter()
        for _ in range(int(iters)):
            _ = model(x)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

    ms_per_iter = total_ms / float(iters)
    ms_per_sample = ms_per_iter / max(bs, 1)
    throughput_sps = 1000.0 / ms_per_sample if ms_per_sample > 0 else float("inf")

    return {
        "latency_ms_per_sample": float(ms_per_sample),
        "throughput_samples_per_sec": float(throughput_sps),
        "peak_mem_mb": float(peak_mb),
        "warmup": int(warmup),
        "iters": int(iters),
        "batch_size": int(bs),
        "input_shape": list(x.shape),
        "device": str(device),
    }


def main(args_):
    if args_.resume:
        checkpoint = torch.load(
            args_.resume,
            **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {})
        )
        if checkpoint.get("epoch", 0) >= args_.epochs:
            print("Training has already been completed.")
            return

    args_.no_cuda |= not torch.cuda.is_available()
    device = torch.device("cpu" if args_.no_cuda else f"cuda:{args_.gpu}")
    preload = None if args_.preload == "none" else device if args_.preload == "device" else args_.preload

    torch.set_num_threads(args_.threads)

    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    if not args_.no_cuda:
        torch.cuda.manual_seed(args_.seed)

    if args_.det:
        if version.parse(torch.__version__) >= version.parse("1.9.0"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        else:
            print("Warning: SPHARM-Net uses non-deterministic algorithms.")

    print("Loading data...")
    if args_.sphere is not None:
        v, f = read_mesh(os.path.join(args_.sphere))
    else:
        v, f = Icosphere(args_.ico).sphere[-1]

    partition = ["train", "val", "test"]
    dataset = {}
    for p in partition:
        dataset[p] = SphericalDataset(
            data_dir=args_.data_dir,
            partition=p,
            fold=args_.fold,
            n_vert=v.shape[0],
            classes=None,
            seed=args_.seed,
            aug=args_.aug,
            n_splits=args_.n_splits,
            hemi=args_.hemi,
            in_ch=args_.in_ch,
            data_norm=args_.data_norm,
            preload=preload,
            task="age",
        )

    loader = {
        "train": DataLoader(dataset["train"], batch_size=args_.batch_size, shuffle=True, drop_last=False),
        "val":   DataLoader(dataset["val"], batch_size=args_.batch_size, shuffle=False, drop_last=False),
        "test":  DataLoader(dataset["test"], batch_size=args_.batch_size, shuffle=False, drop_last=False),
    }

    age_mean = dataset["train"].age_mean
    age_std = dataset["train"].age_std
    if age_mean is None or age_std is None:
        raise RuntimeError("dataset.train에서 age_mean/std를 제공하지 않습니다.")
    age_mean = torch.tensor(age_mean, device=device, dtype=torch.float32)
    age_std = torch.tensor(age_std, device=device, dtype=torch.float32)

    os.makedirs(args_.log_dir, exist_ok=True)
    logger = {p: Logger(os.path.join(args_.log_dir, p + ".log")) for p in partition}
    for p in partition:
        if p == "train":
            logger[p].write(vars(args_))
        logger[p].write({"fold_data": dataset[p].subj_list})
        if args_.true_age:
            logger[p].write(["Epoch", "Loss", "MAE(year)", "RMSE(year)"], args_.log_format)
        else:
            logger[p].write(["Epoch", "Loss", "MAE(z)", "RMSE(z)"], args_.log_format)

    print("Loading model (SPHARM+Mamba age head)...")
    model = SPHARMMambaAge(
        device=device,
        sphere=(v, f),
        in_ch=len(args_.in_ch),
        L=args_.bandwidth,
        interval=args_.interval,
        threads=args_.threads,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of params {params}")

    optimizer = optim.AdamW(model.parameters(), lr=args_.lr, weight_decay=2.0e-3)

    
    warmup_epochs = int(max(1, min(20, round(0.10 * float(args_.epochs)))))

    base_lr = float(args_.lr)
    lr_min = max(1e-8, base_lr * 0.01)  

    total_epochs = int(args_.epochs)

    def _warmup_cosine_lr_lambda(epoch_idx: int):
        
        e = int(epoch_idx)

        if warmup_epochs > 0 and e < warmup_epochs:
            # linear warmup: 0 -> 1
            return float(e + 1) / float(warmup_epochs)

        # cosine annealing: from 1 -> lr_min/base_lr
        remain = max(1, total_epochs - warmup_epochs)
        t = float(e - warmup_epochs) / float(remain)  # 0..1
        t = min(max(t, 0.0), 1.0)

        min_mult = float(lr_min) / float(base_lr) if base_lr > 0 else 1.0
        return min_mult + 0.5 * (1.0 - min_mult) * (1.0 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine_lr_lambda)

    optim_sched_info = {
        "optimizer": "AdamW",
        "lr": float(args_.lr),
        "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        "scheduler": "Warmup+Cosine (LambdaLR)",
        "warmup_epochs": int(warmup_epochs),
        "epochs_total": int(total_epochs),
        "lr_min": float(lr_min),
    }
    logger["train"].write({"optim_sched": optim_sched_info})

    if args_.loss == "mse":
        criterion = nn.MSELoss()
    elif args_.loss == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.SmoothL1Loss(beta=float(args_.smoothl1_beta))

    start_epoch = 0
    best_val = float("inf")
    if args_.resume:
        start_epoch = checkpoint.get("epoch", 0)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val = checkpoint.get("best_val", best_val)

    os.makedirs(args_.ckpt_dir, exist_ok=True)

    for epoch0 in range(start_epoch, args_.epochs):
        epoch = epoch0 + 1
        step(model, loader["train"], device, criterion, epoch, logger["train"], args_.log_format,
             optimizer=optimizer, pbar=epoch, true_age=args_.true_age, age_mean=age_mean, age_std=age_std)

        val_mae = step(model, loader["val"], device, criterion, epoch, logger["val"], args_.log_format,
                       optimizer=None, pbar=None, true_age=args_.true_age, age_mean=age_mean, age_std=age_std)

        if scheduler:
            scheduler.step()

        if val_mae < best_val:
            best_val = val_mae
            print("Saving checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_val": best_val,
                    "args": vars(args_),
                },
                os.path.join(args_.ckpt_dir, f"best_model_fold{args_.fold}.pt"),
            )

    test_ckpt = torch.load(
        os.path.join(args_.ckpt_dir, f"best_model_fold{args_.fold}.pt"),
        **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {}),
    )
    model.load_state_dict(test_ckpt["model_state_dict"])
    model.to(device)

    step(model, loader["test"], device, criterion, test_ckpt["epoch"], logger["test"], args_.log_format,
         optimizer=None, pbar=None, true_age=args_.true_age, age_mean=age_mean, age_std=age_std)

    predict_path = os.path.join(args_.log_dir, "predict_age.log")
    write_predict_age_log_subjectwise(
        model=model,
        dataset_train=dataset["train"],
        device=device,
        out_path=predict_path,
        batch_size=args_.batch_size,
        true_age=args_.true_age,
        age_mean=age_mean,
        age_std=age_std,
    )
    print(f"Saved predict_age.log to: {predict_path}")

    test_age_path = os.path.join(args_.log_dir, "test_age.log")
    write_predict_age_log_subjectwise(
        model=model,
        dataset_train=dataset["test"],
        device=device,
        out_path=test_age_path,
        batch_size=args_.batch_size,
        true_age=args_.true_age,
        age_mean=age_mean,
        age_std=age_std,
    )
    print(f"Saved test_age.log to: {test_age_path}")

    eval_train_loader = DataLoader(dataset["train"], batch_size=args_.batch_size, shuffle=False, drop_last=False)
    eval_val_loader = DataLoader(dataset["val"], batch_size=args_.batch_size, shuffle=False, drop_last=False)
    eval_test_loader = DataLoader(dataset["test"], batch_size=args_.batch_size, shuffle=False, drop_last=False)

    r_tr, r2_tr = compute_pearson_r_and_r2(model, eval_train_loader, device, args_.true_age, age_mean, age_std)
    r_va, r2_va = compute_pearson_r_and_r2(model, eval_val_loader, device, args_.true_age, age_mean, age_std)
    r_te, r2_te = compute_pearson_r_and_r2(model, eval_test_loader, device, args_.true_age, age_mean, age_std)

    append_final_r_to_log(os.path.join(args_.log_dir, "train.log"), r_tr, r2_tr)
    append_final_r_to_log(os.path.join(args_.log_dir, "val.log"), r_va, r2_va)
    append_final_r_to_log(os.path.join(args_.log_dir, "test.log"), r_te, r2_te)

    print(f"[Final(best_ckpt)] train  PearsonR={r_tr:.6f}, R2={r2_tr:.6f}")
    print(f"[Final(best_ckpt)] val    PearsonR={r_va:.6f}, R2={r2_va:.6f}")
    print(f"[Final(best_ckpt)] test   PearsonR={r_te:.6f}, R2={r2_te:.6f}")

    # --- Save plots like the paper figure ---
    pred_tr, true_tr = collect_pred_true_year(model, eval_train_loader, device, args_.true_age, age_mean, age_std)
    save_pred_vs_true_hexbin(pred_tr, true_tr, os.path.join(args_.log_dir, "pred_vs_true_train.png"),
                             title="Train: Predicted vs Chronological Age")

    pred_va, true_va = collect_pred_true_year(model, eval_val_loader, device, args_.true_age, age_mean, age_std)
    save_pred_vs_true_hexbin(pred_va, true_va, os.path.join(args_.log_dir, "pred_vs_true_val.png"),
                             title="Val: Predicted vs Chronological Age")

    pred_te, true_te = collect_pred_true_year(model, eval_test_loader, device, args_.true_age, age_mean, age_std)
    save_pred_vs_true_hexbin(pred_te, true_te, os.path.join(args_.log_dir, "pred_vs_true_test.png"),
                             title="Test: Predicted vs Chronological Age")

    
    first_batch = next(iter(eval_test_loader))
    x0 = first_batch[0]  # (x, y_z, subj) 중 x

    bench = benchmark_inference_forward_only(
        model=model,
        sample_x=x0,
        device=device,
        warmup=50,
        iters=200,
    )

    
    logger["test"].write({"inference_benchmark_forward_only": bench})

    
    with open(os.path.join(args_.log_dir, "test.log"), "a") as f:
        f.write(
            "InferenceBench(forward-only)\t"
            f"latency_ms_per_sample\t{bench['latency_ms_per_sample']:.6f}\t"
            f"throughput_sps\t{bench['throughput_samples_per_sec']:.6f}\t"
            f"peak_mem_mb\t{bench['peak_mem_mb']:.3f}\n"
        )


if __name__ == "__main__":
    main(args())
 