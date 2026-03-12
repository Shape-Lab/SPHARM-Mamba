"""
Inference script for age prediction with SPHARMMambaAge.

- 학습 때 사용한 checkpoint(.pt)를 불러와서
- 동일한 icosphere / data-norm / in_ch 설정으로
- y_norm → 실제 나이(age)로 역정규화해서 출력/저장

사용 예시:
    python -m scripts.age.infer_age \
        --data-dir /data/pub/Age_Prediction/baseline/SphericalPred-master/dataset \
        --ckpt /path/to/best_model_fold1.pt \
        --partition test \
        --batch-size 4 \
        --gpu 0
"""

import os
import argparse
import numpy as np
from packaging import version

import torch
from torch.utils.data import DataLoader

from spharmdnns.core.spharm_mamba import SPHARMMambaAge
from spharmdnns.sphere import Icosphere
from spharmdnns.io import read_mesh
from spharmdnns.utils import SphericalDataset, parse_args, save_config


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required
    group = parser.add_argument_group("required arguments")
    group.add_argument("--data-dir", type=str, required=True, help="Path to re-tessellated data (features/labels)")
    group.add_argument("--ckpt", type=str, required=True, help="Trained model checkpoint (.pt)")

    # Optional
    group = parser.add_argument_group("optional arguments")
    group.add_argument("--partition", type=str, choices=["train", "val", "test"], default="test",
                       help="Which split to run inference on")
    group.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    group.add_argument("--gpu", type=int, default=0, help="GPU ID for inference")
    group.add_argument("--no-cuda", action="store_true", help="Force CPU")
    group.add_argument("--threads", type=int, default=1, help="# of CPU threads")
    group.add_argument("--config", type=str, help="Path to the config file")
    group.add_argument("--save-config", type=str, help="Path to save the config file")
    group.add_argument("--out-csv", type=str, default=None,
                       help="Path to save CSV (subject, pred_age, true_age_if_exists)")

    # section_prefix는 그냥 구분용 태그라 이름만 바꿔서 사용
    args = parse_args(parser, section_prefix="age.infer")

    if args.save_config:
        save_config(args, parser, args.save_config,
                    exclude_keys=["config", "save_config"],
                    section_prefix="age.infer")

    return args


def main(args):
    # -----------------------------
    # 1) checkpoint & args 로딩
    # -----------------------------
    ckpt = torch.load(
        args.ckpt,
        map_location=torch.device("cpu"),
        **({"weights_only": False} if version.parse(torch.__version__) >= version.parse("2.6.0") else {}),
    )
    ckpt_args = ckpt["args"]

    # device 설정
    args.no_cuda |= not torch.cuda.is_available()
    device = torch.device("cpu" if args.no_cuda else f"cuda:{args.gpu}")
    torch.set_num_threads(args.threads)

    # -----------------------------
    # 2) age 정규화용 mean / std
    #    (checkpoint → args 순서로 조회)
    # -----------------------------
    age_mean = ckpt.get("age_mean", None)
    age_std = ckpt.get("age_std", None)

    if age_mean is None or age_std is None:
        # 예전 ckpt 또는 저장 안 되어 있을 때 대비
        age_mean = ckpt_args.get("age_mean", 0.0)
        age_std = ckpt_args.get("age_std", 1.0)
        print(f"[WARN] age_mean/std not found as top-level in checkpoint. "
              f"Using ckpt_args: mean={age_mean:.4f}, std={age_std:.4f}")
    else:
        print(f"[INFO] Loaded age_mean/std from checkpoint: mean={age_mean:.4f}, std={age_std:.4f}")

    age_mean = float(age_mean)
    age_std = float(age_std)

    # -----------------------------
    # 3) sphere (icosphere or mesh) 로딩
    #    → train.py와 동일한 설정 사용
    # -----------------------------
    if ckpt_args.get("sphere", None) is not None:
        sphere_path = ckpt_args["sphere"]
        v, f = read_mesh(sphere_path)
    else:
        # Icosphere 레벨은 ckpt_args["ico"] 에 저장되어 있음
        ico_level = ckpt_args.get("ico", 6)
        v, f = Icosphere(ico_level).sphere[-1]

    sphere = (v, f)
    n_vert = v.shape[0]

    # -----------------------------
    # 4) Dataset / DataLoader
    #    → train 때와 동일한 SphericalDataset 사용
    # -----------------------------
    dataset = SphericalDataset(
        data_dir=args.data_dir,
        partition=args.partition,
        fold=ckpt_args["fold"],
        n_vert=n_vert,
        classes=ckpt_args["classes"],
        in_ch=ckpt_args["in_ch"],
        seed=ckpt_args["seed"],
        aug=ckpt_args["aug"],
        n_splits=ckpt_args["n_splits"],
        hemi=ckpt_args["hemi"],
        data_norm=ckpt_args["data_norm"],
        preload=None,           # inference에서는 굳이 preload 안 해도 됨
        task="age",
    )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, drop_last=False)

    print(f"[INFO] Loaded {len(dataset)} subjects for partition='{args.partition}'")

    # -----------------------------
    # 5) 모델 생성 & weight 로드
    #    → train.py에서 썼던 SPHARMMambaAge와 동일한 구조
    # -----------------------------
    print("[INFO] Building SPHARMMambaAge model...")
    model = SPHARMMambaAge(
        device=device,
        sphere=sphere,
        L=ckpt_args["bandwidth"],
        d_model=64,     # train.py에서 사용한 값과 동일하게 유지
        d_state=64,
        d_conv=4,
        expand=2,
        mlp_hidden=128,
        threads=args.threads,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # -----------------------------
    # 6) Inference loop
    # -----------------------------
    all_subj = []
    all_pred_age = []
    all_true_age = []  # label이 있을 경우 역정규화해서 같이 저장

    with torch.no_grad():
        for (data, label_norm, subj) in loader:
            # data: [B, C, n_vert]
            # label_norm: [B, 1] (정규화된 age), subj: list of subj_name
            data = data.to(device, dtype=torch.float32)

            # SPHARMMambaAge forward
            y_norm, _ = model(data)            # y_norm: [B]
            y_norm = y_norm.view(-1).cpu().numpy()

            # 역정규화: age = y_norm * std + mean
            pred_age = y_norm * age_std + age_mean

            # label이 있는 경우 (지금 데이터셋은 있으니)
            label_norm = label_norm.view(-1).cpu().numpy()
            true_age = label_norm * age_std + age_mean

            all_subj.extend(subj)
            all_pred_age.extend(pred_age.tolist())
            all_true_age.extend(true_age.tolist())

    # -----------------------------
    # 7) 결과 출력 / 저장
    # -----------------------------
    print("\n=== Inference results (first 10) ===")
    for i in range(min(10, len(all_subj))):
        print(f"{i:3d} | {all_subj[i]:20s} | "
              f"pred_age = {all_pred_age[i]:6.2f} | "
              f"true_age = {all_true_age[i]:6.2f}")

    if args.out_csv is not None:
        import csv

        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "pred_age", "true_age"])
            for s, p, t in zip(all_subj, all_pred_age, all_true_age):
                writer.writerow([s, f"{p:.4f}", f"{t:.4f}"])

        print(f"\n[INFO] Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main(args())
