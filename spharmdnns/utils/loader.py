"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import os
import numpy as np
import random
import re
import itertools

import torch
from torch.utils.data import Dataset

from .data import normalize_data, squeeze_label
from ..io import read_dat


class SphericalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        partition,
        fold,
        n_vert,
        classes,
        in_ch,
        seed,
        aug,
        n_splits,
        hemi,
        data_norm=True,
        preload=None,
        task="seg",          # "seg" / "reg" / "age"
    ):
        """
        Loader for SPHARM-Net. This module subdivides the input dataset for cross-validation.

        Parameters
        __________
        ...
        task : str, optional
            "seg"  : segmentation (기존과 동일)
            "reg"  : registration (기존 classes = template feature)
            "age"  : age regression (labels/*.float.dat 또는 *.label.dat, scalar age)
        """

        assert partition in ["train", "test", "val"]
        self._n_vert = n_vert
        self._data_norm = data_norm
        self._preload = preload is not None
        self._task = task

        feat_dir = os.path.join(data_dir, "features")
        feat_files = os.listdir(feat_dir)
        feat_files = [f for f in feat_files if f.split(".")[1] in hemi]
        feat_files = [f for ch in in_ch for f in feat_files if ".".join(f.split(".")[3:-1]) == ch]

        # ------------------------------
        # task에 따라 label 모드 결정
        # ------------------------------
        if task == "age":
            self._seg = False
            self._age = True
        else:
            self._age = False
            self._seg = isinstance(classes, list)  # 기존 로직 유지 (seg vs reg)

        if self._seg or self._age:
            label_dir = os.path.join(data_dir, "labels")
            label_files = os.listdir(label_dir)
            label_files = [f for f in label_files if f.split(".")[1] in hemi]

            if self._age:
                # age 모드에서 scalar age가 들어있는 파일만 사용
                #  - subj.lh.aug0.label.dat
                #  - subj.lh.aug0.float.dat 도 허용
                label_files = [
                    f for f in label_files
                    if f.split(".")[3] in ("label", "float")
                ]

                # age 정규화를 위한 전체 mean / std 계산
                ages_all = []
                for f in label_files:
                    f_path = os.path.join(label_dir, f)
                    age_arr = read_dat(f_path, 1)  # 길이 1 배열이라고 가정

                    if isinstance(age_arr, np.ndarray):
                        age_val = float(age_arr[0])
                    else:
                        age_val = float(age_arr)

                    ages_all.append(age_val)

                if len(ages_all) == 0:
                    # 방어 코드
                    self._age_mean = 0.0
                    self._age_std = 1.0
                else:
                    self._age_mean = float(np.mean(ages_all))
                    # std가 0이 되는 극단 상황 방지용 epsilon
                    self._age_std = float(np.std(ages_all) + 1e-6)

        # registration 모드 (self._seg == False, self._age == False) 에서는
        # label_files를 사용하지 않고 classes 배열을 바로 사용 (기존 코드)
        # ------------------------------

        feat_dict = dict()
        for f in feat_files:
            temp = f.split(".")[0:2]  # ['subj_name', 'lh']
            subj = ".".join(temp)     # 'subj_name.lh'
            if subj not in feat_dict:
                feat_dict[subj] = dict()
            key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])  # aug0, aug1, ...
            f_path = os.path.join(feat_dir, f)
            feat_dict[subj].setdefault(key, []).append(f_path)

        if self._seg or self._age:
            label_dict = dict()
            for f in label_files:
                temp = f.split(".")[0:2]  # ['subj_name', 'lh']
                subj = ".".join(temp)     # 'subj_name.lh'
                if subj not in label_dict:
                    label_dict[subj] = dict()
                key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])
                f_path = os.path.join(label_dir, f)
                label_dict[subj][key] = label_dict[subj].setdefault(key, f_path)

        subj_list = feat_dict.keys()
        subj_list = sorted(subj_list)

        random.seed(seed)
        random.shuffle(subj_list)

        train_subj, val_subj, test_subj = self._kfold(subj_list, n_splits, fold)

        # final list
        self._feat_list = []
        self._subj_list = []
        self._label_list = [] if (self._seg or self._age) else np.array([])

        if partition == "train":
            for subj in train_subj:
                for i in range(0, aug + 1):
                    self._feat_list.append(feat_dict[subj]["aug" + str(i)])
                    self._subj_list.append(subj)
                    if self._seg or self._age:
                        self._label_list.append(label_dict[subj]["aug" + str(i)])

        if partition == "val":
            for subj in val_subj:
                self._feat_list.append(feat_dict[subj]["aug0"])
                self._subj_list.append(subj)
                if self._seg or self._age:
                    self._label_list.append(label_dict[subj]["aug0"])

        if partition == "test":
            for subj in test_subj:
                self._feat_list.append(feat_dict[subj]["aug0"])
                self._subj_list.append(subj)
                if self._seg or self._age:
                    self._label_list.append(label_dict[subj]["aug0"])

        # label dictionary / target 준비
        if self._seg:
            # segmentation: LUT로 클래스 압축 (기존과 동일)
            self._lut, _ = squeeze_label(classes)
        elif self._age:
            # age regression: label은 파일에서 scalar로 읽고
            # __getitem__에서 정규화하여 리턴
            pass
        else:
            # registration: classes = [n_target, n_vertex] 템플릿 (기존과 동일)
            self._label_list = classes[:, : self._n_vert]
            if self._data_norm:
                self._label_list = normalize_data(self._label_list)

        if self._preload:
            self._data = []
            self._label = []
            for i in range(len(self._feat_list)):
                data, label, _ = self._read_data(i)
                data = torch.tensor(data, device=preload)
                label = torch.tensor(label, device=preload)
                self._data += [data]
                self._label += [label]

    @property
    def subj_list(self):
        return self._subj_list

    # 🔴 (추가) age mean / std를 외부에서 읽을 수 있도록 property 제공
    @property
    def age_mean(self):
        return getattr(self, "_age_mean", None)

    @property
    def age_std(self):
        return getattr(self, "_age_std", None)

    def __len__(self):
        return len(self._feat_list)

    def __getitem__(self, idx):
        if self._preload:
            return self._data[idx], self._label[idx], self._subj_list[idx]
        else:
            return self._read_data(idx)

    def _read_data(self, idx):
        # load feature files
        data = np.array([])
        for f in self._feat_list[idx]:
            temp = read_dat(f, self._n_vert)
            data = np.append(data, temp)

        data = np.reshape(data, (-1, self._n_vert)).astype(np.float32)

        # x data normalization (기존 로직 유지)
        if self._data_norm:
            data = normalize_data(data)

        if self._seg:
            # segmentation: vertex-wise label (기존 코드)
            label = read_dat(self._label_list[idx], self._n_vert)
            label = label.astype(int)
            label = [self._lut[l] for l in label]
            label = np.asarray(label)
            return data, label, self._subj_list[idx]

        elif self._age:
            # age regression: scalar age → 정규화 후 리턴
            age_arr = read_dat(self._label_list[idx], 1)   # length=1 array
            if isinstance(age_arr, np.ndarray):
                age = float(age_arr[0])
            else:
                age = float(age_arr)

            age_norm = (age - self._age_mean) / self._age_std
            age_norm = np.asarray([age_norm], dtype=np.float32)  # shape (1,)

            return data, age_norm, self._subj_list[idx]

        else:
            # registration: 공통 target 템플릿 사용 (기존 코드)
            return data, self._label_list, self._subj_list[idx]

    def _kfold(self, subj, n_splits=5, fold=1):
        total_subj = len(subj)
        fold_size = total_subj // n_splits
        fold_residual = total_subj - fold_size * n_splits

        fold_size = [fold_size + 1 if i < fold_residual else fold_size for i in range(n_splits)]
        fold_idx = [0] + list(itertools.accumulate(fold_size))

        id_base = n_splits
        id_val = (id_base + fold - 1) % n_splits
        id_test = (id_base + fold) % n_splits

        val = subj[fold_idx[id_val] : fold_idx[id_val] + fold_size[id_val]]
        test = subj[fold_idx[id_test] : fold_idx[id_test] + fold_size[id_test]]
        if id_val > id_test:
            train = subj[fold_idx[id_test] + fold_size[id_test] : fold_idx[id_val]]
        else:
            train = subj[0 : fold_idx[id_val]] + subj[fold_idx[id_test] + fold_size[id_test] : None]

        return train, val, test
