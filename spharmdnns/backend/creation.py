"""
July 2025

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
import torch
import torch.nn.functional as F

from ._registry import register


@register("numpy")
def full_like(input, fill_value):
    return np.full_like(input, fill_value)


@register("torch")
def full_like(input, fill_value):
    return torch.full_like(input, fill_value)


@register("numpy")
def one_hot(tensor, num_classes=-1):
    return np.eye(num_classes)[tensor]


@register("torch")
def one_hot(tensor, num_classes=-1):
    return F.one_hot(tensor, num_classes=num_classes)
