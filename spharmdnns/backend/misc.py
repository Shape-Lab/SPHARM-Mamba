"""
July 2025

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
import torch

from ._registry import register


@register("numpy")
def sqrt(input):
    return np.sqrt(input)


@register("torch")
def sqrt(input):
    return torch.sqrt(input)


@register("numpy")
def clamp_min(input, min=None):
    return np.clip(input, min, None)


@register("torch")
def clamp_min(input, min=None):
    return torch.clamp_min(input, min)
