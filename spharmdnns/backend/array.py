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
def take_along_dim(input, indices, dim=None):
    return np.take_along_axis(input, indices, axis=dim)


@register("torch")
def take_along_dim(input, indices, dim=None):
    return torch.take_along_dim(input, indices, dim=dim)


@register("numpy")
def vstack(tensors):
    return np.vstack(tensors)


@register("torch")
def vstack(tensors):
    return torch.vstack(tensors)


@register("numpy")
def stack(tensors, dim=0):
    return np.stack(tensors, axis=dim)


@register("torch")
def stack(tensors, dim=0):
    return torch.stack(tensors, dim=dim)


@register("numpy")
def hstack(tensors):
    return np.hstack(tensors)


@register("torch")
def hstack(tensors):
    return torch.hstack(tensors)


@register("numpy")
def unbind(input, dim=0):
    return np.moveaxis(input, dim, 0)


@register("torch")
def unbind(input, dim=0):
    return torch.unbind(input, dim=dim)
