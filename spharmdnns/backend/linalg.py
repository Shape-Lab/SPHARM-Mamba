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
def cross(input, other, dim=None):
    return np.cross(input, other, axis=dim)


@register("torch")
def cross(input, other, dim=None):
    return torch.cross(input, other, dim=dim)


@register("numpy")
def inner(input, other, keepdim=True):
    result = np.einsum("ij,ij->i", input, other)
    return result[:, None] if keepdim else result


@register("torch")
def inner(input, other, keepdim=True):
    return (input * other).sum(-1, keepdim=keepdim)


@register("numpy")
def norm(input, dim=None, keepdim=False):
    return np.linalg.norm(input, axis=dim, keepdims=keepdim)


@register("torch")
def norm(input, dim=None, keepdim=False):
    return torch.norm(input, dim=dim, keepdim=keepdim)


@register("numpy")
def normalize(input, dim=1, eps=1e-12):
    return input / np.clip(np.linalg.norm(input, axis=dim, keepdims=True), eps, None)


@register("torch")
def normalize(input, dim=1, eps=1e-12):
    return F.normalize(input, dim=dim, eps=eps)
