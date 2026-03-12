"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn


class AreaLoss(nn.Module):
    def __init__(self, v, f, log_scale=False, reduction="mean"):
        """
        Area ratio loss.

        Parameters
        __________
        v : torch.tensor, shape = [n_vertex, 3]
            3D coordinates of the unit sphere.
        f : torch.tensor, shape = [n_face, 3]
            Triangles of the unit sphere.
        log_scale : bool
            Uses 1/x for x < 1, else x if False; otherwise |log₂(x)|.
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        super().__init__()

        self.f = f
        a, b, c = v[self.f].unbind(1)
        self.base = (a - b).cross(a - c, dim=-1).norm(dim=-1)
        self.base = self.base[None]
        self.scale = lambda x: x.log2().abs() if log_scale else torch.where(x < 1, 1 / x, x)

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, x):
        a, b, c = x[..., self.f, :].unbind(-2)
        area = (a - b).cross(a - c, dim=-1).norm(dim=-1)
        loss = self.scale((area / self.base).clamp_min(1e-7))
        loss = self.reduction(loss)

        return loss
