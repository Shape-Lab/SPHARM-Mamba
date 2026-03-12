"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn


class ArcLoss(nn.Module):
    def __init__(self, v, f, reduction="mean"):
        """
        Arc length loss.

        Parameters
        __________
        v : torch.tensor, shape = [n_vertex, 3]
            3D coordinates of the unit sphere.
        f : torch.tensor, shape = [n_face, 3]
            Triangles of the unit sphere.
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        super().__init__()

        self.idx = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        self.idx = torch.sort(self.idx, dim=1).values
        self.idx = torch.unique(self.idx, dim=0).T

        self.base = ((v[self.idx[0]] * v[self.idx[1]]).sum(-1)).arccos()
        self.base = self.base[None]

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, x):
        arc = (x[..., self.idx[0], :] * x[..., self.idx[1], :]).sum(-1)
        arc = arc.clamp(-1 + 1e-7, 1 - 1e-7).arccos()
        loss = (arc - self.base) ** 2
        loss = self.reduction(loss)

        return loss
