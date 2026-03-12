"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCLoss(nn.Module):
    def __init__(self, reduction="mean"):
        """
        Normalized cross-correlation loss.

        Parameters
        __________
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        super().__init__()

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, input, target):
        loss = 1 - F.cosine_similarity(input - input.mean(-1, keepdim=True), target - target.mean(-1, keepdim=True), dim=-1)
        loss = self.reduction(loss)

        return loss
