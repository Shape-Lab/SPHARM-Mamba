"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-12, weight=None, ignore_index=None, reduction="mean"):
        """
        Dice loss.

        Parameters
        __________
        eps : float
            Small value to avoid division by zero.
        weight : torch.tensor, shape=[n_class]
            A manual rescaling weight given to each class.
        ignore_index : list of int, shape=[n_ignore_classes]
            Target values that are ignored and do not contribute to the input gradient.
        reduction : str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """

        super().__init__()

        self.eps = eps
        self.weight = weight if weight is not None else 1
        self.ignore_index = torch.tensor(ignore_index if ignore_index is not None else [], dtype=torch.long)

        if reduction == "mean":
            weight_sum = weight.sum() if weight is not None else 1
            self.reduction = lambda x: x.mean() / weight_sum
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = nn.Identity()
        else:
            raise ValueError(f"reduction must be one of 'none', 'mean', or 'sum'. Got {reduction}.")

    def forward(self, input, target):
        input_soft = F.softmax(input, dim=1)
        target_onehot = torch.zeros_like(input_soft)
        target_onehot.scatter_(-2, target.unsqueeze(-2), 1)

        inter = (input_soft * target_onehot).sum(-1)
        denom = (input_soft + target_onehot).sum(-1).clamp_min(self.eps)
        loss = self.weight * (1 - (2 * inter / denom))

        mask = torch.ones_like(loss, dtype=torch.bool)
        mask[..., self.ignore_index] = False
        loss = loss[mask].reshape(*mask.shape[:-1], -1)
        loss = self.reduction(loss)

        return loss
