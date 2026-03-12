"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .icosphere import Icosphere
from .trisearch import TriangleSearchTorch
from .transform import axis_angle_to_mat, mat_v, param_to_mat
from .interp import retess
from ..utils import normalize_data
from ..loss import CCLoss


def rigid_alignment(x, y, f, input, target, lr=1e-2, epoch=100, tol=1e-4, data_norm=True, search_intv=0, search_ico=1, search_topk=4, device="cpu"):
    """
    Estimate rigid alignment between two spherical maps using optimization.

    Parameters
    ----------
    x : 2D array, shape = [n_vertex_x, 3]
        3D coordinates of the unit sphere (moving).
    y : 2D array, shape = [n_vertex_y, 3]
        3D coordinates of the unit sphere (target).
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere (target).
    input : 1D array, shape = [n_vertex_x]
        Moving signal defined on x.
    target : 1D array, shape = [n_vertex_y]
        Target signal defined on y.
    lr : float
        Learning rate for optimizer.
    epoch : int
        Number of optimization iterations.
    tol : float
        Early stopping threshold on update magnitude.
    data_norm : bool
        Whether to normalize input and target signals before alignment.
    search_intv : int
        Rotation interval (π/interval) of global search. If 0, disabled.
    search_ico : int
        Icospheral rotation axes of global search.
    search_topk : int
        Number of optimizable candidates of global search.
    device : str
        Device indicator.

    Returns
    -------
    rot : 2D array, shape = [3, 3]
        Estimated rotation matrix.
    """

    x = torch.from_numpy(x).to(device, dtype=torch.float)
    y = torch.from_numpy(y).to(device, dtype=torch.float)
    f = torch.from_numpy(f).to(device, dtype=torch.long)

    normalize = normalize_data if data_norm else lambda x: x
    input = torch.from_numpy(normalize(input)).to(device, dtype=torch.float)
    target = torch.from_numpy(normalize(target)).to(device, dtype=torch.float)
    target = target.reshape(1, target.numel())

    tree = TriangleSearchTorch(y, f)
    criterion = CCLoss(reduction="none")

    axis = Icosphere(search_ico).sphere[-1][0]
    deg = (torch.arange(1, search_intv + 1, device=device) * np.pi / search_intv)[:, None, None]
    axis = torch.from_numpy(axis)[None].to(device, dtype=torch.float)
    mat = torch.cat((torch.eye(3, device=device)[None], axis_angle_to_mat((axis * deg).reshape(-1, 3))))[:, None]
    loss = criterion(input, retess(mat_v(mat, x), target, tree))
    mat = mat[torch.topk(loss.view(-1), min(search_topk, loss.numel()), largest=False, sorted=False).indices]

    rot, best = mat.clone(), torch.full((mat.shape[0], 1), torch.finfo(loss.dtype).max, device=device)
    cache = torch.zeros((mat.shape[0], x.shape[0]), dtype=torch.long, device=device) if device == "cpu" else None
    param = nn.Parameter(mat[..., :2, :].flatten(-2))
    optimizer = optim.SGD([param], lr=lr)

    for _ in range(epoch):
        mat = param_to_mat(param)
        loss = criterion(input, retess(mat_v(mat, x), target, tree, cache=cache))

        idx = best - loss > tol
        rot[idx], best[idx] = mat[idx], loss[idx]

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

    return rot[best.argmin()].squeeze().cpu().detach().numpy()
