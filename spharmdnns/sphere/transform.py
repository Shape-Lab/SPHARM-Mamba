"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn.functional as F

from .interp import retess


def composite(x, y, tree, cache=None):
    """
    Composite warp field. The resulting field is composition of x and y (i.e., y∘x).
    Triangles of x do not need to match those of y.

    Parameters
    __________
    x : torch.tensor, shape = [*, n_vertex_x, 3]
        3D coordinates of the unit sphere.
    y : torch.tensor, shape = [*, n_vertex_y, 3]
        3D warped coordinates of the unit sphere.
    tree : TriangleSearchTorch or its variants
        Triangle search tree for y before warp.
    cache : torch.tensor, shape = [*, n_vertex_x]
        Cached face IDs for fast triangle search on CPU.

    Returns
    _______
    yx : torch.tensor, shape = [*, n_vertex_x, 3]
        3D coordinates of the unit sphere. The output shape is the same as that of x.
    """

    yx = retess(x, y.transpose(-1, -2), tree, cache=cache).transpose(-1, -2)
    yx = F.normalize(yx, dim=-1)

    return yx


def param_to_mat(param):
    """
    6D parameters to rotation matrix.

    Parameters
    __________
    param : torch.tensor, shape = [*, n_vertex, 6]
        3D coordinates of the unit sphere.

    Returns
    _______
    mat : torch.tensor, shape = [*, n_vertex, 3, 3]
        Rotation matrix.

    Notes
    _____
    See [1] for details.
    [1] Zhou, Yi, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li.
        "On the continuity of rotation representations in neural networks."
        In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5745-5753. 2019.
    """

    mat_x = F.normalize(param[..., :3], dim=-1)
    mat_z = F.normalize(mat_x.cross(param[..., 3:], dim=-1), dim=-1)
    mat_y = F.normalize(mat_z.cross(mat_x, dim=-1), dim=-1)

    # mat_x = F.normalize(param[..., :3], dim=-1)
    # y = param[..., 3:] - (mat_x * param[..., 3:]).sum(dim=-1, keepdim=True) * mat_x
    # mat_y = F.normalize(y, dim=-1)
    # mat_z = mat_x.cross(mat_y, dim=-1)

    mat = torch.cat((mat_x, mat_y, mat_z), dim=-1)
    mat = mat.reshape(*param.shape[:-1], 3, 3)

    return mat


def axis_angle_to_mat(axis, angle=None, rodrigues=False):
    """
    Conversion of axis and angle to rotation matrix.

    Parameters
    __________
    axis : torch.tensor, shape = [*, n_vertex, 3]
        Rotation axis. Axis-angle notation is acceptable; axis should be unit otherwise.
    angle : torch.tensor, shape = [*, n_vertex, 1]
        Rotation angle.
    rodrigues : bool
        Computes the rotation matrix using Rodrigues' rotation formula.

    Returns
    _______
    mat : torch.tensor, shape = [*, n_vertex, 3, 3]
        Rotation matrix.
    """

    if rodrigues:
        angle = torch.norm(axis, dim=-1) if angle is None else angle
        axis = F.normalize(axis, dim=-1)
        x, y, z = axis.unbind(-1)
        cos, sin = angle.cos(), angle.sin()
        one_minus_cos = 1 - cos
        xyc, yzc, zxc = x * y * one_minus_cos, y * z * one_minus_cos, z * x * one_minus_cos

        mat = torch.stack(
            (
                cos + x * x * one_minus_cos, xyc - z * sin, zxc + y * sin,
                xyc + z * sin, cos + y * y * one_minus_cos, yzc - x * sin,
                zxc - y * sin, yzc + x * sin, cos + z * z * one_minus_cos,
            ),
            dim=-1,
        ).reshape(*axis.shape, 3)
    else:
        axis = axis if angle is None else axis * angle
        mat = torch.zeros((*axis.shape[:-1], 9), device=axis.device)
        index = torch.zeros_like(axis, dtype=torch.long) + torch.tensor([7, 2, 3], device=axis.device)
        mat.scatter_add_(-1, index, axis)
        index = torch.zeros_like(axis, dtype=torch.long) + torch.tensor([5, 6, 1], device=axis.device)
        mat.scatter_add_(-1, index, -axis)
        mat = mat.reshape(*axis.shape, 3)
        mat = mat.matrix_exp()

    return mat


def mat_to_axis_angle(mat, eps=1e-8):
    """
    Decomposition of rotation matrix to axis and angle.

    Parameters
    __________
    mat : torch.tensor, shape = [*, n_vertex, 3, 3]
        Rotation matrix.
    eps : float
        Small threshold in radians to stabilize angles near 0 or π.

    Returns
    _______
    axis : torch.tensor, shape = [*, n_vertex, 3]
        Rotation axis.
    angle : torch.tensor, shape = [*, n_vertex, 1]
        Rotation angle.
    """

    mat_x, mat_y, mat_z = mat.unbind(-1)
    w = torch.cat(
        (
            mat_y[..., 2:3] - mat_z[..., 1:2],
            mat_z[..., 0:1] - mat_x[..., 2:3],
            mat_x[..., 1:2] - mat_y[..., 0:1],
        ),
        dim=-1,
    )
    trace = mat_x[..., 0:1] + mat_y[..., 1:2] + mat_z[..., 2:3]
    angle = torch.atan2(w.norm(dim=-1, keepdim=True), trace - 1)

    near_zero = angle.abs() < eps
    near_pi = ((angle - torch.pi).abs() < eps).squeeze(-1)

    angle = torch.where(near_zero, torch.zeros(1, device=mat.device), angle)
    axis = torch.where(near_zero, torch.tensor([0.0, 0.0, 1.0], device=mat.device), w)

    id = F.one_hot(mat[near_pi].diagonal(dim1=-2, dim2=-1).argmax(-1), 3)
    axis[near_pi] = (mat[near_pi] * id[..., None]).sum(-2) + id

    near_zero = near_zero.squeeze(-1)
    axis[~near_zero] = F.normalize(axis[~near_zero], dim=-1)

    return axis, angle


def scaling_squaring(mat, v, tree, k=6, eps=5e-4, cache=None):
    """
    Integral over velocity field encoded by rotation matrix.

    Parameters
    __________
    mat : torch.tensor, shape = [*, n_vertex, 3, 3]
        Rotation matrix per vertex.
    v : torch.tensor, shape = [*, n_vertex, 3]
        3D coordinates of the unit sphere.
    tree : TriangleSearchTorch or its variants
        Triangle search tree for v.
    k : int
        Number steps for scaling and squaring.
    eps : float
        Small threshold in radians to stabilize angles near 0 or π.
    cache : torch.tensor, shape = [*, k, n_vertex_x]
        Cached face IDs at each step for fast triangle search on CPU.

    Returns
    _______
    x : torch.tensor, shape = [*, n_vertex, 3]
        3D warped coordinates of the unit sphere.
    """

    if k == 0:
        x = mat_v(mat, v)
    else:
        axis, angle = mat_to_axis_angle(mat, eps=eps)
        x = axis_angle_v(axis, angle / 2**k, v)

        for i in range(k):
            x = composite(x, x, tree, cache=cache[i] if cache is not None else None)

    return x


def mat_v(mat, v):
    """
    Vertex-wise rotation by rotation matrix.

    Parameters
    __________
    mat : torch.tensor, shape = [*, n_vertex, 3, 3]
        Rotation matrix per vertex.
    v : torch.tensor, shape = [*, n_vertex, 3]
        3D coordinates of the unit sphere.

    Returns
    _______
    x : torch.tensor, shape = [*, n_vertex, 3]
        3D warped coordinates of the unit sphere.
    """

    x = (mat @ v[..., None]).squeeze(-1)
    x = F.normalize(x, dim=-1)

    return x


def axis_angle_v(axis, angle, v):
    """
    Vertex-wise rotation by axis and angle.

    Parameters
    __________
    axis : torch.tensor, shape = [*, n_vertex, 3]
        Rotation axis.
    angle : torch.tensor, shape = [*, n_vertex, 1]
        Rotation angle.
    v : torch.tensor, shape = [*, n_vertex, 3]
        3D coordinates of the unit sphere.

    Returns
    _______
    x : torch.tensor, shape = [*, n_vertex, 3]
        3D warped coordinates of the unit sphere.
    """

    dot = (axis * v).sum(-1, keepdim=True)
    cross = axis.cross(v[None], dim=-1)
    cos, sin = angle.cos(), angle.sin()
    x = v * cos + cross * sin + axis * (dot * (1 - cos))
    x = F.normalize(x, dim=-1)

    return x
