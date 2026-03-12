"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seunghwan Lee, shwan@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

from ..backend import ops


def eval_dice(input, target, n_class=None, area=None, eps=1e-12):
    """
    Dice socre.

    Parameters
    __________
    input : array, shape = [*, n_vertex]
        Model inference.
    target : array, shape = [*, n_vertex]
        Target labels.
    n_class : int
        Number of classes.
    area : array, shape = [*, n_vertex]
        Vertex-wise area.
    eps : float
        Small value to avoid division by zero.

    Returns
    _______
    dice : array, shape = [*, n_class]
        Batch-wise Dice score.
    """

    n_class = target.max() + 1 if n_class is None else n_class
    area = area[None] if area is not None and len(input.shape) == 2 else area
    area = 1 if area is None else area

    input_onehot = ops.one_hot(input, num_classes=n_class).swapaxes(-1, -2)
    target_onehot = ops.one_hot(target, num_classes=n_class).swapaxes(-1, -2)

    input_area = input_onehot * area
    target_area = target_onehot * area

    inter = (input_area * target_onehot).sum(-1)
    denom = input_area.sum(-1) + target_area.sum(-1)

    return 2 * inter / ops.clamp_min(denom, eps)


def eval_accuracy(input, target):
    """
    Accuracy.

    Parameters
    __________
    input : array, shape = [*, n_vertex]
        Model inference.
    target : array, shape = [*, n_vertex]
        True labels.

    Returns
    _______
    n_correct : numeric or array, shape = [*]
        Number of correct vertices.
    n_vert : numeric or array, shape = [*]
        Number of vertices.
    """

    n_correct = (input == target).sum(-1)
    n_vert = ops.full_like(n_correct, target.shape[-1])

    return n_correct, n_vert


def eval_distortion(orig, warp, face, eps=1e-12):
    """
    Jacobian distortion metrics.

    Parameters
    __________
    orig : array, shape = [*, n_vertex, 3]
        Original coordinates.
    warp : array, shape = [*, n_vertex, 3]
        Warped coordinates.
    face : array, shape = [*, n_face, 3]
        Vertex IDs of faces common to orig and warp.
    eps : float
        Small value to avoid division by zero.

    Returns
    _______
    area : array, shape = [*, n_face]
        Area distortion.
    shape : array, shape = [*, n_face]
        Shape distortion.
    """

    f0, f1, f2 = ops.unbind(face[..., None], -2)
    v0o, v0w = ops.take_along_dim(orig, f0, -2), ops.take_along_dim(warp, f0, -2)

    e1o = ops.take_along_dim(orig, f1, -2) - v0o
    e2o = ops.take_along_dim(orig, f2, -2) - v0o
    e1w = ops.take_along_dim(warp, f1, -2) - v0w
    e2w = ops.take_along_dim(warp, f2, -2) - v0w

    bo = ops.cross(ops.cross(e1o, e2o, dim=-1), e1o, dim=-1)
    bo = ops.normalize(bo, dim=-1, eps=eps)
    bw = ops.cross(ops.cross(e1w, e2w, dim=-1), e1w, dim=-1)
    bw = ops.normalize(bw, dim=-1, eps=eps)

    x00 = ops.norm(e1w, dim=-1) / ops.clamp_min(ops.norm(e1o, dim=-1), eps)
    x11 = (e2w * bw).sum(-1) / ops.clamp_min((e2o * bo).sum(-1), eps)

    area, tr = x00 * x11, x00 + x11
    disc = ops.sqrt(ops.clamp_min(tr**2 - 4 * area, 0))
    shape = (tr + disc) / ops.clamp_min((tr - disc), eps)

    return area, shape
