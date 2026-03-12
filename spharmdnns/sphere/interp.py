"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

from ..backend import ops


def barycentric(q, p, normal, area):
    """
    Query-wise barycentric coefficients.

    Parameters
    __________
    q : 2D array, shape = [n_vertex, 3]
        Query coordinates.
    p : 3D array, shape = [n_vertex, 3, 3]
        3D coordinates of triangles.
    normal : 2D array, shape = [n_vertex, 3]
        Face normal.
    area : 2D array, shape = [n_vertex, 1]
        Face area.

    Returns
    _______
    bary : 2D array, shape = [n_vertex, 3]
        Barycentric coefficients per query.
    """

    qa, qb, qc = ops.unbind(p - q[:, None], 1)
    u = ops.inner(ops.cross(qb, qc, dim=-1), normal) / area
    v = ops.inner(ops.cross(qc, qa, dim=-1), normal) / area
    w = 1 - u - v

    return ops.hstack((u, v, w))


def retess(q, feat, tree, nearest=False, cache=None):
    """
    Data retessellation. The resulting geometry is sampeld at q.

    Parameters
    __________
    q : array, shape = [*, n_vertex_q, 3]
        3D coordinates of the unit sphere.
    feat : array, shape = [*, in_ch, n_vertex_feat]
        Geometric features.
    tree : TriangleSearch or its variants
        Triangle search tree for feat.
    nearest : bool
        Enables nearest neighbor interpolation.
    cache : array, shape = [*, n_vertex_q]
        Cached face IDs for fast triangle search on CPU.

    Returns
    _______
    tess : array, shape = [*, in_ch, n_vertex_q]
        Retessellated geometric features.
    """

    if len(feat.shape) <= 2 or feat.shape[0] == 1:
        single_channel = len(feat.shape) == 1
        if single_channel == 1:
            feat = feat[None]
        fid, bary = tree.query(q, cache=cache)
        feat = feat.swapaxes(-1, -2).reshape(-1, feat.shape[-2])
        tess = feat[tree.f[fid, bary.argmax(-1)]] if nearest else (feat[tree.f[fid]] * bary[..., None]).sum(-2)

        return tess.squeeze(-1) if single_channel else tess.swapaxes(-1, -2)

    if len(q.shape) == 2:
        q = q[None]

    fid, bary = tree.query(q, cache=cache)

    return (
        ops.take_along_dim(feat, tree.f[fid, bary.argmax(-1)][:, None], -1)
        if nearest
        else (
            ops.stack(
                (
                    ops.take_along_dim(feat, tree.f[fid, 0][:, None], -1),
                    ops.take_along_dim(feat, tree.f[fid, 1][:, None], -1),
                    ops.take_along_dim(feat, tree.f[fid, 2][:, None], -1),
                ),
                dim=-1,
            )
            * bary[:, None]
        ).sum(-1)
    )
