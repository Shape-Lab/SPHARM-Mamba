"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from scipy.sparse import csr_array

from .topology import edge_list
from ..backend import ops


def edge_length(v, f, norm=False):
    """
    Edge-wise length.

    Parameters
    __________
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.
    norm : bool
        Enforces unit sphere.

    Returns
    _______
    length : 1D array, shape = [n_edge]
        Length per edge.
    """

    if norm:
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    idx = edge_list(f)

    return np.linalg.norm(v[idx[0]] - v[idx[1]], axis=-1)


def vertex_area(v, f, norm=False):
    """
    Vertex-wise area. The area is approximated by a third of neighborhood triangle areas.

    Parameters
    __________
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.
    norm : bool
        Enforces unit sphere.

    Returns
    _______
    area : 1D array, shape = [n_vertex]
        Area per vertex.
    """

    area = csr_array((face_area(v, f, norm=norm).repeat(3), (f.ravel(), np.arange(f.shape[0]).repeat(3))), shape=(v.shape[0], f.shape[0]))

    return area.sum(1) / 3


def face_area(v, f, norm=False):
    """
    Face-wise area.

    Parameters
    __________
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.
    norm : bool
        Enforces unit sphere.

    Returns
    _______
    area : 1D array, shape = [n_face]
        Area per face.
    """

    if norm:
        v = ops.normalize(v, dim=-1)
    a, b, c = ops.unbind(v[f], 1)

    return ops.norm(ops.cross(b - a, c - b, dim=-1), dim=-1) / 2


def face_normal(v, f, norm=False, eps=1e-12, fix_orientation=False):
    """
    Face-wise normal.

    Parameters
    __________
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.
    norm : bool
        Enforces unit sphere.
    eps : float
        Triangles smaller than eps use the centroid orientation.
    fix_orientation : bool
        Corrects normal direction for small or misoriented triangles.

    Returns
    _______
    normal : 2D array, shape = [n_face, 3]
        Normal per face.
    """

    if norm:
        v = ops.norm(v, dim=-1, keepdim=True)

    a, b, c = ops.unbind(v[f], 1)
    n = ops.cross(b - a, c - b, dim=-1)

    if fix_orientation:
        n[ops.inner(a, n, keepdim=False) < 0] *= -1

    length = ops.norm(n, dim=-1, keepdim=True)
    idx = (length > eps).ravel()
    n = n / ops.clamp_min(length, eps)
    n[~idx] = ops.normalize(v[f[~idx]].sum(-1), dim=-1)

    return n
