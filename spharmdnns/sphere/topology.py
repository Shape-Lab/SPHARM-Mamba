"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from scipy.sparse import csr_array


def edge_list(f):
    """
    Edge list sorted by triangle ID and edge order.

    Parameters
    __________
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.

    Returns
    _______
    idx : 2D array, shape = [2, n_edge]
        Pairs of vertex IDs for each edge.
    """

    nvert = f.max() + 1
    idx = f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).T

    sel = idx[0] < idx[1]
    order0 = np.flatnonzero(sel)
    order1 = np.flatnonzero(~sel)
    order0 = csr_array((order0, idx[:, order0]), shape=(nvert, nvert))[(*idx[:, order1][[1, 0]],)]
    sel = np.zeros(idx.shape[1], dtype=bool)
    sel[np.where(order0 < order1, order0, order1)] = True
    idx = idx[:, sel]

    return idx


def n_ring_adj(f, ring):
    """
    Adjacency matrix for neighbors within <= n rings.

    Parameters
    __________
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.
    ring : int
        Ring size.

    Returns
    _______
    adj : sparse matrix, shape = [n_vertex, n_vertex]
        Symmetric adjacency matrix of n-ring neighbors.
    """

    nvert = f.max() + 1
    rows = f[:, [0, 1, 2, 1, 2, 0]].ravel()
    cols = f[:, [1, 2, 0, 0, 1, 2]].ravel()
    adj1 = csr_array((np.ones(len(rows)), (rows, cols)), shape=(nvert, nvert), dtype=int)
    adj = power = adj1

    for _ in range(1, ring):
        power = power @ adj1
        adj = adj.maximum(power)
    adj.data[:] = 1

    return adj
