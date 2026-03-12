"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from collections import defaultdict

from scipy import stats
from scipy.sparse import csr_array, csgraph


def squeeze_label(data):
    """
    Label mapping to squeeze. Input labels are not necessarily continuous,
    which can consume more space without squeeze.

    Parameters
    __________
    data : 1D int array
        Input labels.

    Returns
    _______
    lut_old2new : dict (int -> int)
        Mapping to squeezed indices.
    lut_new2old : dict (int -> int)
        Mapping to original indices.
    """

    lut_old2new = defaultdict(lambda: 0)
    lut_new2old = dict()
    for i, label in enumerate(data):
        lut_old2new[label] = i
        lut_new2old[i] = label

    return lut_old2new, lut_new2old


def normalize_data(data):
    """
    Data normalization.

    Parameters
    __________
    data : array, shape = [*, n_vert]
        Input data.

    Returns
    _______
    data : array, shape = [*, n_vert]
        Normalized data.
    """

    data = stats.zscore(data, axis=-1)
    data[data < -3] = -3 - (1 - np.exp(3 + data[data < -3]))
    data[data > 3] = 3 + (1 - np.exp(3 - data[data > 3]))
    data /= np.std(data, axis=-1, keepdims=True)
    data[data < -3] = -3 - (1 - np.exp(3 + data[data < -3]))
    data[data > 3] = 3 + (1 - np.exp(3 - data[data > 3]))

    return data


def refine_label(data, f):
    """
    Label refinement.
    This function assumes that each ROI has a single connected component.
    Do NOT use this function if any ROI has more than one connected component.

    Parameters
    __________
    data : 2D array, shape = [n_label, n_vert]
        Model inference.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.

    Returns
    _______
    data : 2D array, shape = [n_label, n_vert]
        Refined model inference.
    """

    n_label, n_vert = data.shape
    v1_ = f[:, [0, 1, 2]].ravel()
    v2_ = f[:, [1, 2, 0]].ravel()
    n_comp = 0
    label = np.argmax(data, 0)
    label_old = np.zeros(data.shape[-1])

    while n_comp != n_label and (label_old != label).any():
        label_old = label

        idx = label[v1_] == label[v2_]
        v1 = v1_[idx]
        v2 = v2_[idx]

        m = csr_array((np.ones(v1.shape[0]), (v1, v2)), shape=(n_vert, n_vert))
        n_comp, comp = csgraph.connected_components(m, directed=False, return_labels=True)

        comp_size = [len(label[comp == i]) for i in range(n_comp)]
        comp_ordered = np.argsort(comp_size)[::-1]

        for i in range(n_label, n_comp):
            idx = comp == comp_ordered[i]
            data[label[idx], idx] = np.finfo(data.dtype).min

        label = np.argmax(data, 0)

    return data
