"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seunghwan Lee, shwan@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from scipy.sparse import csr_array

from .topology import edge_list


class Icosphere:
    def __init__(self, level):
        """
        Vectorized icospheres derived from an icosahedron.
        The module returns a list of vertices and faces.

        Parameters
        __________
        level : int
            The level of recursion to build an icosphere.
        """

        self.sphere = [self._icosahedron()]
        for i in range(level):
            self.sphere.append(self._subdivision(*self.sphere[i]))

    def _icosahedron(self):
        z_ring = 1 / np.sqrt(5)
        r_ring = 2 * z_ring
        intv = np.pi / 5
        angle = np.arange(5) * 2 * intv

        upper = np.column_stack((r_ring * np.cos(angle), r_ring * np.sin(angle), np.full(5, z_ring)))
        angle = np.roll(angle, 1) + intv
        lower = np.column_stack((r_ring * np.cos(angle), r_ring * np.sin(angle), np.full(5, -z_ring)))

        v = np.vstack([[0.0, 0.0, 1.0], upper, lower, [0.0, 0.0, -1.0]])
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        f = np.array([
            [0, 3, 4], [0, 4, 5], [0, 5, 1], [0, 1, 2], [0, 2, 3],
            [3, 2, 8], [3, 8, 9], [3, 9, 4], [4, 9, 10], [4, 10, 5],
            [5, 10, 6], [5, 6, 1], [1, 6, 7], [1, 7, 2], [2, 7, 8],
            [8, 11, 9], [9, 11, 10], [10, 11, 6], [6, 11, 7], [7, 11, 8],
        ], dtype=int)

        return v, f

    def _subdivision(self, v, f):
        # edges
        idx = edge_list(f)

        # midpoints
        vNew = v[idx[0]] + v[idx[1]]
        vNew /= np.linalg.norm(vNew, axis=1, keepdims=True)
        vNew = np.vstack((v, vNew))

        # lookup table
        idx = csr_array(
            (
                np.tile(np.arange(v.shape[0], idx.shape[1] + v.shape[0]), 2),
                (idx.ravel(), idx[[1, 0]].ravel()),
            ),
            shape=(v.shape[0], v.shape[0]),
        )

        # midpoint indices
        vID1 = idx[f[:, 0], f[:, 1]]
        vID2 = idx[f[:, 1], f[:, 2]]
        vID3 = idx[f[:, 2], f[:, 0]]

        # new triangles
        fNew = np.empty((3, f.shape[0] * 4), dtype=int)
        fNew[:, 0::4] = [f[:, 0], vID1, vID3]
        fNew[:, 1::4] = [vID1, f[:, 1], vID2]
        fNew[:, 2::4] = [vID3, vID2, f[:, 2]]
        fNew[:, 3::4] = [vID1, vID2, vID3]
        fNew = np.ascontiguousarray(fNew.T)

        return vNew, fNew
