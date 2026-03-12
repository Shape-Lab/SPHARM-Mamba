"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree

import torch
import torch.nn.functional as F

from .geometry import face_area, face_normal
from .interp import barycentric
from .topology import n_ring_adj


class TriangleSearch:
    def __init__(self, v, f, eps=1e-8, ring=2):
        """
        Fast closest triangle search.
        The module only works on spherical tessellation.

        Parameters
        __________
        v : 2D array, shape = [n_vertex, 3]
            3D coordinates of the unit sphere.
        f : 2D array, shape = [n_face, 3]
            Triangles of the unit sphere.
        eps : float
            The area is empty if its size is less than eps.
        ring : int
            Search range bounded by ring size.
        """

        self._v, self._f = v, f

        centroid = self._v[self._f].sum(1)
        centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)
        self._MD = KDTree(centroid)

        self._normal = face_normal(self._v, self._f, eps=eps, fix_orientation=True)
        self._inner = (self._v[self._f[:, 0]] * self._normal).sum(1, keepdims=True)
        self._area = face_area(self._v, self._f)[:, None] * 2
        self._area[self._area < eps] = 1
        self._nn = n_ring_adj(self._f, ring).sum(1).max()

    @property
    def v(self):
        return self._v

    @property
    def f(self):
        return self._f

    def _valid_query(self, q, fid, tol):
        """
        Check if q are inside triangles.
        """

        # inner = (q * self._normal[k]).sum(1, keepdims=True)
        inner = np.einsum("ij,ij->i", q, self._normal[fid])[:, None]
        valid = inner[:, 0] > 0
        inner[~valid] = 1
        q_proj = q * (self._inner[fid] / inner)
        b = barycentric(q_proj, self._v[self._f[fid]], self._normal[fid], self._area[fid])
        valid &= (b >= -tol).all(axis=1)

        return valid, fid[valid], b[valid]

    def query(self, query, tol=1e-5, cache=None):
        """
        Find triangles that contain query points and then compute their barycentric coefficients.
        """

        dim = np.shape(query)
        query = query.reshape(-1, 3)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)

        qID = np.arange(query.shape[0])
        fid = np.zeros(query.shape[0], dtype=int)
        bary = np.zeros((query.shape[0], 3), dtype=float)

        for i in range(self._nn + 1):
            q = query[qID]

            if i == 0:
                if cache is None:
                    continue
                k = cache.ravel()
                d = np.zeros_like(k)
                assert k.shape[0] == query.shape[0]
            else:
                d, k = self._MD.query(q, [i])
                k, d = k[:, 0], d[:, 0]

            valid, fid[qID[valid]], bary[qID[valid]] = self._valid_query(q, k, tol)
            qID, d = qID[~valid], d[~valid]
            if qID.size == 0:
                break

        if qID.size != 0:
            d = d.max()
            for i in range(1, 6):
                k = self._MD.query_ball_point(query[qID], d * 2**i)
                qID_ball = np.repeat(qID, np.fromiter((len(i) for i in k), dtype=int))

                valid, fid[qID_ball[valid]], bary[qID_ball[valid]] = self._valid_query(query[qID_ball], np.concatenate(k), tol)
                qID = qID[~np.isin(qID, qID_ball[valid])]
                if qID.size == 0:
                    break

        if qID.size != 0:
            print(f"No triangle at {qID}. Increase tol or ring size to allow a wider search range.")

        return fid.reshape(dim[:-1]), bary.reshape(dim)


class TriangleSearchTorch:
    def __init__(self, v, f, norm=False, eps=1e-8):
        """
        GPU-accelerated fast closest triangle search.
        This module only works on spherical tessellation.
        If a GPU device is unavailable, TriangleSearch will be used instead.

        Parameters
        __________
        v : torch.tensor, shape = [*, n_vertex, 3]
            3D coordinates of the unit sphere.
        f : torch.tensor, shape = [n_face, 3]
            Triangles of the unit sphere.
        norm : bool
            Enforces unit sphere.
        eps : float
            The area is empty if its size is less than eps.
        """

        v = v[None] if v.dim() == 2 else v
        f = f[None] if f.dim() == 2 else f
        self._batch = v.shape[0]
        if self._batch > 1:
            f = f + torch.arange(self._batch, device=f.device)[:, None, None] * v.shape[-2]
        if norm:
            v = F.normalize(v, dim=-1)

        self._v, self._f = v.reshape(-1, 3), f.reshape(-1, 3)

        self._normal = face_normal(self._v, self._f, eps=eps, fix_orientation=True)
        self._inner = (self._v[self._f[:, 0]] * self._normal).sum(1, keepdim=True)
        self._area = face_area(self._v, self._f)[:, None] * 2
        self._area[self._area < eps] = 1
        self._ncand = f.shape[-2]

        if v.device.type != "cpu":
            from ..extensions import TriangleSearchCUDA

            self._tree = TriangleSearchCUDA
            self._offset = torch.Tensor()
            self._cand = torch.Tensor() if self._batch == 1 else torch.arange(self._f.shape[-2], device=f.device)
        else:
            self._tree = [TriangleSearch(v[i].detach().numpy(), f[0].detach().numpy(), eps=eps) for i in range(self._batch)]

    @property
    def v(self):
        return self._v

    @property
    def f(self):
        return self._f[: self._ncand]

    def query(self, query, tol=1e-4, norm=False, bary=True, cand=None, ncand=None, offset=None, cache=None):
        """
        Find triangles that contain query points and then compute their barycentric coefficients.
        The cache is automatically updated after each operation when the CPU version is used.
        """

        if norm:
            query = F.normalize(query, dim=-1)

        if query.dim() == 3 and self._batch != query.shape[0]:
            if query.shape[0] == 1:
                query = query.repeat_interleave(self._batch, dim=0)
            elif self._batch != 1:
                raise ValueError(f"Batch size mismatch: query has {query.shape[0]} samples, but only {self._batch} trees are available.")

        dim = query.shape
        device = query.device

        if device.type != "cpu":
            q = query.reshape(-1, 3).contiguous()
            if offset is None:
                offset = self._offset if query.dim() == 2 else torch.arange(self._batch).repeat_interleave(query.shape[-2])
                offset = offset.to(device)
            if cand is None:
                cand = self._cand.to(device)
            if ncand is None:
                ncand = self._ncand
            fid = self._tree.query(self._v, self._f, cand, ncand, offset, q, self._normal, self._inner, self._area, tol)
            if self._batch > 1:
                fid -= offset * ncand
        else:
            q = query.reshape(self._batch, -1, 3)
            cache = cache.reshape(self._batch, -1) if isinstance(cache, torch.Tensor) else [None] * self._batch
            fid = [
                torch.tensor(
                    self._tree[i].query(q[i].detach().numpy(), tol=tol, cache=cache[i].detach().numpy() if cache[i] is not None else None)[0],
                    dtype=torch.long,
                    device=device,
                )
                for i in range(self._batch)
            ]
            fid = torch.cat(fid, dim=0)
            if isinstance(cache, torch.Tensor):
                cache.reshape(dim[:-1]).copy_(fid.reshape(dim[:-1]))

        if bary:
            q = query.reshape(-1, 3)
            q = q * (self._inner[fid] / (q * self._normal[fid]).sum(1, keepdim=True))
            b = barycentric(q, self._v[self._f[fid]], self._normal[fid], self._area[fid])

            return fid.reshape(dim[:-1]), b.reshape(dim)
        else:
            return fid.reshape(dim[:-1])


class TriangleSearchIcoTorch:
    def __init__(self, v, f, norm=False):
        """
        GPU-accelerated fast closest triangle search within an icosphere.
        The module works only on hierarchical tessellation that generates equal # of triangles per subdivision.
        Do NOT use this module without a GPU device. The performance will not be guaranteed, otherwise.

        Parameters
        __________
        v : a list of torch.tensor, shape = [[n_vertex_ico0, 3], ...]
            3D coordinates of the unit sphere.
        f : a list of torch.tensor, shape = [[n_face_ico0, 3], ...]
            Triangles of the unit sphere.
        norm : bool
            Enforces unit sphere.
        """

        self.level = len(v)
        self._cand = [torch.Tensor()]
        self._ncand = [f[0].shape[0]]
        self._tree = [TriangleSearchTorch(v[0], f[0], norm)]
        self._v, self._f = v[-1], f[-1]

        for i in range(self.level - 1):
            q = v[i + 1][f[i + 1]].sum(-2)
            q = F.normalize(q, dim=-1)
            fid = self._tree[i].query(q, bary=False)
            _, idx = torch.sort(fid)
            _, counts = torch.unique(fid, return_inverse=False, return_counts=True)
            assert counts.min() == counts.max() == 4

            self._cand.append(idx)
            self._ncand.append(int(counts.min()))
            self._tree.append(TriangleSearchTorch(v[i + 1], f[i + 1], norm))

    @property
    def v(self):
        return self._v

    @property
    def f(self):
        return self._f

    def query(self, query, tol=1e-4, norm=False, cache=None):
        device = query.device
        q = query.contiguous()
        if norm:
            q = F.normalize(q, dim=-1)

        offset = torch.Tensor().to(device)
        for i in range(self.level - 1):
            fid = self._tree[i].query(q, cand=self._cand[i].to(device), ncand=self._ncand[i], offset=offset, tol=tol, norm=False, bary=False)
            offset = fid.clone()

        return self._tree[-1].query(q, cand=self._cand[-1].to(device), ncand=self._ncand[-1], offset=offset, tol=tol, norm=False, cache=cache)
