"""
August 2022

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungeun Lee, selee@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn

from .spharmnet import SPHARMNet
from .layers import SHT, ISHT
from ..sphere import TriangleSearchTorch, TriangleSearchIcoTorch, Icosphere
from ..sphere import vertex_area, spharm_real, retess, composite, axis_angle_to_mat, param_to_mat, mat_v, scaling_squaring


class SHRotationBlock(nn.Module):
    def __init__(self, sphere, area, in_ch, tree=None):
        """
        SHRotationBlock [1].
        Geometric features are rigidly adjusted for better feature extraction.

        [1] Lee, Seungeun, Seunghwan Lee, Sunghwa Ryu, and Ilwoo Lyu.
            "SPHARM-Reg: Unsupervised Cortical Surface Registration using Spherical Harmonics."
            IEEE Transactions on Medical Imaging.

        Parameters
        __________
        sphere : a list of tuples of (v, f)
            A list of icosphere tuples.
            The input spheres should be hierarchical as recursive subdivision if len(list) > 1.
            The hierarchy will considerably accelerate triangle query.
        area : torch.tensor, shape = [n_vertex]
            Area per vertex.
        in_ch : int
            Number input geometric features.
        tree : TriangleSearchTorch or its variants
            Triangle search tree for icosphere.

        Notes
        _____
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, in_ch, n_vertex]
        """

        super().__init__()

        self.v, _ = sphere[-1]

        rot_param = 6
        self.param_to_mat = param_to_mat

        # rigid rotation
        self.w0 = nn.Parameter(area.reshape(-1, 1) / area.sum())

        self.linear = nn.Conv1d(in_ch, rot_param, kernel_size=1, stride=1, bias=True)
        self.linear.weight.data.zero_()

        # a little perturbation
        rot = axis_angle_to_mat(torch.tensor([1, 0, 0]) * 0.05)
        self.linear.bias.data = rot[:2].ravel()

        self.tree = TriangleSearchIcoTorch(*zip(*sphere)) if tree is None else tree
        self.cache = None

    def set_cache(self, cache):
        """
        Face ID cache for CPU-based triangle search acceleration. This does not affect GPU operations.

        Parameters
        __________
        cache : torch.tensor
            Cached face IDs for fast triangle search on CPU.
            The cache shape should match the output structure for tree.query.
            The cache is automatically updated after each tree.query operation.

        Notes
        _____
        Input shape : [batch, n_vertex]
        """

        self.cache = cache

    def forward(self, input):
        # geometry to 6 rot params
        x = (input @ self.w0.abs()) * 1e-3
        x = self.linear(x)
        x = x.transpose(-1, -2)

        # 6 rot params to mat
        x = self.param_to_mat(x)

        # mat to rotated verts
        x = mat_v(x, self.v)

        # retess
        x = retess(x, input, self.tree, cache=self.cache)
        self.cache = None

        return x


class SHWarpBlock(nn.Module):
    def __init__(self, sphere, area, Y, DL=40, k=6, tree=None):
        """
        SHWarpBlock [1].
        The raw 6D parameters undergo constraint by SHT to enforce rigid and non-rigid warps simultaneously [2].
        These parameters are then refined and converted into a spatial-varying rotation, i.e., velocity field.
        The smoothness of the estimated field is guaranteed due to the use of SHT,
        which is defined by trigonometric functions and offers L-inf continuity.
        Finally, the field is traced via numerical integration in a scaling and squaring manner.

        [1] Lee, Seungeun, Seunghwan Lee, Sunghwa Ryu, and Ilwoo Lyu.
            "SPHARM-Reg: Unsupervised Cortical Surface Registration using Spherical Harmonics."
            IEEE Transactions on Medical Imaging.

        [2] Lyu, Ilwoo, Hakmook Kang, Neil D. Woodward, Martin A. Styner, and Bennett A. Landman.
            "Hierarchical spherical deformation for cortical surface registration."
            Medical image analysis 57 (2019): 72-88.

        Parameters
        __________
        sphere : a list of tuples of (v, f)
            A list of icosphere tuples.
            The input spheres should be hierarchical as recursive subdivision if len(list) > 1.
            The hierarchy will considerably accelerate triangle query.
        area : torch.tensor, shape = [n_vertex]
            Area per vertex.
        Y : 2D tensor, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.
        DL : int
            Spectral bandwidth for velocity field decomposition (see the paper for details).
        k : int
            Steps for numerical integration powered by scaling and squaring (see the paper for details).
        tree : TriangleSearchTorch or its variants
            Triangle search tree for icosphere.

        Notes
        _____
        In channel shape  : [batch, 6, n_vertex] for moving subject
        Out channel shape : [batch, n_vertex, 3] for warped spherical coordinates
        """

        super().__init__()

        self.v, _ = sphere[-1]
        self.k = k

        rot_param = 6
        self.param_to_mat = param_to_mat

        self.SHT = SHT(DL, Y.T, area)
        self.ISHT = ISHT(Y)

        self.w = nn.Parameter(torch.zeros((rot_param, (DL + 1) ** 2)))
        self.b = nn.Parameter(torch.empty((rot_param, (DL + 1) ** 2)))
        self.b.data.uniform_(-5e-3, 5e-3)
        self.b.data[0, 0] = self.b.data[4, 0] = (4 * torch.pi) ** 0.5

        self.tree = TriangleSearchIcoTorch(*zip(*sphere)) if tree is None else tree
        self.cache = None

    def set_cache(self, cache):
        """
        Face ID cache for CPU-based triangle search acceleration. This does not affect GPU operations.

        Parameters
        __________
        cache : torch.tensor
            Cached face IDs for fast triangle search on CPU.
            The cache shape should match the output structure for tree.query.
            The cache is automatically updated after each tree.query operation.

        Notes
        _____
        Input shape : [k, batch, n_vertex]
        """

        self.cache = cache

    def forward(self, x):
        # 6 params to adjusted sph coeffs
        x = self.SHT(x)
        x = x * self.w + self.b

        # sph coeffs to 6 rot params
        x = self.ISHT(x)
        x = x.transpose(-1, -2)

        # 6 rot params to mat
        x = self.param_to_mat(x)

        # mat to warped verts
        x = scaling_squaring(x, self.v, self.tree, k=self.k, cache=self.cache)
        self.cache = None

        return x


class SPHARMReg(nn.Module):
    def __init__(self, device, ico, composite=3, in_ch=2, DL=40, k=6, C=16, L=80, D=3, interval=5, threads=1, verbose=False):
        """
        SPHARM-Reg [1].
        The SHRotationBlock [1] computes the optimal rotation to the target, which allows for better feature extraction.
        The backbone architecture of the system is SPHARM-Net, which estimates 6D rotation parameters from spherical data.
        These raw parameters are then fed to the SHWarpBlock [1].
        SPHARM-Reg computes a composite warp field by inferring warp fields and applying them to the previous field.
        The final composite field warps the original sphere, which enables the warped features to closely match the target as possible.

        [1] Lee, Seungeun, Seunghwan Lee, Sunghwa Ryu, and Ilwoo Lyu.
            "SPHARM-Reg: Unsupervised Cortical Surface Registration using Spherical Harmonics."
            IEEE Transactions on Medical Imaging.

        Parameters
        __________
        device : torch.device
            Device indicator.
        ico : int
            Level of icosahedral subdivision.
        composite : int
            Number warp field compositions (see the paper for details).
        in_ch : int
            Number input geometric features.
        DL : int
            Spectral bandwidth for velocity field decomposition (see the paper for details).
        k : int
            Steps for numerical integration powered by scaling and squaring (see the paper for details).
        C : int
            Number channels in the entry layer (see the paper for details).
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        D : int
            Depth of encoding/decoding levels (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        threads : int
            Number CPU threads for basis reconstruction. Useful if the unit sphere has dense tesselation.

        Notes
        _____
        In channel shape  : [batch, in_ch // 2, n_vertex] for moving subject, [batch, in_ch // 2, n_vertex] for target
        Out channel shape : [batch, n_vertex, 3] for warped spherical coordinates, [batch, in_ch // 2, n_vertex] for corresponding target
        """

        super().__init__()

        icosphere = Icosphere(ico).sphere

        v, f = icosphere[-1]
        v = v.astype(float)

        area = torch.tensor(vertex_area(v, f), dtype=torch.float32, device=device)
        Y = torch.tensor(spharm_real(v, max(L, DL), threads=threads), dtype=torch.float32, device=device)

        self.composite = composite

        if device.type == "cpu":
            icosphere = [icosphere[-1]]
        sphere = [(torch.from_numpy(v).to(device, dtype=torch.float32), torch.from_numpy(f).to(device, dtype=torch.long)) for v, f in icosphere]
        self.v, self.f = sphere[-1]
        self.tree = TriangleSearchIcoTorch(*zip(*sphere))
        self.cache_composite = [None] * self.composite
        self.cache_retess = [None] * (self.composite + 1)

        rot_param = 6
        self.rblock = []
        self.velocity = []
        self.wblock = []
        for _ in range(self.composite + 1):
            self.rblock.append(SHRotationBlock(sphere, area, in_ch // 2, self.tree))
            self.velocity.append(
                nn.Sequential(
                    SPHARMNet(device, None, area, Y, in_ch, C, C, L, D, interval, threads, verbose),
                    nn.LeakyReLU(negative_slope=0.0001),
                    nn.Conv1d(C, rot_param, kernel_size=1, stride=1, bias=True),
                )
            )
            self.wblock.append(SHWarpBlock(sphere, area, Y, DL, k, self.tree))
        self.rblock = nn.ModuleList(self.rblock)
        self.velocity = nn.ModuleList(self.velocity)
        self.wblock = nn.ModuleList(self.wblock)

    def set_cache(self, cache):
        """
        Face ID cache for CPU-based triangle search acceleration. This does not affect GPU operations.

        Parameters
        __________
        cache : dict
            Cached face IDs for fast triangle search on CPU for retesselation, composite, and rotation & warp blocks.
            The cache is automatically updated after each tree.query operation.
            The cache shape should match the output structure for tree.query.
            The cache should be initialized before every forward operation, as it is set to None afterward to prevent unintended references to the cache.
            Set each element of the cache to None if it is not intended for use.

        Notes
        _____
        cache["composite"] : a list of torch.tensor, shape = [composite, [batch, n_vertex]]
        cache["retess"]    : a list of torch.tensor, shape = [composite + 1, [batch, n_vertex]]
        cache["rblock"]    : a list of torch.tensor, shape = [composite + 1, [batch, n_vertex]]
        cache["wblock"]    : a list of torch.tensor, shape = [composite + 1, [k, batch, n_vertex]]
        """

        if isinstance(cache, dict):
            for i in range(self.composite):
                self.cache_composite[i] = cache["composite"][i]
            for i in range(self.composite + 1):
                self.cache_retess[i] = cache["retess"][i]
                self.rblock[i].set_cache(cache["rblock"][i])
                self.wblock[i].set_cache(cache["wblock"][i])

    def forward(self, input, target):
        x = self.rblock[0](input)
        x = self.velocity[0](torch.cat([x, target], dim=1))
        x = self.wblock[0](x)

        for i in range(self.composite):
            tree = TriangleSearchTorch(x, self.f)
            x_ = retess(self.v[None], input, tree, cache=self.cache_retess[i])
            self.cache_retess[i] = None
            x_ = self.rblock[i + 1](x_)
            x_ = self.velocity[i + 1](torch.cat([x_, target], dim=1))
            x_ = self.wblock[i + 1](x_)
            x = composite(x, x_, self.tree, cache=self.cache_composite[i])
            self.cache_composite[i] = None

        target = retess(x, target, self.tree, cache=self.cache_retess[self.composite])
        self.cache_retess[self.composite] = None

        return x, target
