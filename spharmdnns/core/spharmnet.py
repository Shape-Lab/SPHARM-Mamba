"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungbo Ha, mj0829@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SHConv, SHT, ISHT
from ..sphere import vertex_area, spharm_real
from ..io import read_mesh


class SHConvBlock(nn.Module):
    def __init__(self, Y, area, in_ch, out_ch, L, interval, nonlinear=None, fullband=True, bn=True):
        """
        SHConvBlock [1].
        The SHConvBlock applies SHT to a spherical signal, followed by rotation-equivariant spectral convolution and ISHT.
        If fullband is set to True, the module behaves like SHConv-FB, enabling constrained full-bandwidth convolution.
        Otherwise, the module only allows spectral convolution, which assumes spectral pooling.

        [1] Ha, Seungbo, and Ilwoo Lyu.
            "SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation."
            IEEE Transactions on Medical Imaging 41, no. 10 (2022): 2739-2751.

        Parameters
        __________
        Y : torch.tensor, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.
        area : torch.tensor, shape = [n_vertex]
            Area per vertex.
        in_ch : int
            Number input channels.
        out_ch : int
            Number output channels.
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        nonlinear : None or torch.nn.functional
            Non-linear activation. If not set, nn.Identity will be used.
        fullband : bool
            Full-bandwidth convolution.
        bn : bool
            Batch normalization before non-linear activation.

        Notes
        _____
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, out_ch, n_vertex]
        """

        super().__init__()

        self.shconv = nn.Sequential(SHT(L, Y.T, area), SHConv(in_ch, out_ch, L, interval), ISHT(Y))
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=not bn) if fullband else lambda _: 0
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False) if bn else nn.Identity()
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

    def forward(self, x):
        x = self.shconv(x) + self.impulse(x)
        x = self.bn(x)
        x = self.nonlinear(x)

        return x


class SPHARMNet(nn.Module):
    def __init__(self, device, sphere, area=None, Y=None, in_ch=3, n_class=32, C=128, L=80, D=3, interval=5, threads=1, verbose=False):
        """
        SPHARM-Net [1].
        During the encoding phase, each block halves the harmonic bandwidth L while simultaneously doubling the number of channels C.
        Conversely, the decoding phase doubles the harmonic bandwidth L and halves the number of channels C.
        The final block aggregates the learned information to infer labels.
        The SHConv block applies SHT to a spherical signal, followed by rotation-equivariant spectral convolution and ISHT.
        On the other hand, the SHConv-FB block enhances the SHConv block by adding a scaled spherical signal,
        enabling constrained full-bandwidth convolution.

        [1] Ha, Seungbo, and Ilwoo Lyu.
            "SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation."
            IEEE Transactions on Medical Imaging 41, no. 10 (2022): 2739-2751.

        Parameters
        __________
        device : torch.device
            Device indicator.
        sphere : str or tuple of (v, f)
            Sphere mesh file. VTK or FreeSurfer format. This argument will be ignored if both area and Y are provided.
        area : torch.tensor, shape = [n_vertex]
            Area per vertex.
        Y : torch.tensor, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.
        in_ch : int
            Number input geometric features.
        n_class : int
            Number labels, i.e., the output layer size.
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
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, n_class, n_vertex]
        """

        super().__init__()

        L_in = L
        self.down = []
        self.up = []

        ch_inc = 2
        out_ch = C

        if area is None or Y is None:
            v, f = read_mesh(sphere) if isinstance(sphere, str) else sphere
            v = v.astype(float)
            if area is None:
                area = torch.tensor(vertex_area(v, f, norm=True), dtype=torch.float32, device=device)
            if Y is None:
                Y = torch.tensor(spharm_real(v, L, threads=threads), dtype=torch.float32, device=device)

        nonlinear = F.relu

        # encoding
        for i in range(D):
            L = L_in // 2**i
            if verbose:
                print(f"Down {i + 1}\t| C:{in_ch} -> {out_ch}\t| L:{L}")
            self.down.append(SHConvBlock(Y, area, in_ch, out_ch, L, interval, nonlinear, i == 0))
            in_ch = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        in_ch = out_ch
        if verbose:
            print(f"Bottom\t| C:{in_ch} -> {out_ch}\t| L:{L}")
        self.down.append(SHConvBlock(Y, area, in_ch, out_ch, L, interval, nonlinear, False))

        # decoding
        for i in range(D - 1):
            L = L_in // 2 ** (D - 1 - i)
            in_ch = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print(f"Up {i + 1}\t| C:{in_ch} -> {out_ch}\t| L:{L}")
            self.up.append(SHConvBlock(Y, area, in_ch, out_ch, L, interval, nonlinear, False))

        in_ch = out_ch * 2
        L *= 2
        if verbose:
            print(f"Up {D}\t| C:{in_ch} -> {out_ch}\t| L:{L}")
        self.up.append(SHConvBlock(Y, area, in_ch, out_ch, L, interval, nonlinear, True))
        if verbose:
            print(f"Final\t| C:{out_ch} -> {n_class}\t| L:{L}")
        self.final = SHConvBlock(Y, area, out_ch, n_class, L_in, interval, None, True)

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.down[0](x)]
        for i in range(len(self.down) - 1):
            x_.append(self.down[i + 1](x_[-1]))
        x = x_[-1]
        for i in range(len(self.up)):
            x = torch.cat([x, x_[-2 - i]], dim=1)
            x = self.up[i](x)
        x = self.final(x)

        return x
