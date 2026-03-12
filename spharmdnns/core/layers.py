"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungbo Ha, mj0829@unist.ac.kr
Sunghwa Ryu, sunghwaryu@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import math
import torch
import torch.nn as nn


class SHConv(nn.Module):
    def __init__(self, in_ch, out_ch, L, interval):
        """
        The spectral convolutional filter has L+1 coefficients.
        Among the L+1 points, we set anchor points for every interval of "interval".
        Those anchors are linearly interpolated to fill the blank positions.

        Parameters
        __________
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        L : int
            Bandwidth of input channels. An individual harmonic coefficient is learned in this bandwidth.
        interval : int
            Interval of anchor points. Harmonic coefficients are learned at every "interval".
            The intermediate coefficients between the anchor points are linearly interpolated.

        Notes
        _____
        Input shape  : [batch, in_ch, (L+1)**2]
        Output shape : [batch, out_ch, (L+1)**2]
        """

        super().__init__()

        ncpt = int(math.ceil(L / interval)) + 1
        interval2 = 1 if interval == 1 else L - (ncpt - 2) * interval

        self.weight = nn.Parameter(torch.empty(in_ch, out_ch, ncpt, 1))
        self.register_buffer("l0", torch.arange(0, 1, 1.0 / interval).repeat(1, ncpt - 2).view(ncpt - 2, interval), persistent=False)
        self.register_buffer("l1", torch.arange(0, 1 + 1e-8, 1.0 / interval2).view(1, interval2 + 1), persistent=False)
        self.register_buffer("repeats", torch.tensor([(2 * l + 1) for l in range(L + 1)]), persistent=False)

        stdv = 1.0 / math.sqrt(in_ch * (L + 1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        w1 = (((1 - self.l0) * self.weight[..., :-2, :]) + (self.l0 * self.weight[..., 1:-1, :])).flatten(-2)
        w2 = (((1 - self.l1) * self.weight[..., -2:-1, :]) + (self.l1 * self.weight[..., -1:, :])).flatten(-2)
        w = torch.cat([w1, w2], dim=2).repeat_interleave(self.repeats, dim=2)

        x = (w[None] * x[..., None, :]).sum(1)

        return x


class SHT(nn.Module):
    def __init__(self, L, Y_inv, area):
        """
        Spherical harmonic transform (SHT).
        Spherical signals are transformed into the spectral components.

        Parameters
        __________
        L : int
            Bandwidth of SHT. This should match L in SpectralConv.
        Y_inv : torch.tensor, shape = [n_vertex, (L+1)**2]
            Matrix form of harmonic basis.
        area : torch.tensor, shape = [n_vertex]
            Area per vertex.

        Notes
        _____
        Input shape  : [batch, n_ch, n_vertex]
        Output shape : [batch, n_ch, (L+1)**2]
        """

        super().__init__()

        self.Y_inv = Y_inv[:, : (L + 1) ** 2]
        self.area = area

    def forward(self, x):
        x = (self.area * x) @ self.Y_inv

        return x


class ISHT(nn.Module):
    def __init__(self, Y):
        """
        Inverse spherical harmonic transform (ISHT).
        Spherical signals are reconstructed from the spectral components.

        Parameters
        __________
        Y : torch.tensor, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.

        Notes
        _____
        Input shape  : [batch, n_ch, (L+1)**2]
        Output shape : [batch, n_ch, n_vertex]
        """

        super().__init__()

        self.Y = Y

    def forward(self, x):
        x = x @ self.Y[: x.shape[-1]]

        return x


class HeatKernel(nn.Module):
    def __init__(self, L, ch):
        """
        Heat kernel smoothing.
        Spherical signals are adaptively smoothed by heat kernel filters.

        Parameters
        __________
        L : int
            Bandwidth of SHT. This should match L in SHT.
        ch : int
            Number of input channels.

        Notes
        _____
        Input shape  : [batch, ch, (L+1)**2]
        Output shape : [batch, ch, (L+1)**2]
        """

        super().__init__()

        constant = -torch.arange(0, L + 1) * torch.arange(1, L + 2)
        constant = constant.repeat_interleave(torch.tensor([(2 * l + 1) for l in range(L + 1)]))
        constant = constant[None].repeat(ch, 1)

        self.register_buffer("const", constant * 1e-2)
        self.sigma = nn.Parameter(torch.empty(ch, 1))
        self.sigma.data.uniform_(1e-3, 1e-2)

    def forward(self, x):
        x = (self.const * self.sigma.clamp(0, 1)).exp() * x

        return x
