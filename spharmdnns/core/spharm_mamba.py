# spharm_mamba.py
"""
SPHARM + Mamba age regression model (AUX-FREE).

- SHConvBlock: SHT -> SHConv -> ISHT (+ impulse skip) + BN + nonlinearity
- SPHARMMambaAge:
    x -> stage1 -> stage2 -> SHT(L_low) -> tokens (M=(L_low+1)^2) -> Mamba -> mean pool -> head
Forward returns ONLY pred (no aux).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SHConv, SHT, ISHT
from spharmdnns.sphere import vertex_area, spharm_real
from spharmdnns.io import read_mesh

from mamba_ssm.modules.mamba2_simple import Mamba2Simple


class SHConvBlock(nn.Module):
    def __init__(
        self,
        Y: torch.Tensor,
        area: torch.Tensor,
        in_ch: int,
        out_ch: int,
        L: int,
        interval: int,
        nonlinear=None,
        fullband: bool = True,
        bn: bool = True,
    ):
        super().__init__()

        self.shconv = nn.Sequential(
            SHT(L, Y.T, area),
            SHConv(in_ch, out_ch, L, interval),
            ISHT(Y),
        )

        self.impulse = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=not bn)
            if fullband
            else (lambda _: 0)
        )

        #groupnorm이 아니라 batchnorm을 사용하고 싶으면 아래 주석 활성화 하고 아래아래 코드를 주석처리
        self.bn = (
            nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False)
            if bn
            else nn.Identity()
        )
        # self.bn = (
        #     nn.GroupNorm(
        #         num_groups=(
        #             16 if (out_ch % 16 == 0)
        #             else 8 if (out_ch % 8 == 0)
        #             else 4 if (out_ch % 4 == 0)
        #             else 2 if (out_ch % 2 == 0)
        #             else 1
        #         ),
        #         num_channels=out_ch,
        #         affine=True,
        #     )
        #     if bn
        #     else nn.Identity()
        # )


        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shconv(x) + self.impulse(x)
        x = self.bn(x)
        x = self.nonlinear(x)
        return x


def _get_nonlinear(nonlinearity: Optional[str]):
    if nonlinearity is None:
        return None
    name = nonlinearity.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unknown nonlinearity: {nonlinearity}")


def _prepare_sphere(
    sphere,
    device: torch.device,
    L_high: int,
    threads: int,
    area: Optional[torch.Tensor] = None,
    Y: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (area, Y) where
      area: [N]
      Y:    [(L_high+1)^2, N]
    """
    if area is not None and Y is not None:
        return area, Y

    v, f = read_mesh(sphere) if isinstance(sphere, str) else sphere
    v = v.astype(float)

    if area is None:
        area = torch.tensor(vertex_area(v, f, norm=True), dtype=torch.float32, device=device)
    if Y is None:
        Y = torch.tensor(spharm_real(v, L_high, threads=threads), dtype=torch.float32, device=device)

    return area, Y


class SPHARMMambaAge(nn.Module):
    """
    AUX-FREE version.

    Pipeline:
      x [B,C,N]
        -> stage1 -> [B,128,N]
        -> stage2 -> [B,256,N]
        -> SHT(L_low) -> coeff_low [B,256,M], M=(L_low+1)^2
        -> tokens_scalar = coeff_low^T -> [B,M,256]
        -> token_embed -> Mamba
        -> mean pool over M -> [B,d_model]
        -> MLP -> pred [B,1]
    """

    def __init__(
        self,
        device,
        sphere,
        in_ch: int = 4,
        area: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,

        # train.py compatibility
        L: Optional[int] = None,
        interval: Optional[int] = None,

        # explicit high/low
        L_high: int = 80,
        L_low: int = 40,
        interval_high: int = 5,
        interval_low: int = 5,

        # mamba cfg
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,

        # head
        mlp_hidden: int = 128,

        # misc
        threads: int = 0,
        bn: bool = True,
        use_impulse: bool = True,
        nonlinearity: str = "relu",
        token_norm: str = "none",  # "none" | "layernorm"

        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.in_ch = int(in_ch)


        if interval is not None:
            interval_high = int(interval)
            interval_low = int(interval)

        self.L_high = int(L_high)
        self.L_low = int(L_low)
        self.num_coeff_low = (self.L_low + 1) ** 2
        self.num_tokens_low = self.L_low + 1

        area, Y = _prepare_sphere(
            sphere=sphere,
            device=device,
            L_high=self.L_high,
            threads=int(threads),
            area=area,
            Y=Y,
        )
        self.register_buffer("area", area, persistent=False)
        self.register_buffer("Y", Y, persistent=False)

        nonlinear_fn = _get_nonlinear(nonlinearity)

        self.stage1 = SHConvBlock(
            Y=Y,
            area=area,
            in_ch=self.in_ch,
            out_ch=128,
            L=self.L_high,
            interval=int(interval_high),
            nonlinear=nonlinear_fn,
            fullband=bool(use_impulse),
            bn=bool(bn),
        )
        self.stage2 = SHConvBlock(
            Y=Y,
            area=area,
            in_ch=128,
            out_ch=256,
            L=self.L_low,
            interval=int(interval_low),
            nonlinear=nonlinear_fn,
            fullband=bool(use_impulse),
            bn=bool(bn),
        )

        self.sht_low = SHT(self.L_low, Y.T, area)

        self.token_embed = nn.Linear(256, d_model, device=device)

        if token_norm.lower() == "layernorm":
            self.token_norm = nn.LayerNorm(d_model, device=device)
        elif token_norm.lower() == "none":
            self.token_norm = nn.Identity()
        else:
            raise ValueError("token_norm must be 'none' or 'layernorm'")

        self.mamba_layers = nn.ModuleList(
            [
                Mamba2Simple(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=d_model,
                    ngroups=1,
                    use_mem_eff_path=True,
                    device=device,
                )
                for _ in range(int(n_layers))
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(d_model, mlp_hidden, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.0),  
            nn.Linear(mlp_hidden, 1, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        if C != self.in_ch:
            raise ValueError(f"expects in_ch={self.in_ch}, got {C}")

        x128 = self.stage1(x)        # [B,128,N]
        x256 = self.stage2(x128)     # [B,256,N]

        coeff_low = self.sht_low(x256)  # [B,256,M]
        M = coeff_low.shape[-1]
        if M != self.num_coeff_low:
            raise RuntimeError(f"coeff_low has M={M}, expected {self.num_coeff_low}")

        
        coeff_sq = coeff_low * coeff_low  # [B,256,M]
        power_list = []
        for l in range(self.num_tokens_low):
            start = l * l
            end = (l + 1) * (l + 1)
            power_list.append(coeff_sq[:, :, start:end].mean(dim=-1))  # [B,256]
        power_low = torch.stack(power_list, dim=-1)  # [B,256,L_low+1]

        tokens_scalar = power_low.permute(0, 2, 1).contiguous()  # [B,L_low+1,256]
        tokens_flip = torch.flip(tokens_scalar, dims=[1])      #flip
        tokens_scalar = torch.cat([tokens_scalar, tokens_flip], dim=1)  # [B,2*(L_low+1),256]
        tokens = self.token_embed(tokens_scalar)                 # [B,L_low+1,d_model]
                                                                    ## [B,2*(L_low+1),d_model]
        tokens = self.token_norm(tokens)

        h = tokens
        for layer in self.mamba_layers:
            h = layer(h)                                         # [B,L_low+1,d_model]

        feat = h.mean(dim=1)                                     # [B,d_model]
        pred = self.readout(feat)                                # [B,1]
        return pred


__all__ = [
    "SHConvBlock",
    "SPHARMMambaAge",
]
