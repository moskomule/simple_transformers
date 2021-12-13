from __future__ import annotations

import math

import torch
from homura import Registry
from torch import nn


def _ensure_tuple(x, size) -> tuple:
    if isinstance(x, (tuple, list)):
        assert len(x) == size
        return tuple(x)
    return tuple((x for x in range(size)))


class PatchEmbed2d(nn.Module):
    def __init__(self,
                 patch_size: int or tuple,
                 emb_dim: int,
                 in_channels: int
                 ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        # input: BxCxHxW -> BxNxC'
        return self.proj(input).flatten(2).transpose(1, 2)


class _PosEmbed2dBase(nn.Module):
    def __init__(self,
                 grid_size: int | tuple[int, int],
                 use_cls_token: bool):
        super().__init__()
        self.grid_size = grid_size
        self.use_cls_token = use_cls_token
        self.num_patches = math.prod(_ensure_tuple(grid_size, 2))
        self.pos_emb = None

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        return self.pos_emb + input


POSEMB2D = Registry('pos_emb2d')


@POSEMB2D.register(name='learnable')
class LearnablePosEmbed2d(_PosEmbed2dBase):
    def __init__(self,
                 emb_dim: int,
                 grid_size: int | tuple[int, int],
                 use_cls_token: bool):
        super().__init__(grid_size, use_cls_token)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + int(use_cls_token), emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)


@POSEMB2D.register(name='sinusoidal')
class SinusoidalPosEmbed2d(_PosEmbed2dBase):
    # from facebookresearch/mocov3
    def __init__(self,
                 emb_dim: int,
                 grid_size: int | tuple[int, int],
                 use_cls_token: bool,
                 temperature: float = 10_000.0
                 ):
        super().__init__(grid_size, use_cls_token)
        h, w = _ensure_tuple(grid_size, 2)
        grid_w, grid_h = torch.meshgrid(torch.arange(w, dtype=torch.float), torch.arange(h, dtype=torch.float),
                                        indexing='ij')
        pos_dim = emb_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float) / pos_dim
        omega = 1 / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
        if use_cls_token:
            pe_token = torch.zeros([1, 1, emb_dim], dtype=torch.float)
            pos_emb = torch.cat([pe_token, pos_emb])
        self.register_buffer("pos_emb", pos_emb)
