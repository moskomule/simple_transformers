from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from typing import Type

import torch
from homura import Registry
from homura.modules import EMA
from torch import nn

from .attentions import SelfAttention
from .base import TransformerBase
from .blocks import TimmPreLNBlock

ViTs = Registry("vit", nn.Module)


class PatchEmbed(nn.Module):
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


class ViT(TransformerBase):
    """ Vision Transformer
    """

    def __init__(self,
                 attention: SelfAttention,
                 num_classes: int,
                 image_size: int or tuple,
                 patch_size: int or tuple,
                 emb_dim: int,
                 num_layers: int,
                 emb_dropout_rate: float,
                 dropout_rate: float,
                 droppath_rate: float,
                 in_channels: int,
                 norm: Type[nn.LayerNorm] = nn.LayerNorm,
                 mlp_widen_factor: int = 4,
                 activation: str = "gelu",
                 enable_checkpointing=False
                 ):
        blocks = [TimmPreLNBlock(emb_dim, deepcopy(attention), dropout_rate=dropout_rate, droppath_rate=r,
                                 widen_factor=mlp_widen_factor, norm=norm, activation=activation, )
                  for r in [x.item() for x in torch.linspace(0, droppath_rate, num_layers)]
                  ]
        super().__init__(nn.Sequential(*blocks), enable_checkpointing)
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = math.prod(image_size) // math.prod(patch_size)

        self.patch_emb = PatchEmbed(patch_size, emb_dim, in_channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.dropout = nn.Dropout(emb_dropout_rate)
        self.norm = norm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.init_weights()

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.patch_emb(input)  # BxNxC
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # Bx(N+1)xC
        x = self.dropout(self.pos_emb + x)
        x = self.norm(self.blocks(x))
        return self.fc(x[:, 0])

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.trunc_normal_(self.patch_emb.proj.weight)
        nn.init.zeros_(self.patch_emb.proj.bias)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @classmethod
    def construct(cls,
                  emb_dim: int,
                  num_layers: int,
                  num_heads: int,
                  patch_size: int,
                  emb_dropout_rate: float = 0,
                  attn_dropout_rate: float = 0,
                  proj_dropout_rate: float = 0,
                  dropout_rate: float = 0,
                  droppath_rate: float = 0,
                  num_classes: int = 1_000,
                  image_size: int = 224,
                  in_channels: int = 3,
                  layernorm_eps: float = 1e-6,
                  activation: str = "gelu",
                  **kwargs
                  ) -> ViT:
        attention = SelfAttention(emb_dim, num_heads, attn_dropout_rate, proj_dropout_rate,
                                  qkv_bias=False)
        return cls(attention, num_classes, image_size, patch_size, emb_dim, num_layers,
                   emb_dropout_rate, proj_dropout_rate, droppath_rate, in_channels=in_channels,
                   norm=partial(nn.LayerNorm, eps=layernorm_eps), activation=activation)


class ViTEMA(EMA):
    def param_groups(self):
        return self.original_model.param_groups()


@ViTs.register
def vit_b16(**kwargs) -> ViT:
    return ViT.construct(768, 12, 12, 16, droppath_rate=0.1, **kwargs)
