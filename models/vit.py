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
from .blocks import ACT, BLOCK
from .embeddings import PatchEmbed2d, LearnablePosEmbed2d, POSEMB2D

ViTs = Registry("vit", nn.Module)


class ViT(TransformerBase):
    """ Vision Transformer
    """

    def __init__(self,
                 blocks: nn.Sequential,
                 num_classes: int,
                 image_size: int or tuple,
                 patch_size: int or tuple,
                 emb_dim: int,
                 emb_dropout_rate: float,
                 in_channels: int,
                 norm: Type[nn.LayerNorm] = nn.LayerNorm,
                 pos_emb: nn.Module = None,
                 enable_checkpointing=False,
                 init_method: str = None
                 ):
        super().__init__(blocks, enable_checkpointing)
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_emb = PatchEmbed2d(patch_size, emb_dim, in_channels)
        self.pos_emb = pos_emb or LearnablePosEmbed2d(emb_dim, image_size // patch_size, True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.dropout = nn.Dropout(emb_dropout_rate)
        self.norm = norm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.init_weights(init_method)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.patch_emb(input)  # BxNxC
        b, n, c = x.size()
        cls_token = self.cls_token.expand(b, -1, -1)
        # x = torch.cat((cls_token, x), dim=1)  # Bx(N+1)xC
        tmp = input.new_empty(b, n + 1, c)
        tmp[:, :1].copy_(cls_token)
        tmp[:, 1:].copy_(x)
        x = tmp
        x = self.dropout(self.pos_emb(x))
        x = self.norm(self.blocks(x))
        return self.fc(x[:, 0])

    def init_weights(self,
                     method: str = None):
        assert method in (None, 'fairseq')
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if method == 'fairseq':
                    if 'qkv' in name:
                        # to treat QKV separately
                        s0, s1 = module.weight.shape
                        val = math.sqrt(6. / (s0 // 3 + s1))
                        nn.init.uniform_(module.weight, -val, val)
                    else:
                        nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, 1e-6)

            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        proj_w = self.patch_emb.proj
        fan_in = proj_w.in_channels * math.prod(proj_w.kernel_size)
        nn.init.trunc_normal_(proj_w.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(proj_w.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @classmethod
    def construct(cls,
                  emb_dim: int,
                  num_layers: int,
                  num_heads: int,
                  patch_size: int,
                  dropout_rate: float = 0,
                  attn_dropout_rate: float = 0,
                  droppath_rate: float = 0,
                  num_classes: int = 1_000,
                  image_size: int = 224,
                  in_channels: int = 3,
                  layernorm_eps: float = 1e-6,
                  activation: str = "gelu",
                  pos_emb: str = "learnable",
                  mlp_widen_factor: int = 4,
                  block: str = None,
                  enable_checkpointing: bool = False,
                  init_method: str = None,
                  ) -> ViT:
        norm = partial(nn.LayerNorm, eps=layernorm_eps)
        pos_emb = POSEMB2D(pos_emb)(emb_dim, image_size // patch_size, True)
        activation = ACT(activation)
        attention = SelfAttention(emb_dim, num_heads, attn_dropout_rate, dropout_rate)
        block_kwargs = dict(dropout_rate=dropout_rate, widen_factor=mlp_widen_factor, norm=norm, activation=activation)
        if block is None:
            block = "pre_ln"
        blocks = [BLOCK(block)(emb_dim, deepcopy(attention), droppath_rate=droppath_rate, **block_kwargs)
                  for _ in range(num_layers)]

        return cls(nn.Sequential(*blocks), num_classes, image_size, patch_size, emb_dim, dropout_rate, in_channels,
                   norm, pos_emb, enable_checkpointing=enable_checkpointing, init_method=init_method)


class ViTEMA(EMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_model.eval()

    @property
    def param_groups(self):
        return self.original_model.param_groups


@ViTs.register
def vit_t16(**kwargs) -> ViT:
    return ViT.construct(192, 12, 3, 16, **kwargs)


@ViTs.register
def vit_t16_384(**kwargs) -> ViT:
    return ViT.construct(192, 12, 3, 16, image_size=384, **kwargs)


@ViTs.register
def vit_b16(**kwargs) -> ViT:
    return ViT.construct(768, 12, 12, 16, **kwargs)


@ViTs.register
def vit_b16_384(**kwargs) -> ViT:
    return ViT.construct(768, 12, 12, 16, image_size=384, **kwargs)


@ViTs.register
def vit_b32(**kwargs) -> ViT:
    return ViT.construct(768, 12, 12, 32, **kwargs)


@ViTs.register
def vit_l16(**kwargs) -> ViT:
    return ViT.construct(1024, 24, 16, 16, **kwargs)


@ViTs.register
def vit_l32(**kwargs) -> ViT:
    return ViT.construct(1024, 24, 16, 32, **kwargs)
