from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from typing import Type, Optional

import torch
from homura import Registry
from homura.modules import EMA
from torch import nn

from .attentions import SelfAttention
from .base import TransformerBase
from .blocks import LayerScaleBlock, TimmPreLNBlock, ACT, BLOCK
from .embeddings import PatchEmbed2d

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
                 enable_checkpointing=False,
                 init_method: str = None
                 ):
        super().__init__(blocks, enable_checkpointing)
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = math.prod(image_size) // math.prod(patch_size)

        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_emb = PatchEmbed2d(patch_size, emb_dim, in_channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
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
        x = self.dropout(self.pos_emb + x)
        x = self.norm(self.blocks(x))
        return self.fc(x[:, 0])

    def init_weights(self,
                     method: str = None):
        assert method in (None, 'fairseq')
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'fairseq':
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
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
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
                  mlp_widen_factor: int = 4,
                  block: str = None,
                  enable_checkpointing: bool = False,
                  init_method: str = None
                  ) -> ViT:
        norm = partial(nn.LayerNorm, eps=layernorm_eps)
        activation = ACT(activation)
        attention = SelfAttention(emb_dim, num_heads, attn_dropout_rate, dropout_rate)
        block_kwargs = dict(dropout_rate=dropout_rate, widen_factor=mlp_widen_factor, norm=norm, activation=activation)
        if block is None:
            blocks = [TimmPreLNBlock(emb_dim, deepcopy(attention), droppath_rate=r, **block_kwargs)
                      for r in [x.item() for x in torch.linspace(0, droppath_rate, num_layers)]]
        else:
            blocks = [BLOCK(block)(emb_dim, deepcopy(attention), **block_kwargs) for _ in range(num_layers)]

        return cls(nn.Sequential(*blocks), num_classes, image_size, patch_size, emb_dim, dropout_rate, in_channels,
                   norm, enable_checkpointing=enable_checkpointing, init_method=init_method)


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


class ClassAttention(SelfAttention):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 attn_dropout_rate: float,
                 proj_dropout_rate: float,
                 qkv_bias: bool = True,
                 proj_bias: bool = True):
        super().__init__(emb_dim, num_heads, attn_dropout_rate, proj_dropout_rate, qkv_bias, proj_bias)
        self.qkv = nn.Linear(emb_dim, 2 * emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # input: BxNxC
        b, n, c = input.size()
        # BxC -> Bx1xHxC'
        query = self.query(input[:, 0]).view(b, 1, self.num_heads, -1)
        # BxNx2C -> BxNxHxC'
        key, value = self.qkv(input).view(b, n, 2, self.num_heads, -1).unbind(2)
        attention = self.attn_fn(query, key, value, mask, self.attn_dropout).reshape(b, self.emb_dim, -1)
        return self.proj_dropout(self.proj(attention))


class CaiTSequential(nn.Sequential):
    # a helper module for CaiT to use gradient checkpointing
    def forward(self,
                input: torch.Tensor,
                cls_token: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            if isinstance(module.attention, ClassAttention):
                cls_token = module(input, cls_token)
            else:
                input = module(input)
        return input, cls_token


class CaiT(TransformerBase):
    """ CaiT from Touvron+2021 Going deeper with Image Transformers. https://github.com/facebookresearch/deit
    """

    def __init__(self,
                 attention: SelfAttention,
                 cls_attention: ClassAttention,
                 num_classes: int,
                 image_size: int or tuple,
                 patch_size: int or tuple,
                 emb_dim: int,
                 num_layers: int,
                 num_cls_layers: int,
                 emb_dropout_rate: float,
                 dropout_rate: float,
                 droppath_rate: float,
                 in_channels: int,
                 norm: Type[nn.LayerNorm] = nn.LayerNorm,
                 mlp_widen_factor: int = 4,
                 activation: str = "gelu",
                 enable_checkpointing=False,
                 init_scale: float = 1e-5
                 ):
        activation = ACT(activation)
        blocks1 = [LayerScaleBlock(emb_dim, deepcopy(attention), dropout_rate=dropout_rate, droppath_rate=droppath_rate,
                                   widen_factor=mlp_widen_factor, norm=norm, activation=activation,
                                   init_scale=init_scale)
                   for _ in range(num_layers)]
        blocks2 = [LayerScaleBlock(emb_dim, deepcopy(cls_attention), dropout_rate=0, droppath_rate=0,
                                   widen_factor=mlp_widen_factor, norm=norm, activation=activation,
                                   init_scale=init_scale)
                   for _ in range(num_cls_layers)]
        blocks = blocks1 + blocks2
        super().__init__(CaiTSequential(*blocks), enable_checkpointing)
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = math.prod(image_size) // math.prod(patch_size)

        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_emb = PatchEmbed2d(patch_size, emb_dim, in_channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
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
        x = self.dropout(self.pos_emb + x)
        x = torch.cat(self.blocks(x, cls_token), dim=1)
        x = self.norm(x)
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

        for name, param in self.named_parameters():
            if "talk" in name:
                nn.init.trunc_normal_(param, std=0.02)

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
                  num_cls_layers: int,
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
                  **kwargs
                  ) -> CaiT:
        attention = SelfAttention(emb_dim, num_heads, attn_dropout_rate, dropout_rate, talking_heads=True)
        cls_attention = ClassAttention(emb_dim, num_heads, attn_dropout_rate, dropout_rate)
        return cls(attention, cls_attention, num_classes, image_size, patch_size, emb_dim, num_layers, num_cls_layers,
                   dropout_rate, dropout_rate, droppath_rate, in_channels=in_channels,
                   norm=partial(nn.LayerNorm, eps=layernorm_eps), activation=activation, **kwargs)


@ViTs.register
def cait_xs24(**kwargs):
    return CaiT.construct(emb_dim=288, num_layers=24, num_cls_layers=2, num_heads=6, patch_size=16, **kwargs)


@ViTs.register
def cait_s24_224(**kwargs):
    return CaiT.construct(emb_dim=384, num_layers=24, num_cls_layers=2, num_heads=8, patch_size=16, **kwargs)


@ViTs.register
def cait_s24_384(**kwargs):
    return CaiT.construct(emb_dim=384, num_layers=24, num_cls_layers=2, num_heads=8, patch_size=16, image_size=384,
                          **kwargs)


@ViTs.register
def cait_m36_384(**kwargs):
    return CaiT.construct(emb_dim=384, num_layers=36, num_cls_layers=2, num_heads=8, patch_size=16, image_size=384,
                          init_scale=1e-6, **kwargs)


ViTs.register(cait_s24_384, name="cait_s24")
ViTs.register(cait_m36_384, name="cait_m36")
