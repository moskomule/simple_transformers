from __future__ import annotations

from typing import Optional, Type

import torch
from homura import Registry
from torch import nn
from torchvision.ops import StochasticDepth

from .attentions import SelfAttention

BLOCK = Registry("block", nn.Module)
ACT = Registry("activation", nn.Module)

ACT.register_from_dict(
    {"relu": nn.ReLU,
     "leaky_relu": nn.LeakyReLU,
     "gelu": nn.GELU,
     "silu": nn.SiLU}
)


class BlockBase(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 widen_factor: int = 4,
                 activation: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.LayerNorm] = nn.LayerNorm,
                 droppath_rate: float = None):
        super().__init__()
        self.ln1 = norm(emb_dim)
        self.ln2 = norm(emb_dim)
        self.attention = attention
        self.mlp = nn.Sequential(nn.Linear(emb_dim, widen_factor * emb_dim),
                                 activation(),
                                 nn.Linear(widen_factor * emb_dim, emb_dim),
                                 nn.Dropout(dropout_rate))
        self.droppath = nn.Identity()
        if droppath_rate is not None:
            self.droppath = StochasticDepth(droppath_rate, 'row')

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        raise NotImplementedError


@BLOCK.register(name="post_ln")
class PostLNBlock(BlockBase):
    """ Transformer Block from "Attention is All You Need"
    """

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = input
        x = x + self.droppath(self.attention(x, mask))
        x = self.ln1(x)
        x = x + self.droppath(self.mlp(x))
        return self.ln2(x)


@BLOCK.register(name="bert_pre_ln")
class BertPreLNBlock(BlockBase):
    """ BERT's Transformer Block
    """

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = input
        x = self.ln1(x)
        x = x + self.droppath(self.attention(x, mask))
        x = self.ln2(x)
        return x + self.droppath(self.mlp(x))


@BLOCK.register(name="pre_ln")
class PreLNBlock(BlockBase):
    """ Megatron-LM's Transformer Block
    """

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = input
        x = x + self.droppath(self.attention(self.ln1(x), mask))
        return x + self.droppath(self.mlp(self.ln2(x)))
