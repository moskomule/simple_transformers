from typing import Optional, Type

import torch
from homura import Registry
from torch import nn

from .attentions import SelfAttention

BLOCK = Registry("block", nn.Module)


def act_func(name: str
             ) -> nn.Module:
    _acts = {"relu": nn.ReLU,
             "leaky_relu": nn.LeakyReLU,
             "gelu": nn.GELU,
             "silu": nn.SiLU}
    return _acts[name]()


class BlockBase(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 widen_factor: int = 4,
                 activation: str = "gelu",
                 norm: Type[nn.LayerNorm] = nn.LayerNorm):
        super().__init__()
        self.ln1 = norm(emb_dim)
        self.ln2 = norm(emb_dim)
        self.attention = attention
        self.mlp = nn.Sequential(nn.Linear(emb_dim, widen_factor * emb_dim),
                                 act_func(activation),
                                 nn.Linear(widen_factor * emb_dim, emb_dim),
                                 nn.Dropout(dropout_rate))

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
        x = x + self.attention(x, mask)
        x = self.ln1(x)
        x = x + self.mlp(x)
        return self.ln2(x)


@BLOCK.register(name="pre_ln")
class PreLNBlock(BlockBase):
    """ BERT's Transformer Block
    """

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = input
        x = self.ln1(x)
        x = x + self.attention(x, mask)
        x = self.ln2(x)
        return x + self.mlp(x)


@BLOCK.register(name="ipre_ln")
class ImprovedPreLNBlock(BlockBase):
    """ Megatron-LM's Transformer Block
    """

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        x = input
        x = x + self.attention(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


@BLOCK.register(name="timm_ln")
class TimmPreLNBlock(ImprovedPreLNBlock):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 widen_factor: int = 4,
                 activation: str = "gelu",
                 norm: Type[nn.LayerNorm] = nn.LayerNorm):
        super().__init__(emb_dim, attention, dropout_rate, widen_factor, activation, norm)
        # double dropout
        self.mlp = nn.Sequential(nn.Linear(emb_dim, widen_factor * emb_dim),
                                 act_func(activation),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(widen_factor * emb_dim, emb_dim),
                                 nn.Dropout(dropout_rate))
