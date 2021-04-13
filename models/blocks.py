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


class TimmPreLNBlock(BlockBase):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 droppath_rate: float,
                 widen_factor: int,
                 activation: str,
                 norm: Type[nn.LayerNorm]):
        super().__init__(emb_dim, attention, dropout_rate, widen_factor, activation, norm)
        # double dropout
        self.mlp = nn.Sequential(nn.Linear(emb_dim, widen_factor * emb_dim),
                                 act_func(activation),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(widen_factor * emb_dim, emb_dim),
                                 nn.Dropout(dropout_rate))
        self.droppath_rate = droppath_rate
        self.emb_dim = emb_dim

    def forward(self,
                input: torch.Tensor,
                mask=None
                ) -> torch.Tensor:
        x = input
        x = x + self.drop_path(self.attention(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

    def drop_path(self,
                  input: torch.Tensor,
                  ) -> torch.Tensor:
        if not self.training or self.droppath_rate == 0:
            return input

        keep_prob = 1 - self.droppath_rate
        # 1 with prob. of keep_prob
        drop = input.new_empty(input.size(0), 1, 1).bernoulli_(keep_prob)
        return input.div(keep_prob).mul(drop)


class LayerScaleBlock(TimmPreLNBlock):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 droppath_rate: float,
                 widen_factor: int,
                 activation: str,
                 norm: Type[nn.LayerNorm],
                 init_scale: float):
        super().__init__(emb_dim, attention, dropout_rate, droppath_rate, widen_factor, activation, norm)
        self.gamma1 = nn.Parameter(init_scale * torch.ones(emb_dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(emb_dim))

    def forward(self,
                input: torch.Tensor,
                cls_token: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if cls_token is None:
            x = input
        else:
            x = cls_token
            input = torch.cat((cls_token, input), dim=1)
        x = x + self.drop_path(self.gamma1 * self.attention(self.ln1(input)))
        x = x + self.drop_path(self.gamma1 * self.mlp(self.ln2(x)))
        return x
