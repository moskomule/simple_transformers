from __future__ import annotations

import functools
import math
from typing import Callable, Optional

import torch
from homura import Registry
from torch import nn

try:
    import opt_einsum

    print("opt_einsum is installed, so einsum=opt_einsum")

    einsum = functools.partial(opt_einsum.contract, backend="torch")

except ImportError:
    print("no opt_einsum")

    einsum = torch.einsum

ATTENTIONS = Registry("attentions", )


@ATTENTIONS.register(name="dotprod")
def dotproduct_self_attention(query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor] = None,
                              dropout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              pre_talk: Optional[torch.Tensor] = None,
                              post_talk: Optional[torch.Tensor] = None,
                              ) -> torch.Tensor:
    """ dot-product self-attention

    Args:
        query: tensor of shape BHKN
        key: tensor of shape BHKM
        value: tensor of shape BHVN
        mask:
        dropout:

    Returns: results

    """

    # attn/\sqrt{dim_head}
    context = einsum("bhkn,bhkm->bhmn", query, key).div(math.sqrt(query.size(-2)))
    if pre_talk is not None:
        context = einsum("bhmn,hk->bkmn", context, pre_talk)
    if mask is not None:
        size = context.size(-1)
        context = context.masked_fill(mask[:, :, :size, :size] == 0, float('-inf'))
    context = context.softmax(dim=-1)
    if post_talk is not None:
        context = einsum("bkmn,kh->bhmn", context, post_talk)
    if dropout is not None:
        context = dropout(context)
    return einsum("bhmn,bhvm->bhvn", context, value)


class SelfAttention(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 attn_dropout_rate: float,
                 proj_dropout_rate: float,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 talking_heads: bool = False
                 ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.key = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self._query = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.value = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=proj_bias)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)
        self.talking_heads = talking_heads
        if self.talking_heads:
            self.pre_talk = nn.Parameter(torch.randn(self.num_heads, self.num_heads))
            self.post_talk = nn.Parameter(torch.randn(self.num_heads, self.num_heads))

    def query(self,
              input: torch.Tensor
              ) -> torch.Tensor:
        b = input.size(0)
        return self._query(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # input: BxNxC
        b = input.size(0)
        # BxNxC -> BxCxN -> BxHxC'xN
        query = self.query(input)
        key = self.key(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)
        value = self.value(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)
        attention = dotproduct_self_attention(query, key, value, mask, self.attn_dropout)
        attention = attention.reshape(b, self.emb_dim, -1).transpose(-1, -2)
        return self.proj_dropout(self.proj(attention))


class ClassAttention(SelfAttention):
    def query(self,
              input: torch.Tensor
              ) -> torch.Tensor:
        # BxC->BxHxC'x1
        return self._query(input[:, 0]).view(input.size(0), self.num_heads, self.emb_dim // self.num_heads, 1)
