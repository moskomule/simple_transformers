from typing import Optional

import torch
from homura import Registry
from torch import nn

from .attentions import SelfAttention

BLOCK = Registry("block", nn.Module)


class BlockBase(nn.Module):
    def __init__(self,
                 ebm_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(ebm_dim)
        self.ln2 = nn.LayerNorm(ebm_dim)
        self.attention = attention
        self.mlp = nn.Sequential(nn.Linear(ebm_dim, 4 * ebm_dim),
                                 nn.GELU(),
                                 nn.Linear(4 * ebm_dim, ebm_dim),
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
