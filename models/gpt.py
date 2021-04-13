import warnings
from copy import deepcopy
from typing import Optional, Tuple

import torch
from torch import nn

from .attentions import SelfAttention
from .base import TransformerBase
from .blocks import BLOCK, BlockBase


class MaskedSequential(nn.Sequential):
    def forward(self,
                input: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        for module in self:
            input = module(input, mask)
        return input


class GPT(TransformerBase):
    def __init__(self,
                 block: BlockBase,
                 vocab_size: int,
                 max_len: int,
                 emb_dim: int,
                 num_layers: int,
                 emb_dropout_rate: float,
                 enable_checkpoint: bool = False
                 ):
        super().__init__(MaskedSequential(*[deepcopy(block) for _ in range(num_layers)]), enable_checkpoint)
        self.max_len = max_len
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        self.dropout = nn.Dropout(emb_dropout_rate)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, vocab_size, bias=False))
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))[None, None])
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, 0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = input.size()
        mask = self.mask if mask is None else self.mask & mask.bool()
        token_emb = self.tok_emb(input)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.dropout(token_emb + pos_emb)  # BxNxC
        x = self.blocks(x, None if mask is None else mask[:, None, None, :])
        logits = self.head(x)  # BxNxV
        return logits

    @classmethod
    def construct(cls,
                  block: str,
                  vocab_size: int,
                  max_len: int,
                  num_heads: int = 12,
                  emb_dim: int = 768,
                  num_layers: int = 12,
                  emb_dropout_rate: float = 0.1,
                  attn_dropout_rate: float = 0.1,
                  proj_dropout_rate: float = 0.1,
                  enable_checkpoint: bool = False,
                  **kwargs
                  ):
        if len(kwargs) > 0:
            warnings.warn(f"kwargs={kwargs} are not used")
        block = BLOCK(block)(emb_dim,
                             SelfAttention(emb_dim, num_heads, attn_dropout_rate, proj_dropout_rate),
                             proj_dropout_rate)
        return cls(block, vocab_size, max_len, emb_dim, num_layers, emb_dropout_rate, enable_checkpoint)
