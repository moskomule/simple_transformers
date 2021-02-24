import functools
import math
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

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
BLOCK = Registry("block", nn.Module)


@ATTENTIONS.register(name="dotprod")
def dotproduct_self_attention(query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              mask: Optional[torch.Tensor] = None,
                              dropout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
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
    context = einsum("bhkn,bhkm->bhmn", query, key).div(math.sqrt(query.size(1)))
    if mask is not None:
        size = context.size(-1)
        context = context.masked_fill(mask[:, :, :size, :size], float('-inf'))
    context = context.softmax(dim=-1)
    if dropout is not None:
        context = dropout(context)
    return einsum("bhmn,bhvm->bhvn", context, value)


class CausalSelfAttention(nn.Module):
    def __init__(self,
                 max_len: int,
                 emb_dim: int,
                 num_heads: int,
                 attn_dropout_rate: float,
                 proj_dropout_rate: float):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.value = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.proj_dropout = nn.Dropout(proj_dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)[None, None])

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        # input: BxNxC
        b = input.size(0)
        # BxNxC -> BxCxN -> BxHxC'xN
        key = self.key(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)
        query = self.query(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)
        value = self.value(input).transpose(-1, -2).view(b, self.num_heads, self.emb_dim // self.num_heads, -1)
        attention = dotproduct_self_attention(query, key, value, self.mask, self.attn_dropout)
        attention = attention.reshape(b, self.emb_dim, -1).transpose(-1, -2)
        return self.proj_dropout(self.proj(attention))


class BlockBase(nn.Module):
    def __init__(self,
                 ebm_dim: int,
                 attention: CausalSelfAttention,
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
                input: torch.Tensor
                ) -> torch.Tensor:
        raise NotImplementedError


@BLOCK.register(name="post_ln")
class PostLNBlock(BlockBase):
    """ Transformer Block from "Attention is All You Need"
    """

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = input
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        return self.ln2(x)


@BLOCK.register(name="pre_ln")
class PreLNBlock(BlockBase):
    """ BERT's Transformer Block
    """

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = input
        x = self.ln1(x)
        x = x + self.attention(x)
        x = self.ln2(x)
        return x + self.mlp(x)


@BLOCK.register(name="ipre_ln")
class ImprovedPreLNBlock(BlockBase):
    """ Megatron-LM's Transformer Block
    """

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = input
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self,
                 block: BlockBase,
                 vocab_size: int,
                 max_len: int,
                 emb_dim: int,
                 num_layers: int,
                 emb_dropout_rate: float
                 ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        self.dropout = nn.Dropout(emb_dropout_rate)
        self.blocks = nn.Sequential(*[deepcopy(block) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, vocab_size, bias=False))
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
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = input.size()
        token_emb = self.tok_emb(input)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.dropout(token_emb + pos_emb)  # BxNxC
        x = self.blocks(x)  # BxNxC
        logits = self.head(x)  # BxNxV
        return logits

    @property
    def param_groups(self
                     ) -> Dict[str, List]:

        decay = set()
        no_decay = set()
        apply_decay = (nn.Linear,)
        no_apply_decay = (nn.LayerNorm, nn.Embedding)
        for name, param in self.named_parameters():
            if "pos_emb" in name:
                no_decay.add(param)
        for module in self.modules():
            if isinstance(module, no_apply_decay):
                for param in module.parameters():
                    no_decay.add(param)
            elif isinstance(module, apply_decay):
                decay.add(module.weight)
                if module.bias is not None:
                    no_decay.add(module.bias)
        assert len([param for param in self.parameters()]) == len(decay) + len(no_decay)
        return {"decay": list(decay), "no_decay": list(no_decay)}

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
                  **kwargs
                  ):
        block = BLOCK(block)(emb_dim,
                             CausalSelfAttention(max_len, emb_dim, num_heads, attn_dropout_rate, proj_dropout_rate),
                             proj_dropout_rate)
        return cls(block, vocab_size, max_len, emb_dim, num_layers, emb_dropout_rate)
