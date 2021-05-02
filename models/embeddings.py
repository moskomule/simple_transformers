import torch
from torch import nn


class PatchEmbed2d(nn.Module):
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
