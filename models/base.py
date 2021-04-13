from __future__ import annotations

from torch import nn
from torch.nn.modules.conv import _ConvNd


class TransformerBase(nn.Module):
    """ Baseclass supports checkpointing and weight_decay branching

    """

    def __init__(self,
                 blocks: nn.Sequential,
                 checkpointing: bool):
        super().__init__()
        self._blocks = blocks
        self._blocks_train = self._blocks
        if checkpointing:
            from fairscale.nn.misc import checkpoint_wrapper
            self._blocks_train = checkpoint_wrapper(self._blocks)

    @property
    def blocks(self):
        if self.training:
            return self._blocks_train
        else:
            return self._blocks

    def init_weights(self):
        raise NotImplementedError

    @property
    def param_groups(self
                     ) -> dict[str, list]:

        decay = set()
        no_decay = set()
        apply_decay = (nn.Linear, _ConvNd)
        no_apply_decay = (nn.LayerNorm, nn.Embedding)
        for name, param in self.named_parameters():
            if "pos_emb" in name or "token" in name:
                no_decay.add(param)
        for module in self.modules():
            if isinstance(module, no_apply_decay):
                for param in module.parameters():
                    no_decay.add(param)
            elif isinstance(module, apply_decay):
                decay.add(module.weight)
                if module.bias is not None:
                    no_decay.add(module.bias)
        for param in self.parameters():
            if param not in no_decay:
                decay.add(param)
        assert len([param for param in self.parameters()]) == len(decay) + len(no_decay)
        return {"decay": list(decay), "no_decay": list(no_decay)}

    @classmethod
    def construct(cls,
                  *args,
                  **kwargs
                  ) -> TransformerBase:
        raise NotImplementedError
