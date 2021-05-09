from typing import Tuple

import torch


def fast_collate(batch: list
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    # based on NVidia's Apex
    # but it's not faster than the default probably because transforms.ToTensor is too slow.
    imgs = torch.stack([img for img, target in batch], dim=0)
    targets = torch.tensor([target for img, target in batch], dtype=torch.int64)
    return imgs, targets


def gen_mixup_collate(alpha):
    # see https://github.com/moskomule/mixup.pytorch
    beta = torch.distributions.Beta(alpha + 1, alpha)

    def f(batch):
        tensors, targets = fast_collate(batch)
        indices = torch.randperm(tensors.size(0))
        _tensors = tensors.clone()[indices]
        gamma = beta.sample()
        tensors.mul_(gamma).add_(_tensors, alpha=1 - gamma)
        return tensors, targets

    return f
