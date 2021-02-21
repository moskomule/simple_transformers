import pathlib
from os import environ
from typing import Dict, Tuple

import datasets
import torch
from homura.trainers import SupervisedTrainer
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from torch.utils.data import DataLoader


def get_data(batch_size,
             max_len,
             num_workers=4,
             train_full=False
             ):
    max_len += 1
    environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer_path = pathlib.Path(f"wikitext_tokenizer{max_len}.json")
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.enable_truncation(max_length=max_len)
        tokenizer.enable_padding(length=max_len)

        def batch_iterator(bs):
            for i in range(0, len(dataset), bs):
                yield dataset[i: i + bs]["text"]

        tokenizer.train_from_iterator(batch_iterator(1_000), length=len(dataset))
        tokenizer.save(str(tokenizer_path))

    train_ds, val_ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split=['train' if train_full
                                                                                       else 'train[:10%]',
                                                                                       'validation'])
    train_ds = train_ds.map(lambda sent: {"ids": tokenizer.encode(sent['text']).ids})
    val_ds = val_ds.map(lambda sent: {"ids": tokenizer.encode(sent['text']).ids})
    train_ds.set_format(type='torch', columns=['ids'])
    val_ds.set_format(type='torch', columns=['ids'])

    return (
        DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers),
        tokenizer,
        tokenizer.get_vocab_size()
    )


class GPTTrainer(SupervisedTrainer):
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs.pop('cfg')
        self.optim_cfg = kwargs.pop('optim_cfg')
        super().__init__(*args, **kwargs)

    def set_optimizer(self
                      ) -> None:
        params_dict = self.model.param_groups
        optim_groups = [
            {"params": params_dict['decay'], "weight_decay": self.optim_cfg.weight_decay},
            {"params": params_dict['no_decay'], "weight_decay": 0}
        ]
        cls = torch.optim._multi_tensor.AdamW if self.optim_cfg.multi_tensor else torch.optim.AdamW
        self.optimizer = cls(optim_groups, lr=self.optim_cfg.lr, betas=self.optim_cfg.betas)

    def data_preprocess(self,
                        data: Dict[str, torch.Tensor]
                        ) -> Tuple[torch.Tensor, int]:
        tensor = data['ids']
        return tensor.to(self.device, non_blocking=self._cuda_nonblocking), tensor.size(0)

    def iteration(self,
                  data: torch.Tensor
                  ) -> None:
        input, target = data[:, :-1], data[:, 1:]
        with torch.cuda.amp.autocast(self._use_amp):
            logits, loss = self.model(input, target)
        self.reporter.add("loss", loss.detach())
        if self.is_train:
            self.optimizer.zero_grad(set_to_none=True)
            if self._use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm_clip)
            if self._use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

    @torch.no_grad()
    def sample(self,
               x: torch.Tensor,
               num_steps: int,
               temperature: float = 1.0,
               sampling: bool = False,
               ) -> torch.Tensor:
        x = x.clone()
        self.model.eval()
        max_len = self.model.max_len
        for k in range(num_steps):
            x_cond = x if x.size(1) <= max_len else x[:, -max_len:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax(dim=-1)
            if sampling:
                next = torch.multinomial(probs, num_samples=1)
            else:
                next = probs.argmax(dim=-1)
            x = torch.cat([x, next], dim=1)
        return x
