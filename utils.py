import pathlib
from os import environ
from typing import Dict, Optional, Tuple

import datasets
import torch
from datasets.utils.logging import set_verbosity_error
from homura import TensorTuple
from homura.trainers import SupervisedTrainer
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from torch.nn import functional as F
from torch.utils.data import DataLoader


def get_data(name,
             batch_size,
             max_len,
             num_workers=4,
             train_full=False
             ):
    max_len += 1
    datasets.disable_progress_bar()
    set_verbosity_error()

    # followed https://github.com/EleutherAI/gpt-neo/
    environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer_path = pathlib.Path(f"{name}_tokenizer{max_len}.json")
    _name = {"wikitext": ("wikitext", "wikitext-103-v1"),
             "gigaword": ("gigaword",)
             }[name]
    _column_name = {"wikitext": "text",
                    "gigaword": "document"
                    }[name]
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        dataset = datasets.load_dataset(*_name, split="train+test+validation")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.enable_truncation(max_length=max_len)
        tokenizer.enable_padding(length=max_len)
        trainer = trainers.BpeTrainer(min_frequency=2, special_tokens=["<EOS>", "<PAD>"])

        def batch_iterator(bs):
            for i in range(0, len(dataset), bs):
                yield dataset[i: i + bs][_column_name]

        tokenizer.train_from_iterator(batch_iterator(1_000), trainer=trainer, length=len(dataset))
        tokenizer.save(str(tokenizer_path))

    train_ds, val_ds = datasets.load_dataset(*_name,
                                             split=['train' if train_full else 'train[:20%]', 'validation'])

    def to_ids(sent):
        tokenized = tokenizer.encode(sent[_column_name])
        return {"ids": tokenized.ids, "mask": tokenized.attention_mask}

    train_ds = train_ds.filter(lambda e: len(e[_column_name]) > 20).map(to_ids, num_proc=10)
    val_ds = val_ds.filter(lambda e: len(e[_column_name]) > 20).map(to_ids)
    train_ds.set_format(type='torch', columns=['ids', 'mask'])
    val_ds.set_format(type='torch', columns=['ids', 'mask'])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
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
        base = torch.optim._multi_tensor if self.optim_cfg.multi_tensor else torch.optim
        cls = getattr(base, "AdamW" if self.optim_cfg.name == "adamw" else "Adam")
        self.optimizer = cls(optim_groups, lr=self.optim_cfg.lr, betas=self.optim_cfg.betas)
        self.logger.debug(self.optimizer)

    def _loop(self,
              data_loader,
              mode: str
              ) -> None:

        self.inner_tqdm = self._tqdm(data_loader)
        for data in self.inner_tqdm:
            if self.is_train:
                # increment step here for `callbacks`
                self._step += 1
            self._iteration(data, mode)

        self.reporter.report(self.epoch, mode)
        self.logger.debug(f"epoch {self.epoch} finished")

    def data_preprocess(self,
                        data: Dict[str, torch.Tensor]
                        ) -> Tuple[torch.Tensor, int]:
        ids, mask = data['ids'], data['mask']
        return TensorTuple((ids, mask)).to(self.device, non_blocking=self._cuda_nonblocking), ids.size(0)

    def iteration(self,
                  data: torch.Tensor
                  ) -> None:
        ids, mask = data
        input, target = ids[:, :-1], ids[:, 1:]
        ignore_index = -100
        target = target.masked_fill(mask[:, :-1] == 0, ignore_index)
        with torch.cuda.amp.autocast(self._use_amp):
            logits = self.model(input, mask[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, -2), target.reshape(-1), ignore_index=ignore_index)
        self.reporter.add("loss", loss.detach())
        if self.is_train:
            self.optimizer.zero_grad(set_to_none=True)
            if self._use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.cfg.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm_clip)
            if self._use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            if self.step % 500 == 0:
                self.inner_tqdm.set_postfix({"loss": f"{loss.cpu().item():.3e}"})

    @torch.no_grad()
    def sample(self,
               x: torch.Tensor,
               num_steps: int,
               temperature: float = 1.0,
               sampling: bool = False,
               only_tok_k: Optional[int] = None
               ) -> torch.Tensor:
        x = x.clone()
        self.model.eval()
        max_len = self.model.max_len
        for k in range(num_steps):
            x_cond = x if x.size(1) <= max_len else x[:, -max_len:]
            logits = self.model(x_cond)
            logits = logits[:, -1, :] / temperature

            if only_tok_k is not None:
                val, idx = logits.topk(k=only_tok_k)
                logits[logits < val[:, [-1]]] = float('-inf')

            probs = logits.softmax(dim=-1)

            if sampling:
                next = torch.multinomial(probs, num_samples=1)
            else:
                next = probs.argmax(dim=-1)
            x = torch.cat([x, next], dim=1)
        return x
