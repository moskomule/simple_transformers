from __future__ import annotations

from typing import Dict, Optional, Tuple

import chika
import homura
import torch
from homura import TensorTuple
from homura.trainers import SupervisedTrainer
from torch.nn import functional as F

from models.gpt import GPT
from nlp_utils import get_data


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
        target = target.masked_fill(mask[:, 1:] == 0, ignore_index)
        with torch.cuda.amp.autocast(self._use_amp):
            logits = self.model(input, mask[:, 1:])
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


@chika.config
class DataConfig:
    name: str = chika.choices("wikitext", "gigaword")
    batch_size: int = 64
    max_len: int = 150
    train_full: bool = False


@chika.config
class OptimConfig:
    epochs: int = 20
    name: str = chika.choices("adamw", "adam")
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: Tuple[float] = chika.sequence(0.9, 0.98)
    warmup_iters: int = 1_000
    multi_tensor: bool = False


@chika.config
class ModelConfig:
    block: str = chika.choices("ipre_ln", "pre_ln", "post_ln")
    grad_norm_clip: float = 1.0

    num_heads: int = 8
    emb_dim: int = 768
    num_layers: int = 12
    emb_dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    proj_dropout_rate: float = 0.1

    enable_checkpoint: bool = False


@chika.config
class Config:
    model: ModelConfig
    optim: OptimConfig
    data: DataConfig
    seed: int = 1
    gpu: int = 0
    amp: bool = False


@chika.main(cfg_cls=Config, strict=True)
def main(cfg: Config):
    print(cfg)
    torch.cuda.set_device(cfg.gpu)
    homura.set_seed(cfg.seed)
    train_loader, val_loader, tokenizer, vocab_size = get_data(**cfg.data.to_dict())
    model = GPT.construct(**cfg.model.to_dict(), vocab_size=vocab_size, max_len=cfg.data.max_len)
    # optimizer is setup automatically
    scheduler = homura.lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs * len(train_loader), 1,
                                                              cfg.optim.warmup_iters)
    sample_text = tokenizer.encode("however, as can be seen from")
    # sample_text = tokenizer.encode("in the beginning was the word")
    sample_tensor = torch.tensor(sample_text.ids[:sum(sample_text.attention_mask)]).view(1, -1)
    with GPTTrainer(model, None, None,
                    reporters=[homura.reporters.TensorboardReporter(".")],
                    scheduler=scheduler,
                    cfg=cfg.model,
                    optim_cfg=cfg.optim,
                    use_amp=cfg.amp
                    ) as trainer:
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(val_loader, "val")
            sampled = trainer.sample(sample_tensor.to(trainer.device), num_steps=64, sampling=True, only_tok_k=100)
            sampled_text = tokenizer.decode(sampled.view(-1).cpu().tolist(), False)
            print(f"[{ep:>4}] train loss = {trainer.history['loss/train'][-1]:.3e}"
                  f" val loss={trainer.history['loss/val'][-1]:.3e}|| {sampled_text}")
            trainer.save("outputs", f"{ep}")


if __name__ == "__main__":
    import warnings

    # to avoid "Detected call of `lr_scheduler.step()` before `optimizer.step()`... when using AMP
    warnings.filterwarnings("ignore", message="Detected call of")
    main()
