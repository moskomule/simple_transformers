from __future__ import annotations

from typing import Tuple

import chika
import homura
import torch

from models import GPT
from utils import GPTTrainer, get_data


@chika.config
class DataConfig:
    name: str = chika.choices("wikitext", "gigaword")
    batch_size: int = 256
    max_len: int = 128
    train_full: bool = False


@chika.config
class OptimConfig:
    epochs: int = 20
    name: str = chika.choices("adam", "adamw")
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float] = chika.sequence(0.9, 0.95)
    warmup_iters: int = 1_000
    multi_tensor: bool = False


@chika.config
class ModelConfig:
    block: str = chika.choices("ipre_ln", "pre_ln", "post_ln")
    grad_norm_clip: float = 1.0

    num_heads: int = 8
    emb_dim: int = 512
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
            trainer.save("outputs", f"{ep}.pt")


if __name__ == "__main__":
    import warnings

    # to avoid "Detected call of `lr_scheduler.step()` before `optimizer.step()`... when using AMP
    warnings.filterwarnings("ignore", message="Detected call of")
    main()
