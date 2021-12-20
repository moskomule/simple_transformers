""" Training script of ViT following He+21 (https://arxiv.org/abs/2111.06377).
"""
import pathlib

import chika
import homura
import torch
from homura import lr_scheduler, reporters
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY
from torch import nn
from torchvision.transforms import AutoAugment, RandAugment, InterpolationMode

from models.vit import ViTEMA, ViTs
from vision_utils import fast_collate, gen_mix_collate

try:
    from models import experimental
except ImportError as e:
    print('experimental not found!')


class ViTTrainer(SupervisedTrainer):
    def __init__(self, *args, **kwargs):
        self.optim_cfg = kwargs.pop('optim_cfg')
        super().__init__(*args, **kwargs)

    def set_optimizer(self
                      ) -> None:
        params_dict = self.accessible_model.param_groups
        optim_groups = [
            {"params": params_dict['decay'], "weight_decay": self.optim_cfg.weight_decay},
            {"params": params_dict['no_decay'], "weight_decay": 0}
        ]
        kwargs = dict(lr=self.optim_cfg.lr, betas=self.optim_cfg.betas, weight_decay=self.optim_cfg.weight_decay)
        optim = torch.optim._multi_tensor.AdamW
        if self.optim_cfg.zero:
            from torch.distributed.optim import ZeroRedundancyOptimizer as Zero
            self.optimizer = Zero(list(optim_groups[0]['params']), optim, **kwargs)
            self.optimizer.add_param_group(optim_groups[1])
        else:
            self.optimizer = optim(optim_groups, **kwargs)
        self.logger.debug(self.optimizer)

    def state_dict(self):
        if self.optim_cfg.zero:
            # todo: investigate why
            return {'model': self.accessible_model.state_dict(),
                    'epoch': self.epoch,
                    'use_sync_bn': self._use_sync_bn,
                    'use_amp': self._use_amp}
        else:
            return super().state_dict()


@chika.config
class DataConfig:
    batch_size: int = 4_096
    autoaugment: bool = False
    no_randaugment: bool = False
    mixup: float = 0.8
    cutmix: float = 1.0

    def __post_init__(self):
        self.randaugment = not self.no_randaugment


@chika.config
class ModelConfig:
    name: str = chika.choices(*ViTs.choices())
    dropout_rate: float = 0
    droppath_rate: float = 0
    no_ema: bool = False
    ema_rate: float = chika.bounded(0.9999, 0, 1)
    block: str = None
    init_method: str = chika.choices(None, 'fairseq')

    def __post_init__(self):
        self.ema = not self.no_ema


@chika.config
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.3
    label_smoothing: float = 0.1
    epochs: int = 200
    min_lr: float = 1e-7
    warmup_epochs: int = 20
    betas: list[float] = chika.sequence(0.9, 0.95, size=2)
    zero: bool = False
    grad_accum_steps: int = 1


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    num_workers: int = 16
    debug: bool = False
    amp: bool = False
    gpu: int = None
    checkpointing: bool = False
    no_save: bool = False

    def __post_init__(self):
        assert self.optim.lr > self.optim.min_lr
        # though He+21 uses loss scaling, it degenerates training in my environment...
        self.optim.lr *= self.batch_size * homura.get_world_size() / 256
        self.data.batch_size /= self.optim.grad_accum_steps


@chika.main(cfg_cls=Config, change_job_dir=True)
@homura.distributed_ready_main
def main(cfg: Config):
    if cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
    if homura.is_master():
        import rich
        rich.print(cfg)
    vs = DATASET_REGISTRY("imagenet")
    vs.collate_fn = fast_collate if cfg.data.mixup + cfg.data.cutmix == 0 else gen_mix_collate(vs.num_classes,
                                                                                               cfg.data.mixup,
                                                                                               cfg.data.cutmix)
    vs.test_collate_fn = fast_collate
    model = ViTs(cfg.model.name)(droppath_rate=cfg.model.droppath_rate, dropout_rate=cfg.model.dropout_rate,
                                 enable_checkpointing=cfg.checkpointing, block=cfg.model.block,
                                 init_method=cfg.model.init_method)
    train_da = vs.default_train_da.copy()
    test_da = vs.default_test_da.copy()
    train_da[0].size = model.image_size
    test_da[0].size = model.image_size
    test_da[1].size = model.image_size
    if cfg.data.autoaugment:
        train_da.append(AutoAugment(interpolation=InterpolationMode.BILINEAR))
    if cfg.data.randaugment:
        train_da.append(RandAugment(interpolation=InterpolationMode.BILINEAR))
    train_loader, test_loader = vs(batch_size=cfg.data.batch_size,
                                   train_da=train_da,
                                   test_da=test_da,
                                   train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   num_workers=cfg.num_workers)
    if cfg.model.ema:
        model = ViTEMA(model, cfg.model.ema_rate)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       warmup_epochs=cfg.optim.warmup_epochs,
                                                       min_lr=cfg.optim.min_lr)

    with ViTTrainer(model,
                    None,
                    nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing),
                    reporters=[reporters.TensorboardReporter(".")],
                    scheduler=scheduler,
                    use_amp=cfg.amp,
                    use_cuda_nonblocking=True,
                    report_accuracy_topk=5,
                    optim_cfg=cfg.optim,
                    debug=cfg.debug,
                    grad_accum_steps=cfg.optim.grad_accum_steps
                    ) as trainer:
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
            if not cfg.no_save:
                trainer.save(f"outputs/{cfg.model.name}", f"{ep}")

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")
        if not homura.is_master():
            import shutil
            shutil.rmtree(pathlib.Path(".").resolve())


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
