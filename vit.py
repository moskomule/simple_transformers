import chika
import homura
import torch
from homura import lr_scheduler, reporters
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY
from torch import nn
from torchvision.transforms import AutoAugment, RandomErasing

from models.vit import ViTEMA, ViTs
from vision_utils import fast_collate, gen_mix_collate


@torch.no_grad()
def accuracy(input: torch.Tensor,
             target: torch.Tensor,
             top_k: int = 1
             ) -> torch.Tensor:
    if target.ndim == 2:
        target = target.argmax(dim=1)
    pred_idx = input.argmax(dim=-1, keepdim=True) if top_k == 1 else input.topk(k=top_k, dim=-1).indices
    target = target.view(-1, 1).expand_as(pred_idx)
    return (pred_idx == target).float().sum(dim=1).mean()


class ViTTraner(SupervisedTrainer):
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
        self.optimizer = torch.optim._multi_tensor.AdamW(optim_groups,
                                                         lr=self.optim_cfg.lr,
                                                         weight_decay=self.optim_cfg.weight_decay)
        self.logger.debug(self.optimizer)

    def iteration(self,
                  data: tuple[torch.Tensor, torch.Tensor]
                  ) -> None:
        # will be removed once new accuracy is merged into homura
        input, target = data
        with torch.cuda.amp.autocast(self._use_amp):
            output = self.model(input)
            loss = self.loss_f(output, target)

        if self.is_train:
            self.optimizer.zero_grad()
            if self._use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.update_scheduler_iter:
                self.scheduler.step()
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")

        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(f'accuracy@{top_k}', accuracy(output, target, top_k))


@chika.config
class DataConfig:
    batch_size: int = 128
    autoaugment: bool = False
    random_erasing: bool = False
    mixup: float = 0
    cutmix: float = 0


@chika.config
class ModelConfig:
    name: str = chika.choices(*ViTs.choices())
    dropout_rate: float = 0
    droppath_rate: float = 0
    ema: bool = False
    ema_rate: float = chika.bounded(0.999, 0, 1)


@chika.config
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    epochs: int = 200
    min_lr: float = 1e-5
    warmup_epochs: int = 5


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    num_workers: int = 16
    debug: bool = False
    amp: bool = False
    gpu: int = None
    no_save: bool = False

    def __post_init__(self):
        assert self.optim.lr > self.optim.min_lr
        self.optim.lr *= self.data.batch_size * homura.get_world_size() / 512
        self.optim.min_lr *= self.data.batch_size * homura.get_world_size() / 512


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
    model = ViTs(cfg.model.name)(droppath_rate=cfg.model.droppath_rate, dropout_rate=cfg.model.dropout_rate)
    train_da = vs.default_train_da.copy()
    test_da = vs.default_test_da.copy()
    train_da[0].size = model.image_size
    test_da[0].size = model.image_size
    test_da[1].size = model.image_size
    if cfg.data.autoaugment:
        train_da.append(AutoAugment())
    post_da = [RandomErasing()] if cfg.data.random_erasing else None
    train_loader, test_loader = vs(batch_size=cfg.data.batch_size,
                                   train_da=train_da,
                                   test_da=test_da,
                                   post_norm_train_da=post_da,
                                   train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   num_workers=cfg.num_workers)
    if cfg.model.ema:
        model = ViTEMA(model, cfg.model.ema_rate)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs,
                                                       warmup_epochs=cfg.optim.warmup_epochs,
                                                       min_lr=cfg.optim.min_lr)

    with ViTTraner(model,
                   None,
                   nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing),
                   reporters=[reporters.TensorboardReporter(".")],
                   scheduler=scheduler,
                   use_amp=cfg.amp,
                   use_cuda_nonblocking=True,
                   report_accuracy_topk=5,
                   optim_cfg=cfg.optim,
                   debug=cfg.debug
                   ) as trainer:
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
            if not cfg.no_save:
                trainer.save(f"outputs/{cfg.model.name}", f"{ep}")

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
