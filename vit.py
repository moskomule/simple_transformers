import chika
import homura
import torch
from homura import lr_scheduler, reporters
from homura.modules import SmoothedCrossEntropy
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY
from torchvision.transforms import AutoAugment, RandomErasing

from models.vit import ViTEMA, ViTs
from utils import distributed_ready_main


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


@chika.config
class DataConfig:
    batch_size: int = 128
    autoaugment: bool = False
    random_erasing: bool = False


@chika.config
class ModelConfig:
    name: str = chika.choices(*ViTs.choices())
    droppath_rate: float = 0.1
    ema: bool = False


@chika.config
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    epochs: int = 200


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    debug: bool = False
    amp: bool = False

    def __post_init__(self):
        self.optim.lr *= self.data.batch_size * homura.get_world_size() / 512


@chika.main(cfg_cls=Config)
@distributed_ready_main
def main(cfg: Config):
    if homura.is_master():
        import rich
        rich.print(cfg)
    vs = DATASET_REGISTRY("imagenet")
    train_da = vs.default_train_da.copy() + [AutoAugment()] if cfg.data.autoaugment else None
    post_da = [RandomErasing()] if cfg.data.random_erasing else None
    train_loader, test_loader = vs(batch_size=cfg.data.batch_size,
                                   train_da=train_da,
                                   post_norm_train_da=post_da,
                                   train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   num_workers=8)
    model = ViTs(cfg.model.name)(droppath_rate=cfg.model.droppath_rate)
    if cfg.model.ema:
        model = ViTEMA(model, 0.99996)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, 1, 5)

    with ViTTraner(model,
                   None,
                   SmoothedCrossEntropy(cfg.optim.label_smoothing),
                   reporters=[reporters.TensorboardReporter(".")],
                   scheduler=scheduler,
                   use_amp=cfg.amp,
                   use_cuda_nonblocking=True,
                   report_accuracy_topk=5,
                   optim_cfg=cfg.optim
                   ) as trainer:
        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
