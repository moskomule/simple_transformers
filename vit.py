import chika
from homura import distributed_ready_main, get_num_nodes, lr_scheduler, optim, \
    reporters
from homura.modules import SmoothedCrossEntropy
from homura.trainers import SupervisedTrainer
from homura.vision.data import DATASET_REGISTRY
from torchvision.transforms import AutoAugment

from models.vit import ViT


@chika.config
class DataConfig:
    batch_size: int = 128


@chika.config
class ModelConfig:
    ebm_dim: int = 768
    num_heads: int = 12
    attn_dropout_rate: float = 0
    proj_dropout_rate: float = 0
    emb_dropout_rate: float = 0
    droppath_rate: float = 0


@chika.config
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.05
    epochs: int = 200


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    debug: bool = False
    amp: bool = False


@chika.main(cfg_cls=Config)
@distributed_ready_main
def main(cfg: Config):
    vs = DATASET_REGISTRY(cfg.data.name)
    aa = vs.default_train_da.copy() + [AutoAugment()]
    train_loader, test_loader = DATASET_REGISTRY("imagenet")(cfg.batch_size,
                                                             train_da=aa,
                                                             train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                                             test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                                             num_workers=cfg.num_workers)
    model = ViT.construct(cfg.model)
    optimizer = optim.AdamW(lr=cfg.optim.lr * cfg.data.batch_size * get_num_nodes() / 256,
                            weight_decay=cfg.optim.weight_decay, multi_tensor=True)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, 1, 5)

    with SupervisedTrainer(model,
                           optimizer,
                           SmoothedCrossEntropy(),
                           reporters=[reporters.TensorboardReporter(".")],
                           scheduler=scheduler,
                           use_amp=cfg.amp,
                           use_cuda_nonblocking=True,
                           report_accuracy_topk=5) as trainer:
        for epoch in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
