# Simple Transformers

Simple transformer implementations that I can understand.

## Requirements

```commandline
conda create -n transformer python=3.9
conda activate transformer
conda install -c pytorch -c conda-forge pytorch torchvision cudatoolkit=11.1
pip install -U homura-core chika rich
# for NLP also install
pip install -U datasets tokenizers
# To use checkpointing
pip install -U fairscale
# To accelerate (probably only slightly), 
pip install -U opt_einsum
```

## Examples

### Language Modeling

#### GPT

Train GPT-like models on wikitext or GigaText. Currently, Transformer blocks of improved pre LN, pre LN, and post LN are
available for comparison.

```commandline
python gpt.py [--model.block {ipre_ln, pre_ln, post_ln}] [--amp]
```

#### Bert

Work in progress

### Image Recognition

Train ImageNet classification models.

Currently, ViT, and CaiT are implemented.

```commandline
# single process training
python vit.py [--amp] [--model.ema]
# for multi-process training,
python -m torch.distributed.launch --nproc_per_node=2 vit.py ...
```

## Acknowledgement

For this project, I learned a lot from Andrej's [minGPT](https://github.com/karpathy/mingpt),
Ross's [timm](https://github.com/rwightman/pytorch-image-models), and
FAIR's [ClassyVision](https://github.com/facebookresearch/ClassyVision).
