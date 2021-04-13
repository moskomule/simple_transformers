# Transformers

## Requirements

```commandline
conda create -n transformer python=3.9
conda activate transformer
conda install -c pytorch -c conda-forge pytorch torchvision cudatoolkit
pip install -U homura-core chika datasets tokenizers rich # fairscale 
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
python vit.py [--amp] [--model.ema]
```

## Acknowledgement

For this project, I learned a lot from Andrej's [minGPT](https://github.com/karpathy/mingpt),
Ross's [timm](https://github.com/rwightman/pytorch-image-models), and
FAIR's [ClassyVision](https://github.com/facebookresearch/ClassyVision).
