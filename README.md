# Which Transformer architecture should I use?

## Requirements

```commandline
conda create -n transformer python=3.8
conda activate transformer
conda install -c pytorch pytorch torchvision cudatoolkit=11.0
pip install -U homura-core chika datasets tokenizers
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

Work in progress

## Acknowledgement

For this project, I learned a lot from Andrej's [minGPT](https://github.com/karpathy/mingpt).
