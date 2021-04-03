import pathlib
from os import environ

import datasets
from datasets.utils.logging import set_verbosity_error
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
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
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.enable_truncation(max_length=max_len)
        tokenizer.enable_padding(length=max_len)
        trainer = trainers.BpeTrainer(min_frequency=2, special_tokens=["<eos>", "<pad>", "<unk>", ])

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
