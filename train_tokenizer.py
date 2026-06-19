from datasets import load_dataset

import sentencepiece as spm

from config import *


print("Ładowanie datasetu...")

dataset = load_dataset("opus100", "en-pl")


print("Zapisywanie tekstu...")


with open(
    TOKENIZER_TEXT,
    "w",
    encoding="utf8"
) as f:

    for sample in dataset["train"]:

        pl = sample["translation"]["pl"]

        en = sample["translation"]["en"]

        f.write(pl + "\n")

        f.write(en + "\n")


print("Trenowanie SentencePiece...")


spm.SentencePieceTrainer.train(

    input=TOKENIZER_TEXT,

    model_prefix="data/pl_en",

    vocab_size=VOCAB_SIZE,

    model_type="bpe",

    pad_id=PAD_ID,

    unk_id=UNK_ID,

    bos_id=BOS_ID,

    eos_id=EOS_ID

)


print("Gotowe.")