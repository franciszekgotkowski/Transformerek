import sentencepiece as spm

from config import *


class Tokenizer:

    def __init__(self):

        self.sp = spm.SentencePieceProcessor()

        self.sp.load(
            TOKENIZER_MODEL
        )

    def encode(self, text):

        ids = self.sp.encode(
            text,
            out_type=int
        )

        ids = [BOS_ID] + ids + [EOS_ID]

        return ids

    def load(self, path):
        self.sp.load(path)

    def decode(self, ids):

        ids = [
            i
            for i in ids
            if i not in (
                PAD_ID,
                BOS_ID,
                EOS_ID
            )
        ]

        return self.sp.decode(ids)

    def vocab_size(self):

        return self.sp.get_piece_size()