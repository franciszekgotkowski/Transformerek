import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.embeddings import (
    TokenEmbedding,
    PositionalEncoding
)

class TransformerTranslator(nn.Module):

    def __init__(

        self,

        src_vocab,

        tgt_vocab,

        d_model,

        num_heads,

        num_layers,

        d_ff,

        max_len

    ):

        super().__init__()

        self.src_embedding = TokenEmbedding(

            src_vocab,

            d_model

        )

        self.tgt_embedding = TokenEmbedding(

            tgt_vocab,

            d_model

        )

        self.position = PositionalEncoding(

            d_model,

            max_len

        )

        self.encoder = Encoder(

            num_layers,

            d_model,

            num_heads,

            d_ff

        )

        self.decoder = Decoder(

            num_layers,

            d_model,

            num_heads,

            d_ff

        )

        self.output = nn.Linear(

            d_model,

            tgt_vocab

        )


    def forward(

        self,

        src,

        tgt,

        src_mask=None,

        tgt_mask=None

    ):

        src = self.src_embedding(src)

        src = self.position(src)

        encoder_output = self.encoder(

            src,

            src_mask

        )

        tgt = self.tgt_embedding(tgt)

        tgt = self.position(tgt)

        decoder_output = self.decoder(

            tgt,

            encoder_output,

            src_mask,

            tgt_mask

        )

        logits = self.output(

            decoder_output

        )

        return logits