import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.norm3 = nn.LayerNorm(
            d_model
        )

    def forward(

        self,

        x,

        encoder_output,

        src_mask=None,

        tgt_mask=None

    ):

        self_att = self.self_attention(

            x,

            x,

            x,

            tgt_mask

        )

        x = self.norm1(

            x + self_att

        )

        cross = self.cross_attention(

            x,

            encoder_output,

            encoder_output,

            src_mask

        )

        x = self.norm2(

            x + cross

        )

        ff = self.feed_forward(x)

        x = self.norm3(

            x + ff

        )

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff
    ):

        super().__init__()

        self.layers = nn.ModuleList(

            [

                DecoderLayer(

                    d_model,

                    num_heads,

                    d_ff

                )

                for _ in range(num_layers)

            ]

        )

    def forward(

        self,

        x,

        encoder_output,

        src_mask=None,

        tgt_mask=None

    ):

        for layer in self.layers:

            x = layer(

                x,

                encoder_output,

                src_mask,

                tgt_mask

            )

        return x