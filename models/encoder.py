import torch.nn as nn

from models.attention import MultiHeadAttention
from models.feedforward import FeedForward


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        mask=None
    ):

        attention = self.attention(
            x,
            x,
            x,
            mask
        )

        x = self.norm1(
            x + attention
        )

        ff = self.feed_forward(x)

        x = self.norm2(
            x + ff
        )

        return x


class Encoder(nn.Module):

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

                EncoderLayer(

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
        mask=None
    ):

        for layer in self.layers:

            x = layer(
                x,
                mask
            )

        return x