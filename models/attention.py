import math

import torch

import torch.nn as nn

import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model

        self.num_heads = num_heads

        self.d_head = d_model // num_heads


        self.Wq = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.Wk = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.Wv = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.Wo = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def split_heads(self, x):

        batch_size = x.size(0)

        seq_len = x.size(1)

        x = x.view(

            batch_size,

            seq_len,

            self.num_heads,

            self.d_head

        )

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):

        batch = x.size(0)

        seq_len = x.size(2)

        x = x.transpose(1, 2)

        x = x.contiguous()

        x = x.view(

            batch,

            seq_len,

            self.d_model

        )

        return x

    def forward(

        self,

        query,

        key,

        value,

        mask=None

    ):

        Q = self.Wq(query)

        K = self.Wk(key)

        V = self.Wv(value)

        Q = self.split_heads(Q)

        K = self.split_heads(K)

        V = self.split_heads(V)

        scores = torch.matmul(

            Q,

            K.transpose(-2, -1)

        )

        scores = scores / math.sqrt(self.d_head)

        if mask is not None:

            scores = scores.masked_fill(

                mask == 0,

                -1e9

            )

        attention = F.softmax(

            scores,

            dim=-1

        )

        output = torch.matmul(

            attention,

            V

        )

        output = self.combine_heads(output)

        output = self.Wo(output)

        return output