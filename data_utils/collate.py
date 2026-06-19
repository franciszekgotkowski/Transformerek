import torch

from config import *


def collate_fn(batch):

    src_batch = []

    tgt_batch = []

    for src, tgt in batch:
        src = src[:MAX_LEN - 2]
        tgt = tgt[:MAX_LEN - 2]

        src = [BOS_ID] + src + [EOS_ID]
        tgt = [BOS_ID] + tgt + [EOS_ID]

        src = src[:MAX_LEN]
        tgt = tgt[:MAX_LEN]

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_len = min(max(len(x) for x in src_batch), MAX_LEN)
    tgt_len = min(max(len(x) for x in tgt_batch), MAX_LEN)

    src_out = []

    tgt_out = []

    for src in src_batch:

        src = src + [
            PAD_ID
        ] * (src_len - len(src))

        src_out.append(src)

    for tgt in tgt_batch:

        tgt = tgt + [
            PAD_ID
        ] * (tgt_len - len(tgt))

        tgt_out.append(tgt)

    src = torch.tensor(
        src_out,
        dtype=torch.long
    )

    tgt = torch.tensor(
        tgt_out,
        dtype=torch.long
    )

    return src, tgt