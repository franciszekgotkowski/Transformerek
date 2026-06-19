import torch

def create_padding_mask(x):

    mask = (x != 0)

    mask = mask.unsqueeze(1)

    mask = mask.unsqueeze(2)

    return mask


def create_causal_mask(size):

    mask = torch.tril(

        torch.ones(

            size,

            size

        )

    )

    mask = mask.unsqueeze(0)

    mask = mask.unsqueeze(0)

    return mask