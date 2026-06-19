#cos moze byc nie tak

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset

from config import *

from data_utils.dataset import TranslationDataset
from data_utils.collate import collate_fn

from models.transformer import TransformerTranslator
from models.masks import create_causal_mask

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

dataset = load_dataset("Helsinki-NLP/opus-100", "en-pl")

#train_data = TranslationDataset(dataset["train"])
train_split = dataset["train"].select(range(200000))  # zamiast 1M

train_data = TranslationDataset(train_split)

loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

model = TransformerTranslator(

    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,

    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN

).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


import os

start_epoch = 0

if os.path.exists(CHECKPOINT):

    print("Ładowanie checkpointu...")

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1



for epoch in range(start_epoch, EPOCHS):

    model.train()

    loop = tqdm(loader, desc=f"Epoch {epoch}")

    total_loss = 0

    for src, tgt in loop:

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        tgt_mask = create_causal_mask(
            tgt_input.size(1)
        ).to(device)

        logits = model(
            src,
            tgt_input,
            src_mask=None,
            tgt_mask=tgt_mask
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loop.n % 200 == 0:
            print(f"\n💓 Batch {loop.n} alive | loss: {loss.item():.4f}")

        total_loss += loss.item()

        #loop.set_postfix(loss=loss.item())
        loop.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    print(
        f"Epoch {epoch} | loss: {total_loss:.4f}"
    )


    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        CHECKPOINT
    )


print("Trening zakończony")
