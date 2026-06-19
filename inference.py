import torch

from config import *

from data_utils.tokenizer import Tokenizer

from models.transformer import TransformerTranslator
from models.masks import create_causal_mask


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

tokenizer = Tokenizer()

model = TransformerTranslator(

    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,

    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN

).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint["model"])

model.eval()

def encode(text):

    ids = tokenizer.encode(text)

    return torch.tensor(
        ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

def translate(sentence, max_len=50):

    src = encode(sentence)

    generated = [BOS_ID]

    for _ in range(max_len):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long
        ).to(device)

        tgt_mask = create_causal_mask(
            tgt.size(1)
        ).to(device)

        with torch.no_grad():

            logits = model(
                src,
                tgt,
                src_mask=None,
                tgt_mask=tgt_mask
            )

        next_token = logits[0, -1].argmax().item()

        generated.append(next_token)

        if next_token == EOS_ID:
            break

    return tokenizer.decode(generated)

if __name__ == "__main__":

    print("Translator PL → EN")

    while True:

        text = input("PL > ")

        if text == "exit":
            break

        print(
            "EN >",
            translate(text)
        )

