import torch

from config import *
from models.transformer import TransformerTranslator
from data_utils.tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD TOKENIZER
tokenizer = Tokenizer()
tokenizer.load("data/pl_en.model")  # dostosuj jeśli masz inną nazwę

# 2. LOAD MODEL
model = TransformerTranslator(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN
).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])

model.eval()


# 3. GREEDY DECODING
def translate(text, max_len=MAX_LEN):

    # tokenize input
    src = tokenizer.encode(text)

    src = torch.tensor(src).unsqueeze(0).to(device)

    tgt = [BOS_ID]

    for _ in range(max_len):

        tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src, tgt_tensor)

        next_token = logits[0, -1].argmax().item()

        tgt.append(next_token)

        if next_token == EOS_ID:
            break

    return tokenizer.decode(tgt)


# 4. TEST
while True:
    text = input("\nPL > ")
    out = translate(text)
    print("EN >", out)