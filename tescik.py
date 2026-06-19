from data_utils.tokenizer import Tokenizer

tok = Tokenizer()

ids = tok.encode(
    "Lubię programować."
)

print(ids)

print(
    tok.decode(ids)
)