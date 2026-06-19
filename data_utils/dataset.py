from torch.utils.data import Dataset

from data_utils.tokenizer import Tokenizer


class TranslationDataset(Dataset):

    def __init__(self, hf_dataset):

        self.data = hf_dataset

        self.tokenizer = Tokenizer()

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        pl = sample["translation"]["pl"]

        en = sample["translation"]["en"]

        src = self.tokenizer.encode(pl)

        tgt = self.tokenizer.encode(en)



        return src, tgt