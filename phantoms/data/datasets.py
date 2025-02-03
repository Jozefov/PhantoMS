from torch.utils.data import Dataset
import torch

class MoleculeTextDataset(Dataset):
    """
    Dataset for molecule strings (SMILES or SELFIES).
    """
    def __init__(self, texts, tokenizer, max_len=200):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        token_ids = token_ids[:self.max_len]
        return {
            "input": torch.tensor(token_ids[:-1], dtype=torch.long),
            "target": torch.tensor(token_ids[1:], dtype=torch.long)
        }