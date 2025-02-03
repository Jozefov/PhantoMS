import pandas as pd
from torch.utils.data import Dataset
import torch

class MoleculeTextDataset(Dataset):
    """
    Dataset for molecule strings (SMILES or SELFIES) read from a TSV file.
    The TSV file must have a column named "smiles" containing the molecule strings.
    """
    def __init__(self, tsv_path: str, tokenizer, smiles_column: str = "smiles", max_len: int = 200):
        # Read the TSV file using pandas (assumes tab-delimited)
        df = pd.read_csv(tsv_path, sep="\t")
        # Extract the molecule strings from the specified column.
        self.texts = df[smiles_column].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Encode the text with special tokens (the tokenizer’s post-processor should add SOS/EOS)
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        # Truncate if necessary
        token_ids = token_ids[:self.max_len]
        # For teacher forcing: input = tokens[:-1], target = tokens[1:]
        return {
            "input": torch.tensor(token_ids[:-1], dtype=torch.long),
            "target": torch.tensor(token_ids[1:], dtype=torch.long)
        }