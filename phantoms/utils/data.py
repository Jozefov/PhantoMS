from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from phantoms.utils.constants import *
import re
import os
import numpy as np
import random
import torch


def set_global_seeds(seed: int = 42):
    """
    Set seeds for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic behavior (may slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def smiles_to_formula(smiles: str) -> str:
    """
    Convert SMILES string to molecular formula using RDKit.

    Args:
        smiles (str): SMILES representation of the molecule.

    Returns:
        str: Molecular formula (e.g., 'C6H6').
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return CalcMolFormula(mol)
    else:
        return 'Unknown'

def encode_formula(formula: str) -> torch.Tensor:
        """
        Encode molecular formula into a numerical vector based on element counts.

        Args:
            formula (str): Molecular formula (e.g., 'C6H6').

        Returns:
            torch.Tensor: Encoded formula vector of shape [len(ELEMENTS)].
        """
        # Initialize count dictionary
        counts = {element: 0 for element in ELEMENTS}

        # Parse the formula
        # RDKit's CalcMolFormula returns a string like 'C6H6'
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        for (element, count) in matches:
            if element in counts:
                counts[element] += int(count) if count else 1

        # Create a list in the order of ELEMENTS
        formula_vector = [counts[element] for element in ELEMENTS]
        return torch.tensor(formula_vector, dtype=torch.float32)

def save_embeddings(model, dataloader, save_dir):
    """
    Extract embeddings from the model for the entire dataset and save them.

    Args:
        model: Trained model with a get_embeddings method.
        dataloader: DataLoader for the dataset.
        save_dir: Directory to save the embeddings.
    """
    model.eval()
    embeddings_dict = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            data = batch['spec']
            smiles = batch.get('smiles', None) if model.use_formula else None
            embeddings = model.get_embeddings(data, smiles_batch=smiles)  # Dict of embeddings

            for layer_name, embed in embeddings.items():
                if layer_name not in embeddings_dict:
                    embeddings_dict[layer_name] = []
                embeddings_dict[layer_name].append(embed.numpy())

    # Concatenate all batches for each layer and save
    for layer_name, embed_list in embeddings_dict.items():
        layer_embeddings = np.concatenate(embed_list, axis=0)  # Shape: [total_samples, embedding_dim]
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{layer_name}.npy")
        np.save(save_path, layer_embeddings)
