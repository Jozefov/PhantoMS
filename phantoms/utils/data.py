from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from phantoms.utils.constants import *
import re
import torch

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