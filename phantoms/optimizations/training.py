import random
import torch
import numpy as np


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