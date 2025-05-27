import os
import random
import numpy as np
import torch


def set_global_seeds(seed, cuda_deterministic=False):
    """
    Set seeds for reproducibility.
    
    Args:
        seed (int): Random seed
        cuda_deterministic (bool): Whether to use deterministic CUDA
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed) # for replay buffer sampling
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic: # Slower, but good for reproducibility
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False