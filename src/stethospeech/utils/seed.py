"""Deterministic seed management for PyTorch, NumPy, and standard library random."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set deterministic seeds across Python random, NumPy, and PyTorch.

    Args:
        seed: Integer random seed (default: 42, matching original research split).
        deterministic: If True, configures CuDNN backends for deterministic execution.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
