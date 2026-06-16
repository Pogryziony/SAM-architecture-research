"""Deterministic seeding so experiments are reproducible."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed python, numpy and torch RNGs.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Best-effort determinism; we do not force it (some ops lack
            # deterministic kernels) because this is a research POC.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:  # torch optional for pure-data tooling
        pass
    return seed
