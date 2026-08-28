"""
Shared setup for all evaluation scripts: monkey-patches, warning suppression,
device selection, and small filesystem helpers.

Every evaluation script imports this module FIRST, before importing anything
from `enhancement` or `captioning`, for the same reason main.py does: the
installed torchvision version has removed `functional_tensor`, which the
vendored basicsr code still imports.
"""
import sys
import os
import warnings

from torchvision.transforms.v2 import functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def results_path(*parts):
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def set_all_seeds(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
