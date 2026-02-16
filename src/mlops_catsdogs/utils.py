import os
import random
from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make results more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy for a batch from raw logits."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y).sum().item()
    return correct / y.size(0)


def split_metrics_init() -> Dict[str, float]:
    """Convenience dict for accumulating metrics."""
    return {"loss_sum": 0.0, "acc_sum": 0.0, "n_batches": 0}