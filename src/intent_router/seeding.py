from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch and transformers and request deterministic kernels."""
    # Must be set before the first CUDA matmul for deterministic cuBLAS.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    # warn_only: some ops have no deterministic kernel; we log rather than crash.
    torch.use_deterministic_algorithms(True, warn_only=True)
