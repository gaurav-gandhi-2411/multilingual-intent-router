from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("WANDB_MODE", "disabled")
pytest.importorskip("torch")

from intent_router.cv import load_cv_frame  # noqa: E402
from intent_router.train import TrainConfig, train_fold  # noqa: E402


def _run() -> object:
    # Train split only (never val/test): 64 train rows, 32 eval rows, 2 epochs.
    frame = load_cv_frame()
    frame = frame[frame["split"] == "train"].sort_values("id").reset_index(drop=True)
    train_df, eval_df = frame.iloc[:64], frame.iloc[64:96]
    cfg = TrainConfig(
        model_name="FacebookAI/xlm-roberta-base",
        lr=3e-5,
        epochs=2,
        max_len=64,
        precision="fp16",
        model_seed=0,
        wandb_group="test",
    )
    return train_fold(cfg, train_df, eval_df, {"run_id": "determinism_test"})


def test_two_identical_runs_match() -> None:
    """Same seed twice => identical macro-F1 and probabilities.

    Tolerance: atol=1e-6 on probs. We aim for bit-exact; the tolerance only absorbs
    non-deterministic CUDA kernels without a deterministic variant (warn_only mode),
    which can perturb the last float32 bits. Macro-F1 must match exactly.
    """
    a, b = _run(), _run()
    assert [e["macro_f1"] for e in a.epochs] == [e["macro_f1"] for e in b.epochs]
    assert [e["accuracy"] for e in a.epochs] == [e["accuracy"] for e in b.epochs]
    np.testing.assert_allclose(a.probs, b.probs, rtol=0, atol=1e-6)
    assert not a.nan_detected
