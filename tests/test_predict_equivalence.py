"""The lean serving maths equals the training-side reference implementations.

Runs in the TRAINING venv (needs scipy / sklearn / pandas via intent_router.ood); skipped in
`.venv-serve`, where tests/test_predict_contract.py covers the same functions with hand-checked
values. Only numpy-level helpers of the serving modules are imported here (no torch needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from scipy.stats import rankdata, spearmanr  # noqa: E402

from intent_router import ood, serve_bench, stats  # noqa: E402
from intent_router.evaluate import softmax as ref_softmax  # noqa: E402
from intent_router.evaluate import threshold_at_retention  # noqa: E402
from intent_router.predict import MahalanobisBank, mahalanobis_score, softmax  # noqa: E402

NPZ = Path(__file__).resolve().parents[1] / "outputs" / "final" / "features_logits.npz"


def _synthetic(n: int = 120, d: int = 24, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = np.arange(n) % k
    x = rng.normal(size=(n, d)) + 3.0 * np.eye(k, d)[y]
    return x.astype(np.float32), y


def test_mahalanobis_equals_ood_reference_synthetic() -> None:
    x, y = _synthetic()
    g = ood.fit_gaussian_lw(x, y, 4)
    q = np.random.default_rng(0).normal(size=(30, 24)).astype(np.float32) * 2.0
    ref = ood.score_mahalanobis(q, g)
    assert np.abs(mahalanobis_score(q, g.means, g.precision) - ref).max() == 0.0
    assert np.abs(MahalanobisBank(g.means, g.precision).score(q) - ref).max() == 0.0


@pytest.mark.skipif(not NPZ.exists(), reason="saved v1 arrays not present")
def test_mahalanobis_equals_ood_reference_on_saved_v1_arrays() -> None:
    z = np.load(NPZ)
    # Pseudo-labels from the train logits: the score maths does not care how the classes were
    # obtained, and this keeps the test independent of the (confidential) dataset file.
    y = z["train_logits"].argmax(axis=1)
    n_cls = z["train_logits"].shape[1]
    if len(np.unique(y)) != n_cls:
        pytest.skip("train argmax does not cover every class")
    g = ood.fit_gaussian_lw(z["train_features"], y, n_cls)
    for part in ("train", "val"):  # saved arrays only; the test arrays are never touched here
        f = z[f"{part}_features"]
        diff = np.abs(mahalanobis_score(f, g.means, g.precision) - ood.score_mahalanobis(f, g))
        assert diff.max() == 0.0, (part, diff.max())
    thr = threshold_at_retention(mahalanobis_score(z["val_features"], g.means, g.precision), 0.95)
    assert thr == threshold_at_retention(ood.score_mahalanobis(z["val_features"], g), 0.95)


def test_softmax_matches_reference_up_to_float32_cast() -> None:
    lg = np.random.default_rng(1).normal(size=(10, 12)) * 4
    for t in (1.0, 1.0228):
        assert np.abs(softmax(lg, t) - ref_softmax(lg, t)).max() < 1e-7  # ref returns float32


def test_spearman_and_ranks_equal_scipy() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.integers(0, 6, size=40).astype(float)  # many ties
        b = a + rng.normal(size=40) * 2
        assert np.allclose(serve_bench.rank_average(a), rankdata(a))
        assert serve_bench.spearman(a, b) == pytest.approx(spearmanr(a, b)[0], abs=1e-12)


def test_macro_f1_equals_stats_reference() -> None:
    rng = np.random.default_rng(5)
    for _ in range(10):
        y, p = rng.integers(0, 12, size=74), rng.integers(0, 12, size=74)
        assert serve_bench.macro_f1(y, p, 12) == pytest.approx(stats.macro_f1(y, p, 12), abs=1e-12)
