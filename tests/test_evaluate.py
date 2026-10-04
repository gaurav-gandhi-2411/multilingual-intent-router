from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from intent_router.evaluate import (
    TestEvalRefused,
    aurc,
    bootstrap_metrics_ci,
    ece_bins,
    evaluate_test_once,
    fit_temperature,
    hierarchy_summary,
    mcnemar_exact,
    paired_bootstrap_delta_f1,
    risk_coverage,
    softmax,
    threshold_at_retention,
)
from intent_router.stats import macro_f1


def test_bootstrap_ci_deterministic_and_matches_reference() -> None:
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    p = np.array([0, 1, 1, 1, 2, 0, 0, 1])
    a = bootstrap_metrics_ci(y, p, 3, n_resamples=500, seed=42)
    b = bootstrap_metrics_ci(y, p, 3, n_resamples=500, seed=42)
    c = bootstrap_metrics_ci(y, p, 3, n_resamples=500, seed=7)
    assert a == b
    assert a != c
    # Independent reference: same RNG stream, plain accuracy and sklearn-backed macro-F1.
    idx = np.random.default_rng(42).integers(0, len(y), size=(500, len(y)))
    accs = np.array([np.mean(y[i] == p[i]) for i in idx])
    f1s = np.array([macro_f1(y[i], p[i], 3) for i in idx])
    assert a["accuracy"]["lo"] == pytest.approx(np.quantile(accs, 0.025))
    assert a["accuracy"]["hi"] == pytest.approx(np.quantile(accs, 0.975))
    assert a["macro_f1"]["lo"] == pytest.approx(np.quantile(f1s, 0.025))
    assert a["macro_f1"]["point"] == pytest.approx(macro_f1(y, p, 3))
    assert a["accuracy"]["point"] == pytest.approx(6 / 8)


def test_bootstrap_perfect_predictions_have_degenerate_ci() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    ci = bootstrap_metrics_ci(y, y, 3, n_resamples=200, seed=42)
    assert ci["accuracy"] == {"point": 1.0, "lo": 1.0, "hi": 1.0}


def test_paired_bootstrap_identical_predictions_zero_delta() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 1, 1])
    p = np.array([0, 1, 1, 0, 2, 2, 1, 0])
    d = paired_bootstrap_delta_f1(y, p, p, 3, n_resamples=200, seed=42)
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_mcnemar_exact_hand_example() -> None:
    # 8 discordant pairs: A right in 7, B right in 1 => two-sided p = 2 * P(X <= 1 | n=8)
    a = np.array([1] * 7 + [0] * 1 + [1] * 5, dtype=bool)
    b = np.array([0] * 7 + [1] * 1 + [1] * 5, dtype=bool)
    r = mcnemar_exact(a, b)
    assert (r["a_only_correct"], r["b_only_correct"], r["n_discordant"]) == (7, 1, 8)
    assert r["p_value"] == pytest.approx(2 * (1 + 8) / 256)  # 0.0703125
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_ece_hand_example() -> None:
    # bin (0.9,1.0]: conf .95, acc .5, weight .5 => .225; bin (0.5,0.6]: conf .55, acc 1 => .225
    probs = np.array([[0.95, 0.05], [0.95, 0.05], [0.55, 0.45], [0.55, 0.45]])
    y = np.array([0, 1, 0, 0])
    ece, bins = ece_bins(probs, y, n_bins=10)
    assert ece == pytest.approx(0.45)
    assert sum(b["count"] for b in bins) == 4


def test_temperature_fit_recovers_known_temperature() -> None:
    rng = np.random.default_rng(0)
    n, k, t_true = 20000, 5, 2.5
    z = rng.normal(0, 3.0, size=(n, k))
    p = softmax(z / t_true).astype(np.float64)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    assert fit_temperature(z, y) == pytest.approx(t_true, rel=0.05)


def test_aurc_and_risk_coverage_toy() -> None:
    conf = np.array([0.6, 0.9, 0.7, 0.8])
    correct = np.array([1, 1, 0, 1])
    cov, risk = risk_coverage(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    # order by conf desc: .9(ok) .8(ok) .7(wrong) .6(ok) => risks 0, 0, 1/3, 1/4
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    assert aurc(conf, correct) == pytest.approx((0 + 0 + 1 / 3 + 1 / 4) / 4)


def test_threshold_at_retention_keeps_at_least_target() -> None:
    conf = np.linspace(0.1, 1.0, 40)
    thr = threshold_at_retention(conf, 0.95)
    assert np.mean(conf >= thr) >= 0.95


@pytest.mark.parametrize("n", [20, 74])
def test_threshold_at_retention_keeps_exactly_ceil_95_percent(n: int) -> None:
    conf = np.random.default_rng(42).permutation(np.linspace(0.1, 1.0, n))  # distinct scores
    thr = threshold_at_retention(conf, 0.95)
    assert int((conf >= thr).sum()) == math.ceil(0.95 * n)  # n=20 -> 19, n=74 -> 71


def test_hierarchy_summary_hand_example() -> None:
    labels = ["a", "shipment_information.x", "shipment_information.y", "z"]
    y = np.array([1, 1, 2, 0, 3])
    p = np.array([2, 0, 2, 0, 3])  # one within-family error, one family->outside error
    h = hierarchy_summary(y, p, labels)
    assert h["n_errors"] == 2
    assert h["n_family_gold_errors"] == 2
    assert h["n_family_errors_within_family"] == 1
    assert h["share_family_errors_within_family"] == 0.5
    assert h["parent_accuracy"] == pytest.approx(4 / 5)


def test_test_once_guard_refuses_second_evaluation(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    calls: list[int] = []

    def infer() -> int:
        calls.append(1)
        return 7

    out, m = evaluate_test_once("fp1", "evaluation", infer, log, lambda r: {"v": r}, sha="abc")
    assert out == 7 and m == {"v": 7}
    with pytest.raises(TestEvalRefused):
        evaluate_test_once("fp1", "evaluation", infer, log, lambda r: {"v": r}, sha="abc")
    assert len(calls) == 1  # the refused call must not run inference
    # determinism inference is allowed any number of times, and a new model is a new fingerprint
    evaluate_test_once("fp1", "determinism_inference", infer, log, sha="abc")
    evaluate_test_once("fp2", "evaluation", infer, log, lambda r: {}, sha="abc")
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [(e["model_fingerprint"], e["call_type"]) for e in lines] == [
        ("fp1", "evaluation"),
        ("fp1", "determinism_inference"),
        ("fp2", "evaluation"),
    ]
    assert all({"timestamp", "git_sha"} <= set(e) for e in lines)


def test_test_once_guard_validates_call_type(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        evaluate_test_once("f", "other", lambda: 1, log, sha="x")
    with pytest.raises(ValueError):
        evaluate_test_once("f", "evaluation", lambda: 1, log, None, sha="x")
    with pytest.raises(ValueError):
        evaluate_test_once("f", "determinism_inference", lambda: 1, log, lambda r: {}, sha="x")
    assert not log.exists()
