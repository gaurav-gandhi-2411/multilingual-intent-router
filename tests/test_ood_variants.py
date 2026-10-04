"""Open-set improvement round scoring variants: synthetic arrays only (no dataset text, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from intent_router import ood_variants as ov
from intent_router.evaluate import threshold_at_retention
from intent_router.ood import fit_gaussian_lw, score_mahalanobis


def blobs(n_per: int = 30, k: int = 3, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, d))
    x = np.concatenate([centers[c] + rng.normal(size=(n_per, d)) for c in range(k)])
    y = np.repeat(np.arange(k), n_per)
    return x, y


# ------------------------------------------------------------------------------ neutralisation
def test_neutralize_ids_rewrites_every_prefix_and_keeps_digits() -> None:
    assert ov.neutralize_ids("track PO-123 and LD-45, ref ABCD-9") == (
        "track REF-123 and REF-45, ref REF-9"
    )
    assert ov.neutralize_ids("REF-7 stays") == "REF-7 stays"
    assert ov.neutralize_ids("no ids here, only X-1 and ABCDE-12 and lower-5") == (
        "no ids here, only X-1 and ABCDE-12 and lower-5"
    )  # 1 letter / 5 letters / lowercase are not IDs (word boundary)
    assert ov.neutralize_ids("") == ""
    once = ov.neutralize_ids("PO-1 LD-2")
    assert ov.neutralize_ids(once) == once  # idempotent


def test_id_pattern_equals_the_one_used_by_training_and_swaps() -> None:
    pytest.importorskip("torch")
    from intent_router import robustness as rb
    from intent_router import train

    assert ov.ID_RE.pattern == train._ID_RE.pattern == rb._ID_RE.pattern  # noqa: SLF001


# ------------------------------------------------------------------------------------ scorers
def test_maha_ft_matches_the_shipped_scorer() -> None:
    x, y = blobs()
    ref = score_mahalanobis(x, fit_gaussian_lw(x, y, 3))
    got = ov.fit_feature_scorer("maha_ft", x, y, 3)(x)
    assert np.allclose(got, ref)


def test_relative_maha_is_class_minus_background_distance() -> None:
    x, y = blobs()
    model = ov.fit_relative_maha(x, y, 3)
    got = ov.score_relative_maha(x, model)
    d_cls = -score_mahalanobis(x, model.class_gauss)
    d_bg = -score_mahalanobis(x, fit_gaussian_lw(x, np.zeros(len(x), dtype=int), 1))
    assert np.allclose(got, d_bg - d_cls)
    assert np.isfinite(got).all()


def test_relative_maha_discounts_a_far_away_direction_unlike_plain_maha() -> None:
    x, y = blobs()
    far = np.full((1, x.shape[1]), 50.0)
    plain = ov.fit_feature_scorer("maha_ft", x, y, 3)(far)[0]
    rel = ov.fit_feature_scorer("i1a", x, y, 3)(far)[0]
    assert plain < -10  # far from every class
    assert abs(rel) < abs(plain)  # the background distance is also large, so it largely cancels


def test_l2_scorer_ignores_feature_norm() -> None:
    x, y = blobs(seed=1)
    f = ov.fit_feature_scorer("i1b", x, y, 3)
    assert np.allclose(f(x[:5]), f(x[:5] * 7.0))
    with pytest.raises(KeyError):
        ov.fit_feature_scorer("nope", x, y, 3)


def test_scorers_survive_singular_covariance_and_tiny_classes() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 40))  # 4 distinct rows in 40 dims, repeated: rank 4 << d
    x = np.repeat(base, 3, axis=0)
    y = np.repeat(np.arange(4), 3)
    for name in ov.SCORERS:
        s = ov.fit_feature_scorer(name, x, y, 4)(rng.normal(size=(6, 40)))
        assert np.isfinite(s).all(), name
    # a class with a single row, and a class index with no rows at all
    x2, y2 = blobs(n_per=10, k=2)
    y2 = np.where(np.arange(len(y2)) == 0, 2, y2)  # class 2 has one row; 0 loses a row
    for name in ov.SCORERS:
        assert np.isfinite(ov.fit_feature_scorer(name, x2, y2, 4)(x2)).all(), name  # class 3 absent
    with pytest.raises(ValueError):
        ov.fit_feature_scorer("maha_ft", x2[:0], y2[:0], 4)
    with pytest.raises(ValueError):
        ov.fit_feature_scorer("maha_ft", x2, y2 + 9, 4)  # labels outside the label space


# ------------------------------------------------------------------------------- I6a per class
def test_per_class_thresholds_use_the_ceil_rule_per_predicted_class() -> None:
    scores = np.arange(100, dtype=float)
    pred = np.array([0] * 60 + [1] * 40)
    t = ov.fit_per_class_thresholds(scores, pred, 2, 0.95, 5)
    assert t.thresholds[0] == threshold_at_retention(scores[:60], 0.95)
    assert t.thresholds[1] == threshold_at_retention(scores[60:], 0.95)
    assert t.global_threshold == threshold_at_retention(scores, 0.95)
    assert t.n_fallback == 0 and t.fallback == ()
    assert np.array_equal(
        t.for_pred(np.array([1, 0, 1])), [t.thresholds[1], t.thresholds[0]] * 1 + [t.thresholds[1]]
    )


def test_per_class_thresholds_fall_back_below_five_rows_and_report_counts() -> None:
    scores = np.arange(30, dtype=float)
    pred = np.array([0] * 20 + [1] * 4 + [2] * 6)  # class 1 has 4 rows, class 3 none
    t = ov.fit_per_class_thresholds(scores, pred, 4, 0.95, 5)
    assert t.fallback == (1, 3) and t.n_fallback == 2
    assert t.thresholds[1] == t.global_threshold == t.thresholds[3]
    assert t.thresholds[0] != t.global_threshold or t.thresholds[2] != t.global_threshold
    assert t.n_cal_per_class == (20, 4, 6, 0)
    exactly5 = ov.fit_per_class_thresholds(scores[:5], np.zeros(5, dtype=int), 1, 0.95, 5)
    assert exactly5.fallback == ()  # 5 rows is enough (< 5 falls back)
    with pytest.raises(ValueError):
        ov.fit_per_class_thresholds(np.array([]), np.array([], dtype=int), 2)


def test_per_class_thresholds_keep_95_percent_of_each_class() -> None:
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 3, 200)])
    pred = np.repeat([0, 1], 200)
    t = ov.fit_per_class_thresholds(scores, pred, 2)
    for c in (0, 1):
        assert np.mean(scores[pred == c] >= t.thresholds[c]) >= 0.95


# ---------------------------------------------------------------------------- I6b cross-fitted
def test_stratified_folds_are_balanced_and_deterministic() -> None:
    y = np.repeat([0, 1, 2], [10, 7, 3])
    f1, f2 = ov.stratified_fold_ids(y, 5, 42), ov.stratified_fold_ids(y, 5, 42)
    assert np.array_equal(f1, f2)
    for c, n in ((0, 10), (1, 7), (2, 3)):
        counts = np.bincount(f1[y == c], minlength=5)
        assert counts.max() - counts.min() <= 1 and counts.sum() == n
    assert not np.array_equal(f1, ov.stratified_fold_ids(y, 5, 43))


def test_crossfit_scores_never_use_a_rows_own_fold() -> None:
    x, y = blobs(n_per=12)
    seen: list[int] = []

    def spy(xtr: np.ndarray, ytr: np.ndarray, k: int):
        seen.append(len(xtr))
        return ov.fit_feature_scorer("maha_ft", xtr, ytr, k)

    oof = ov.crossfit_scores(spy, x, y, 3, 5, 42)
    assert len(seen) == 5 and all(n < len(x) for n in seen)  # each fit excludes its own fold
    assert np.isfinite(oof).all() and len(oof) == len(x)


def test_crossfit_threshold_matches_the_ceil_rule_on_its_own_oof_scores() -> None:
    x, y = blobs(n_per=20)
    thr, oof = ov.crossfit_global_threshold(ov.scorer_fit_fn("maha_ft"), x, y, 3, 5, 0.95, 42)
    assert thr == threshold_at_retention(oof, 0.95)
    assert np.mean(oof >= thr) >= 0.95
    thr2, _ = ov.crossfit_global_threshold(ov.scorer_fit_fn("maha_ft"), x, y, 3, 5, 0.95, 42)
    assert thr == thr2


def test_crossfit_tiny_classes_and_too_few_rows() -> None:
    x, y = blobs(n_per=2, k=3)  # 2 rows per class, 5 folds: some folds hold-out a whole class
    oof = ov.crossfit_scores(ov.scorer_fit_fn("maha_ft"), x, y, 3, 5, 42)
    assert np.isfinite(oof).all()
    with pytest.raises(ValueError, match="at least"):
        ov.crossfit_scores(ov.scorer_fit_fn("maha_ft"), x[:3], y[:3], 3, 5, 42)


# ------------------------------------------------------------------------------------ margins
def test_margin_sign_is_the_accept_rule() -> None:
    s = np.array([1.0, 2.0, 3.0])
    m = ov.margins(s, 2.0)
    assert np.array_equal(m >= 0, s >= 2.0)  # a score equal to the threshold is accepted
    assert ov.rejection_rate(m) == pytest.approx(1 / 3)
    assert np.isnan(ov.rejection_rate([]))
