r"""Open-set improvement round open-set scoring variants: numpy / scikit-learn, no torch.

Scorers (feature-based, higher = known; all fitted on known TRAIN rows only):
  * ``maha_ft``  the shipped scorer: minus the smallest class Mahalanobis distance (Ledoit-Wolf).
  * ``i1a``      relative Mahalanobis: minus (min class distance - class-agnostic background
                 distance), both Ledoit-Wolf. Distances (not squared distances) like ``maha_ft``
                 (Ren et al. use squared ones; chosen so I1a differs from maha_ft only by the
                 background term).
  * ``i1b``      Mahalanobis on L2-normalised features.
Threshold rules (calibration-known rows only; accept = score >= threshold):
  * global       ``evaluate.threshold_at_retention`` (the shipped rule, ceil rule).
  * ``i6a``      per predicted class, 95% retention; classes with < ``min_rows`` calibration rows
                 fall back to the global threshold (the fallback counts are returned).
  * ``i6b``      one global threshold from 5-fold cross-fitted scores of known train + calibration
                 rows.
Plus ``neutralize_ids``: every ID ``[A-Z]{2,4}-\d+`` becomes ``REF-<digits>`` (same ID pattern as
the A1 randomisation and the robustness swaps, i.e. with a leading word boundary).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from intent_router.evaluate import threshold_at_retention
from intent_router.ood import GaussianModel, fit_gaussian_lw, l2_normalize, score_mahalanobis

SCORERS = ("maha_ft", "i1a", "i1b")
THRESHOLD_RULES = ("global", "i6a", "i6b")
# Same shape as train._ID_RE / robustness._ID_RE (a test pins the equality).
ID_RE = re.compile(r"\b([A-Z]{2,4})-(\d+)")


# ======================================================================== ID neutralisation
def neutralize_ids(text: str) -> str:
    """Rewrite every ID prefix to REF, digits kept (``PO-123`` -> ``REF-123``)."""
    return ID_RE.sub(lambda m: f"REF-{m.group(2)}", text)


# ============================================================================ Gaussian helpers
def fit_gaussian_present(features: np.ndarray, y: np.ndarray, n_classes: int) -> GaussianModel:
    """fit_gaussian_lw over the classes that have rows (absent classes are simply not candidates).

    Cross-fitting and tiny folds can leave a class without rows; ``fit_gaussian_lw`` refuses
    that, so the present classes are re-indexed. At least one row is required.
    """
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("cannot fit a Gaussian on zero rows")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be in [0, {n_classes}): got [{y.min()}, {y.max()}]")
    present = np.unique(y)
    remap = {int(c): i for i, c in enumerate(present)}
    return fit_gaussian_lw(features, np.array([remap[int(c)] for c in y]), len(present))


@dataclass
class RelativeMaha:
    """I1a: class-conditional Gaussian plus the class-agnostic background Gaussian."""

    class_gauss: GaussianModel
    bg_gauss: GaussianModel


def fit_relative_maha(features: np.ndarray, y: np.ndarray, n_classes: int) -> RelativeMaha:
    """Class Gaussian (shared Ledoit-Wolf covariance) + one background Gaussian over all rows."""
    bg = fit_gaussian_present(features, np.zeros(len(y), dtype=int), 1)
    return RelativeMaha(fit_gaussian_present(features, y, n_classes), bg)


def score_relative_maha(features: np.ndarray, model: RelativeMaha) -> np.ndarray:
    """-(min_k D_k(x) - D_bg(x)) = D_bg - D_min; higher = known. D = Mahalanobis distance."""
    return score_mahalanobis(features, model.class_gauss) - score_mahalanobis(
        features, model.bg_gauss
    )


FitFn = Callable[[np.ndarray, np.ndarray, int], Callable[[np.ndarray], np.ndarray]]


def fit_feature_scorer(
    name: str, features: np.ndarray, y: np.ndarray, n_classes: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit scorer `name` on known train rows; returns features -> score (higher = known)."""
    if name == "maha_ft":
        g = fit_gaussian_present(features, y, n_classes)
        return lambda x: score_mahalanobis(x, g)
    if name == "i1a":
        r = fit_relative_maha(features, y, n_classes)
        return lambda x: score_relative_maha(x, r)
    if name == "i1b":
        g2 = fit_gaussian_present(l2_normalize(features), y, n_classes)
        return lambda x: score_mahalanobis(l2_normalize(x), g2)
    raise KeyError(f"unknown scorer {name!r}; expected one of {SCORERS}")


def scorer_fit_fn(name: str) -> FitFn:
    """`fit_feature_scorer` with the scorer name bound (what the cross-fit needs)."""

    def fit(x: np.ndarray, y: np.ndarray, n_classes: int) -> Callable[[np.ndarray], np.ndarray]:
        return fit_feature_scorer(name, x, y, n_classes)

    return fit


# ================================================================== I6a: per-class thresholds
@dataclass(frozen=True)
class PerClassThresholds:
    """Per predicted-class thresholds with the global fallback.

    thresholds covers EVERY class index in range(n_classes); a class with fewer than `min_rows`
    calibration rows (including none) carries the global threshold and is listed in `fallback`.
    """

    thresholds: tuple[float, ...]
    global_threshold: float
    fallback: tuple[int, ...]
    n_cal_per_class: tuple[int, ...]
    min_rows: int
    retention: float
    _arr: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_arr", np.asarray(self.thresholds, dtype=np.float64))

    @property
    def n_fallback(self) -> int:
        """Number of classes that use the global threshold."""
        return len(self.fallback)

    def for_pred(self, pred: np.ndarray) -> np.ndarray:
        """Threshold of each row's predicted class."""
        return self._arr[np.asarray(pred, dtype=int)]


def fit_per_class_thresholds(
    cal_scores: np.ndarray,
    cal_pred: np.ndarray,
    n_classes: int,
    retention: float = 0.95,
    min_rows: int = 5,
) -> PerClassThresholds:
    """95% retention (ceil rule) per PREDICTED class on calibration-known rows.

    The grouping is by the model's prediction because that is all that is known at serving time.
    A class with fewer than `min_rows` calibration rows falls back to the global threshold
    (all calibration rows pooled); at least one calibration row is required overall.
    """
    s = np.asarray(cal_scores, dtype=np.float64)
    p = np.asarray(cal_pred, dtype=int)
    if len(s) == 0 or len(s) != len(p):
        raise ValueError("calibration scores / predictions must be non-empty and the same length")
    t_global = threshold_at_retention(s, retention)
    counts = np.bincount(p, minlength=n_classes)[:n_classes]
    thr: list[float] = []
    fallback: list[int] = []
    for c in range(n_classes):
        if counts[c] >= min_rows:
            thr.append(threshold_at_retention(s[p == c], retention))
        else:
            thr.append(t_global)
            fallback.append(c)
    return PerClassThresholds(
        tuple(thr), t_global, tuple(fallback), tuple(int(x) for x in counts), min_rows, retention
    )


# ================================================================= I6b: cross-fitted threshold
def stratified_fold_ids(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per row: within each class a seeded permutation dealt round-robin (balanced)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        out[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    return out


def crossfit_scores(
    fit_fn: FitFn,
    features: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold scores: each row is scored by a scorer fitted on the other folds' rows."""
    x = np.asarray(features, dtype=np.float64)
    if len(x) < n_folds:
        raise ValueError(f"need at least {n_folds} rows to cross-fit, got {len(x)}")
    fold = stratified_fold_ids(y, n_folds, seed)
    out = np.full(len(x), np.nan)
    for k in range(n_folds):
        te, tr = fold == k, fold != k
        if not te.any():
            continue
        out[te] = fit_fn(x[tr], np.asarray(y)[tr], n_classes)(x[te])
    return out


def crossfit_global_threshold(
    fit_fn: FitFn,
    features: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    retention: float = 0.95,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """I6b: threshold_at_retention of the pooled out-of-fold scores of known train + cal rows.

    Only the Gaussian is cross-fitted; the network weights are the same for every fold (train rows
    are therefore somewhat optimistic, which the DEV measurement - not this function - judges).
    Returns (threshold, oof_scores).
    """
    oof = crossfit_scores(fit_fn, features, y, n_classes, n_folds, seed)
    if not np.isfinite(oof).all():
        raise ValueError("cross-fit left rows unscored")
    return threshold_at_retention(oof, retention), oof


# =================================================================================== margins
def margins(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    """score - threshold; accept <=> margin >= 0 (the sign of a double difference is exact)."""
    return np.asarray(scores, dtype=np.float64) - threshold


def rejection_rate(unknown_margins: Sequence[float] | np.ndarray) -> float:
    """Strict rejection recall: share of unknown rows with margin < 0."""
    m = np.asarray(unknown_margins, dtype=np.float64)
    return float(np.mean(m < 0)) if len(m) else float("nan")
