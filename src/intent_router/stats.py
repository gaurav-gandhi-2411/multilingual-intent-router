from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 12) -> float:
    """Macro-F1 over all n_classes labels (absent classes score 0, never dropped)."""
    return float(
        f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain accuracy."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def nadeau_bengio_ttest(
    a: np.ndarray, b: np.ndarray, n_train: int, n_test: int
) -> tuple[float, float, int]:
    """Corrected resampled t-test (Nadeau & Bengio 2003) on paired per-fold scores.

    a, b: scores of two configs over the same J = r*k (repetition, fold) pairs.
    d_j = a_j - b_j; t = mean(d) / sqrt((1/J + n_test/n_train) * var(d, ddof=1)); df = J-1.
    Returns (t, two-sided p, df). Identical scores give t=0, p=1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("a and b must be 1-D, equal length, J >= 2")
    d = a - b
    j = d.size
    var = float(np.var(d, ddof=1))
    mean = float(np.mean(d))
    if var == 0.0:
        return (0.0, 1.0, j - 1) if mean == 0.0 else (float(np.sign(mean) * np.inf), 0.0, j - 1)
    t = mean / float(np.sqrt((1.0 / j + n_test / n_train) * var))
    p = float(2.0 * sps.t.sf(abs(t), df=j - 1))
    return float(t), p, j - 1
