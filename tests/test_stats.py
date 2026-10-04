from __future__ import annotations

import numpy as np
import pytest

from intent_router.stats import macro_f1, nadeau_bengio_ttest


def test_nadeau_bengio_hand_computed() -> None:
    # d = [0.02, 0.04, 0.00, 0.02] -> mean 0.02, var(ddof=1) = (0+4e-4+4e-4+0)/3 = 8/3 * 1e-4
    # J=4, n_test/n_train = 0.25 -> denom = sqrt((0.25+0.25) * 8/3e-4) = sqrt(1.3333e-4)
    # t = 0.02 / 0.0115470 = 1.7320508 (= sqrt(3)); df = 3; two-sided p for t=sqrt(3), df=3
    # is closed-form 0.5 - 1/pi = 0.181690 (t cdf, df=3)
    a = np.array([0.82, 0.84, 0.80, 0.82])
    b = np.array([0.80, 0.80, 0.80, 0.80])
    t, p, df = nadeau_bengio_ttest(a, b, n_train=80, n_test=20)
    assert df == 3
    assert t == pytest.approx(np.sqrt(3.0), rel=1e-9)
    assert p == pytest.approx(0.5 - 1 / np.pi, abs=1e-9)


def test_identical_scores_give_p_one() -> None:
    x = np.array([0.5, 0.6, 0.7])
    assert nadeau_bengio_ttest(x, x, 80, 20) == (0.0, 1.0, 2)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        nadeau_bengio_ttest(np.array([1.0]), np.array([1.0]), 80, 20)


def test_macro_f1_counts_absent_classes_as_zero() -> None:
    y = np.array([0, 0, 1, 1])
    # perfect on 2 of 12 classes; the other 10 labels are absent => F1 0 each
    assert macro_f1(y, y, n_classes=12) == pytest.approx(2 / 12)
    assert macro_f1(y, y, n_classes=2) == 1.0
