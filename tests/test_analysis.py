from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intent_router.analysis import (
    load_oof_mean,
    shipment_family_hypothesis,
    slice_masks,
    slice_metrics,
    top_confusions,
)
from intent_router.train import TrainConfig

LABELS = ["a", "shipment_information.x", "shipment_information.y", "b"]


def test_load_oof_mean_averages_and_checks_counts(tmp_path: Path) -> None:
    rows = []
    for rep in range(2):
        rows.append(["i1", rep, 0, 0, 0, 0.9 - 0.2 * rep, 0.1 + 0.2 * rep, 0, 0])
        rows.append(["i2", rep, 0, 0, 1, 0.1, 0.1, 0.8, 0])
    cols = ["id", "fold_seed", "model_seed", "fold", "gold", "prob_0", "prob_1", "prob_2", "prob_3"]
    path = tmp_path / "oof.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    out = load_oof_mean(path, 4, 2)
    assert out["id"].tolist() == ["i1", "i2"]
    assert out.loc[0, "prob_0"] == pytest.approx(0.8) and out.loc[0, "pred"] == 0
    assert out.loc[1, "pred"] == 2
    with pytest.raises(ValueError):
        load_oof_mean(path, 4, 3)


def test_slice_metrics_flags_and_single_class_uses_accuracy() -> None:
    sub = pd.DataFrame({"gold": [1, 1, 1], "pred": [1, 1, 0]})
    m = slice_metrics(sub, 4, 200, 42)
    assert m["indicative_only"] and m["macro_f1"] is None
    assert m["primary_metric"] == "accuracy"
    assert m["accuracy"]["point"] == pytest.approx(2 / 3)


def test_slice_masks_definitions() -> None:
    df = pd.DataFrame(
        {
            "gold": [0, 1, 2, 3, 0, 0],
            "lang": ["en", "es", "undetermined", "en", "es", "es"],
            "code_mixed": [False, False, False, True, False, False],
            "is_noisy": [True, False, False, False, False, False],
            "n_chars": [10, 50, 29, 30, 5, 99],
        }
    )
    m = slice_masks(df, LABELS)
    assert m["non_english_defA"].tolist() == [False, True, False, False, True, True]
    assert m["english"].sum() == 2 and m["undetermined_lang"].sum() == 1
    assert m["short_lt30_chars"].tolist() == [True, False, True, False, True, False]
    assert m["shipment_family"].tolist() == [False, True, True, False, False, False]
    assert "lang_es" not in m  # n = 3 < 5 rows: no per-language slice


def test_family_hypothesis_and_top_confusions_hand_example() -> None:
    # 10 rows: 4 family (gold 1/2), 6 other; 3 errors, all family: 2 within, 1 cross.
    gold = np.array([1, 1, 2, 2, 0, 0, 0, 3, 3, 3])
    pred = np.array([2, 2, 0, 2, 0, 0, 0, 3, 3, 3])
    df = pd.DataFrame({"id": [f"r{i}" for i in range(10)], "gold": gold, "pred": pred})
    df["conf"] = 0.5
    h = shipment_family_hypothesis(df, LABELS)
    assert h["n_errors"] == 3 and h["family_errors"] == 3
    assert h["family_share_of_rows"] == pytest.approx(0.4)
    assert h["family_errors_within_family"] == 2 and h["family_errors_cross_family"] == 1
    assert h["binomial_p_greater"] == pytest.approx(0.4**3)
    top = top_confusions(df, LABELS, k=2, n_examples=3)
    assert top[0]["count"] == 2 and top[0]["gold"] == "shipment_information.x"
    assert top[0]["example_ids"] == ["r0", "r1"]


def test_train_config_new_fields_default_to_old_behaviour() -> None:
    cfg = TrainConfig(model_name="m", lr=1e-5)
    assert cfg.attn_implementation is None and cfg.stop_epoch is None
    with pytest.raises(ValueError):
        TrainConfig(model_name="m", lr=1e-5, epochs=20, stop_epoch=21)
    with pytest.raises(ValueError):
        TrainConfig(model_name="m", lr=1e-5, attn_implementation="flash")
    assert TrainConfig.from_dict({"model_name": "m", "lr": 1, "stop_epoch": 9}).stop_epoch == 9


def test_slice_macro_f1_fixed_gold_label_set() -> None:
    from sklearn.metrics import f1_score

    # gold has classes {0,1}; the out-of-slice prediction 3 must not add a (0-F1) class
    sub = pd.DataFrame({"gold": [0, 0, 1, 1], "pred": [0, 3, 1, 1]})
    m = slice_metrics(sub, 4, 200, 42)
    expected = f1_score(sub["gold"], sub["pred"], labels=[0, 1], average="macro", zero_division=0)
    assert m["macro_f1"]["point"] == pytest.approx(expected)
    assert m["macro_f1"]["point"] > f1_score(sub["gold"], sub["pred"], average="macro") + 0.1


def test_bootstrap_label_set_constant_across_resamples() -> None:
    from intent_router.evaluate import _f1_from_confusion, _resample_confusions

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([0, 0, 0, 0, 0, 0, 3, 1])
    _, confs = _resample_confusions(y, p, 4, 300, 42)
    fixed = np.unique(y)
    # a resample may drop the rare class 1 or draw the out-of-set prediction; the denominator
    # must stay len(fixed) = 2 regardless, so class 3 never contributes
    for c in confs:
        v = _f1_from_confusion(c, False, fixed)
        f1 = [2 * c[i, i] / (c[i].sum() + c[:, i].sum()) if c[i].sum() + c[:, i].sum() else 0
              for i in fixed]  # fmt: skip
        assert v == pytest.approx(sum(f1) / 2)


def test_selective_macro_f1_uses_accepted_gold_classes() -> None:
    from intent_router.evaluate import selective_at_threshold

    conf = np.array([0.9, 0.9, 0.9, 0.1])
    out = selective_at_threshold(conf, np.array([0, 0, 1, 1]), np.array([0, 3, 1, 0]), 0.5, 4)
    # accepted: gold [0,0,1], pred [0,3,1]; classes {0,1}: F1_0 = 2/3, F1_1 = 1 -> 5/6
    assert out["macro_f1_present"] == pytest.approx(5 / 6)


def test_per_fold_run_hypothesis_counts_and_ci() -> None:
    from intent_router.analysis import per_fold_run_hypothesis

    rows = []
    # (gold, wrong predictions out of 4): id0 family gold 1 x2 wrong, id3 non-family gold 3 x1
    for i, (g, n_wrong) in enumerate([(1, 2), (2, 0), (0, 0), (3, 1)]):
        for r in range(4):
            wrong_pred = 0 if g != 0 else 3
            rows.append([f"i{i}", g, wrong_pred if r < n_wrong else g])
    raw = pd.DataFrame(rows, columns=["id", "gold", "pred"])
    out = per_fold_run_hypothesis(raw, LABELS, 200, 42, 3)
    assert out["n_unique_ids"] == 4 and out["n_predictions_per_id"] == 4
    assert out["n_errors"] == 3 and out["family_errors"] == 2
    lo, hi = out["cluster_bootstrap"]["family_share_of_errors_ci95"]
    assert 0.0 <= lo <= 2 / 3 <= hi <= 1.0
