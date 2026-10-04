"""Open-set improvement round: score-table "views" of saved Track B arrays (CPU only, no model, no
dataset text).

A weight set's LOCO run saves one npz (train/cal/eval-known/eval-unknown logits + penultimate
features, plus ``n_``-prefixed arrays of the same rows scored on ID-neutralised text). A *view* is
one open-set scoring variant applied to those arrays: feature scorer (``maha_ft`` / ``i1a`` /
``i1b``) x threshold rule (``global`` / ``i6a`` / ``i6b``) x mode (``raw`` / ``neutral``). The
view table has one row per cal / eval row with the score, the margin to the candidate's own
threshold at 95% and 90% retention (accept <=> margin >= 0) and a sidecar json with the
thresholds, so every downstream metric is a pure function of the saved files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intent_router import ood_variants as ov
from intent_router.evaluate import threshold_at_retention
from intent_router.ood import HoldoutSets

MODES = ("raw", "neutral")
RETENTIONS = (0.95, 0.90)
PARTS = ("cal", "ek", "unk")  # npz row groups scored in a view (train rows only fit the scorer)


def local_gold(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """Index of each row's label in the model's label space."""
    return frame["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()


def view_dir(results: Path, name: str, seed: int, mode: str) -> Path:
    """results/<view name>/s<seed>/views/<mode>/"""
    return results / name / f"s{seed}" / "views" / mode


def build_view(
    sets: HoldoutSets,
    arrays: Mapping[str, np.ndarray],
    scorer: str,
    thr_rule: str,
    mode: str,
    min_rows: int = 5,
    crossfit_folds: int = 5,
    crossfit_seed: int = 42,
    fit: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """(score table, threshold info) of one view of one run (see the module docstring).

    `fit` = the already fitted feature scorer (same scorer, same train rows) to skip a refit when
    several modes of one run are built; omitted => fitted here."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if thr_rule not in ov.THRESHOLD_RULES:
        raise KeyError(f"unknown threshold rule {thr_rule!r}; expected one of {ov.THRESHOLD_RULES}")
    pre = "n_" if mode == "neutral" else ""
    labels, n_cls = sets.labels, len(sets.labels)
    y_tr, y_cal = local_gold(sets.train, labels), local_gold(sets.cal, labels)
    if fit is None:
        fit = ov.fit_feature_scorer(scorer, arrays["train_features"], y_tr, n_cls)
    frames = {"cal": sets.cal, "ek": sets.eval_known, "unk": sets.eval_unknown}
    score = {p: fit(arrays[f"{pre}{p}_features"]) for p in PARTS}
    pred = {p: arrays[f"{pre}{p}_logits"].argmax(1) for p in PARTS}

    oof: np.ndarray | None = None
    if thr_rule == "i6b":  # one cross-fit serves both retention levels
        feats = np.concatenate([arrays["train_features"], arrays[f"{pre}cal_features"]])
        oof = ov.crossfit_scores(
            ov.scorer_fit_fn(scorer),
            feats,
            np.concatenate([y_tr, y_cal]),
            n_cls,
            crossfit_folds,
            crossfit_seed,
        )
    info: dict[str, Any] = {
        "scorer": scorer,
        "thr_rule": thr_rule,
        "mode": mode,
        "per_retention": {},
    }
    thr_rows: dict[float, dict[str, np.ndarray]] = {}
    for ret in RETENTIONS:
        t_global = threshold_at_retention(score["cal"], ret)
        rec: dict[str, Any] = {"global": float(t_global)}
        if thr_rule == "global":
            vec = {p: np.full(len(score[p]), t_global) for p in PARTS}
        elif thr_rule == "i6a":
            pc = ov.fit_per_class_thresholds(score["cal"], pred["cal"], n_cls, ret, min_rows)
            vec = {p: pc.for_pred(pred[p]) for p in PARTS}
            rec |= {
                "per_class": {labels[c]: float(t) for c, t in enumerate(pc.thresholds)},
                "fallback_classes": [labels[c] for c in pc.fallback],
                "n_fallback_classes": pc.n_fallback,
                "n_cal_per_predicted_class": {
                    labels[c]: n for c, n in enumerate(pc.n_cal_per_class)
                },
                "min_rows": min_rows,
            }
        else:
            assert oof is not None
            t_cf = threshold_at_retention(oof, ret)
            vec = {p: np.full(len(score[p]), t_cf) for p in PARTS}
            rec |= {
                "crossfit_threshold": float(t_cf),
                "n_crossfit_rows": int(len(oof)),
                "folds": crossfit_folds,
                "seed": crossfit_seed,
            }
        thr_rows[ret] = vec
        info["per_retention"][f"{round(ret * 100)}"] = rec
    out = []
    for p, is_unknown, set_name in (
        ("cal", False, "cal"),
        ("ek", False, "eval"),
        ("unk", True, "eval"),
    ):
        fr = frames[p]
        t = pd.DataFrame(
            {
                "id": fr["id"].to_numpy(),
                "split": fr["split"].to_numpy(),
                "set": set_name,
                "is_unknown": is_unknown,
                "gold": fr["label"].to_numpy(),
                "pred": [labels[i] for i in pred[p]],
                "score": score[p],
            }
        )
        for ret in RETENTIONS:
            t[f"margin{round(ret * 100)}"] = ov.margins(score[p], thr_rows[ret][p])
        out.append(t)
    return pd.concat(out, ignore_index=True), info
