from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score

from intent_router.evaluate import (
    aurc,
    fit_temperature,
    macro_f1_present,
    softmax,
    threshold_at_retention,
)
from intent_router.stats import macro_f1

# Track B convention everywhere in this module: a HIGHER score means "known / in-domain".
SAFE_LABELS = ("other", "chitchat")  # lenient mode counts routing to these as a safe outcome
NON_LOCO_LABELS = ("other", "chitchat")  # not "in-domain" classes: never held out in LOCO
BASE_METHODS = ("msp", "msp_temp", "max_logit", "neg_energy")


def method_names(knn_ks: tuple[int, ...] = (1, 5), frozen: bool = True) -> list[str]:
    """Ordered method ids: logit-based, fine-tuned-feature (_ft), frozen-feature (_frozen)."""
    names = [*BASE_METHODS, "maha_ft", *[f"knn{k}_ft" for k in knn_ks]]
    if frozen:
        names += ["maha_frozen", *[f"knn{k}_frozen" for k in knn_ks]]
    return names


# --------------------------------------------------------------------- holdout data
@dataclass
class HoldoutSets:
    """Row frames (id, text, label, split, ...) for one Track B holdout run."""

    holdout: list[str]
    labels: list[str]  # the model's label space: remaining classes, sorted; index = position
    train: pd.DataFrame
    cal: pd.DataFrame  # known-only calibration rows (val split minus held-out classes)
    eval_known: pd.DataFrame  # test split minus held-out classes (smoke: == cal)
    eval_unknown: pd.DataFrame  # ALL rows of the held-out classes (smoke: val/train only)
    audit: dict[str, Any]


def _ids_sha(frame: pd.DataFrame) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(frame["id"].astype(str))).encode()).hexdigest()


def build_holdout_sets(df: pd.DataFrame, holdout: list[str], smoke: bool = False) -> HoldoutSets:
    """Split a labelled frame (id, label, split) into train / cal / eval-known / eval-unknown.

    train = train split minus held-out; cal = val split minus held-out; eval known = test split
    minus held-out; eval unknown = every row of the held-out classes from every split. Raises
    AssertionError if any held-out row reaches train or cal, if sets overlap illegitimately, or
    if the label space is not exactly the remaining classes. smoke=True drops the test split
    entirely (no test rows are ever touched) and evaluates the known side on the cal rows.
    """
    all_labels = sorted(df["label"].unique())
    held = sorted(set(holdout))
    if not held or len(held) != len(holdout):
        raise ValueError(f"holdout must be non-empty and duplicate-free: {holdout}")
    missing = set(held) - set(all_labels)
    if missing:
        raise ValueError(f"unknown holdout classes: {sorted(missing)}")
    labels = [lab for lab in all_labels if lab not in held]
    if smoke:
        df = df[df["split"] != "test"]
    is_held = df["label"].isin(held)
    train = df[(df["split"] == "train") & ~is_held].copy()
    cal = df[(df["split"] == "val") & ~is_held].copy()
    unknown = df[is_held].copy()
    eval_known = cal.copy() if smoke else df[(df["split"] == "test") & ~is_held].copy()

    checks = {
        "no_heldout_rows_in_train": not train["label"].isin(held).any(),
        "no_heldout_rows_in_cal": not cal["label"].isin(held).any(),
        "no_heldout_rows_in_eval_known": not eval_known["label"].isin(held).any(),
        "eval_unknown_only_heldout": bool(unknown["label"].isin(held).all()),
        "eval_unknown_has_every_heldout_row": len(unknown) == int(is_held.sum()),
        "label_space_is_remaining_classes": set(labels) == set(all_labels) - set(held)
        and len(labels) == len(all_labels) - len(held),
        "every_remaining_class_in_train": set(train["label"]) == set(labels),
        "train_cal_disjoint": not (set(train["id"]) & set(cal["id"])),
        "train_eval_known_disjoint": not (set(train["id"]) & set(eval_known["id"])),
        "train_unknown_disjoint": not (set(train["id"]) & set(unknown["id"])),
        "cal_unknown_disjoint": not (set(cal["id"]) & set(unknown["id"])),
        "no_test_rows_in_train_or_cal": not (
            (train["split"] == "test").any() or (cal["split"] == "test").any()
        ),
    }
    if not smoke:
        checks["eval_known_is_test_split_only"] = bool((eval_known["split"] == "test").all())
        checks["cal_is_val_split_only"] = bool((cal["split"] == "val").all())
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"holdout leakage / integrity checks failed: {bad}")
    audit = {
        "holdout": held,
        "n_labels": len(labels),
        "n_train": len(train),
        "n_cal": len(cal),
        "n_eval_known": len(eval_known),
        "n_eval_unknown": len(unknown),
        "assertions": checks,
        "ids_sha256": {
            "train": _ids_sha(train),
            "cal": _ids_sha(cal),
            "eval_known": _ids_sha(eval_known),
            "eval_unknown": _ids_sha(unknown),
        },
        "smoke": smoke,
    }
    key = ["id"]
    return HoldoutSets(
        held,
        labels,
        train.sort_values(key).reset_index(drop=True),
        cal.sort_values(key).reset_index(drop=True),
        eval_known.sort_values(key).reset_index(drop=True),
        unknown.sort_values(key).reset_index(drop=True),
        audit,
    )


# ------------------------------------------------------------------------ scores
def score_msp(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability."""
    return softmax(logits).max(axis=1).astype(np.float64)


def score_msp_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Maximum softmax probability of logits / T."""
    return softmax(logits, temperature).max(axis=1).astype(np.float64)


def score_max_logit(logits: np.ndarray) -> np.ndarray:
    """Largest raw logit."""
    return np.asarray(logits, dtype=np.float64).max(axis=1)


def score_neg_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative energy -E(x) = T * logsumexp(logits / T) (T = 1 in Track B); higher = known."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    return temperature * logsumexp(z, axis=1)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero)."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


@dataclass
class GaussianModel:
    """Class means + one shared Ledoit-Wolf covariance (precision stored)."""

    means: np.ndarray  # (K, d)
    precision: np.ndarray  # (d, d)
    shrinkage: float


def fit_gaussian_lw(features: np.ndarray, y: np.ndarray, n_classes: int) -> GaussianModel:
    """Class means + shared Ledoit-Wolf covariance of the within-class residuals.

    Every class in range(n_classes) must have at least one row. Shrinkage is mandatory here
    (n ~ 350 rows against 768 dimensions: the plain covariance is singular).
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"classes without training rows: {np.flatnonzero(counts == 0).tolist()}")
    means = np.stack([x[y == k].mean(axis=0) for k in range(n_classes)])
    resid = x - means[y]
    lw = LedoitWolf(assume_centered=True).fit(resid)  # residuals are centred by construction
    return GaussianModel(means, np.asarray(lw.precision_), float(lw.shrinkage_))


def score_mahalanobis(features: np.ndarray, model: GaussianModel) -> np.ndarray:
    """Minus the smallest class Mahalanobis distance (higher = known)."""
    x = np.asarray(features, dtype=np.float64)
    p = model.precision
    xp = x @ p
    d2 = (
        np.einsum("nd,nd->n", xp, x)[:, None]
        - 2.0 * xp @ model.means.T
        + np.einsum("kd,kd->k", model.means @ p, model.means)[None, :]
    )
    return -np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def score_knn_cosine(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """Cosine similarity to the k-th nearest bank row (L2-normalised features); higher = known."""
    if not 1 <= k <= len(bank):
        raise ValueError(f"k={k} must be in [1, bank size {len(bank)}]")
    sims = l2_normalize(query) @ l2_normalize(bank).T
    return -np.partition(-sims, k - 1, axis=1)[:, k - 1]


@dataclass
class OodScorer:
    """Everything fitted from known TRAIN / CALIBRATION rows only; applied to any other rows."""

    n_classes: int
    temperature: float
    knn_ks: tuple[int, ...]
    ft_bank: np.ndarray
    ft_gauss: GaussianModel
    fz_bank: np.ndarray | None
    fz_gauss: GaussianModel | None

    def score(
        self, logits: np.ndarray, features: np.ndarray, frozen: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """All method scores for a set of rows (frozen=None => frozen methods omitted)."""
        out = {
            "msp": score_msp(logits),
            "msp_temp": score_msp_temp(logits, self.temperature),
            "max_logit": score_max_logit(logits),
            "neg_energy": score_neg_energy(logits),
            "maha_ft": score_mahalanobis(features, self.ft_gauss),
        }
        for k in self.knn_ks:
            out[f"knn{k}_ft"] = score_knn_cosine(features, self.ft_bank, k)
        if frozen is not None:
            if self.fz_gauss is None or self.fz_bank is None:
                raise ValueError("scorer was fitted without frozen features")
            out["maha_frozen"] = score_mahalanobis(frozen, self.fz_gauss)
            for k in self.knn_ks:
                out[f"knn{k}_frozen"] = score_knn_cosine(frozen, self.fz_bank, k)
        return out


def fit_scorer(
    cal_logits: np.ndarray,
    cal_y: np.ndarray,
    train_features: np.ndarray,
    train_y: np.ndarray,
    n_classes: int,
    train_frozen: np.ndarray | None = None,
    knn_ks: tuple[int, ...] = (1, 5),
) -> OodScorer:
    """Fit temperature on calibration NLL; Gaussian + kNN bank on TRAIN known features only."""
    temperature = fit_temperature(cal_logits, cal_y)
    return OodScorer(
        n_classes,
        temperature,
        knn_ks,
        np.asarray(train_features, dtype=np.float64),
        fit_gaussian_lw(train_features, train_y, n_classes),
        None if train_frozen is None else np.asarray(train_frozen, dtype=np.float64),
        None if train_frozen is None else fit_gaussian_lw(train_frozen, train_y, n_classes),
    )


# ------------------------------------------------------------------------- metrics
def auroc_unknown_positive(known: np.ndarray, unknown: np.ndarray) -> float:
    """AUROC, unknown positive scored by -score; equals P(known > unknown) with ties at 1/2."""
    known = np.asarray(known, dtype=np.float64)
    unknown = np.asarray(unknown, dtype=np.float64)
    ranks = rankdata(np.concatenate([known, unknown]))  # ascending: high score = high rank
    n_k, n_u = len(known), len(unknown)
    return float((ranks[:n_k].sum() - n_k * (n_k + 1) / 2) / (n_k * n_u))


def aupr_unknown_positive(known: np.ndarray, unknown: np.ndarray) -> float:
    """Average precision with unknown as positive, scored by -score."""
    y = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
    s = -np.concatenate([np.asarray(known, float), np.asarray(unknown, float)])
    return float(average_precision_score(y, s))


def fpr_at_known_tpr(known: np.ndarray, unknown: np.ndarray, tpr: float = 0.95) -> float:
    """Fraction of unknowns accepted at the largest threshold that accepts >= tpr of the knowns.

    Known is the positive class (accepted = score >= t), so FPR = share of unknown rows with
    score >= t where t is set on THESE known scores (an operating-point-free summary metric).
    """
    t = threshold_at_retention(known, tpr)
    return float(np.mean(np.asarray(unknown, dtype=np.float64) >= t))


def _fmt_counts(pred_names: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(pred_names, return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True)}


def operating_point(
    threshold: float,
    known_scores: np.ndarray,
    known_pred: np.ndarray,
    known_gold: np.ndarray,
    unk_scores: np.ndarray,
    unk_pred: np.ndarray,
    labels: list[str],
    safe_labels: tuple[str, ...] = SAFE_LABELS,
) -> dict[str, Any]:
    """Behaviour at a fixed threshold (accept = score >= threshold). Predictions are local indices.

    Strict mode: a rejection is the only safe outcome for an unknown. Lenient mode: a
    rejection OR an accepted prediction of a label in safe_labels (other / chitchat) counts as
    safe. Known retention is the same in both modes; known rows that end at a safe label are
    reported separately so the lenient gain can be read against its cost.
    """
    safe_idx = np.array([i for i, lab in enumerate(labels) if lab in safe_labels], dtype=int)
    k_acc = known_scores >= threshold
    u_acc = unk_scores >= threshold
    u_safe = ~u_acc | np.isin(unk_pred, safe_idx)
    k_to_safe = k_acc & np.isin(known_pred, safe_idx)
    k_to_safe_wrong = k_to_safe & ~np.isin(known_gold, safe_idx)
    n_cls = len(labels)
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "retention_known": float(k_acc.mean()),
        "n_known_accepted": int(k_acc.sum()),
        "strict_rejection_recall": float((~u_acc).mean()),
        "lenient_rejection_recall": float(u_safe.mean()),
        "n_unknown_not_rejected": int(u_acc.sum()),
        "known_accepted_routed_to_safe": int(k_to_safe.sum()),
        "known_accepted_routed_to_safe_gold_not_safe": int(k_to_safe_wrong.sum()),
        "known_accepted_routed_to_safe_frac_of_known": float(k_to_safe.mean()),
        "known_accepted_routed_to_safe_gold_not_safe_frac_of_known": float(k_to_safe_wrong.mean()),
        "unknown_not_rejected_by_pred_label": _fmt_counts(
            np.array([labels[i] for i in unk_pred[u_acc]], dtype=object)
        ),
        "unknown_all_by_pred_label": _fmt_counts(
            np.array([labels[i] for i in unk_pred], dtype=object)
        ),
        "known_accepted_macro_f1_present": None,
        "known_accepted_accuracy": None,
    }
    if k_acc.any():
        yt, yp = known_gold[k_acc], known_pred[k_acc]
        out["known_accepted_macro_f1_present"] = macro_f1_present(yt, yp, n_cls)
        out["known_accepted_accuracy"] = float(np.mean(yt == yp))
    return out


def open_set_aurc(
    known_scores: np.ndarray,
    known_correct: np.ndarray,
    unk_scores: np.ndarray,
) -> float:
    """AURC over the pooled eval rows: an accepted unknown is always an error, a known one is an
    error when misclassified. Lower is better."""
    conf = np.concatenate([known_scores, unk_scores])
    correct = np.concatenate([known_correct.astype(bool), np.zeros(len(unk_scores), dtype=bool)])
    return aurc(conf, correct)


def run_metrics(
    df: pd.DataFrame,
    methods: list[str],
    labels: list[str],
    retention: float = 0.95,
    safe_labels: tuple[str, ...] = SAFE_LABELS,
) -> dict[str, Any]:
    """Every Track B metric of one run, recomputable from its score table alone.

    df columns: set ('cal' | 'eval'), is_unknown (bool), gold, pred (label names; pred is in the
    model's label space), and one score column per method. Threshold per method = retention
    on the known-only 'cal' rows; metrics use the 'eval' rows (known side + unknown side).
    """
    l2i = {lab: i for i, lab in enumerate(labels)}
    cal = df[df["set"] == "cal"]
    ev = df[df["set"] == "eval"]
    kn = ev[~ev["is_unknown"].astype(bool)]
    un = ev[ev["is_unknown"].astype(bool)]
    if cal["is_unknown"].astype(bool).any():
        raise AssertionError("calibration rows contain unknowns")
    kg = kn["gold"].map(l2i).to_numpy()
    kp = kn["pred"].map(l2i).to_numpy()
    up = un["pred"].map(l2i).to_numpy()
    n_cls = len(labels)
    closed = {
        "n_known_eval": len(kn),
        "macro_f1": macro_f1(kg, kp, n_cls),
        "accuracy": float(np.mean(kg == kp)),
        "note": "closed-set, all known eval rows, no abstention; context for Track A",
    }
    per_method: dict[str, Any] = {}
    for m in methods:
        thr = threshold_at_retention(cal[m].to_numpy(), retention)
        ks, us = kn[m].to_numpy(), un[m].to_numpy()
        rec = {
            "auroc": auroc_unknown_positive(ks, us),
            "aupr": aupr_unknown_positive(ks, us),
            "fpr_at_95tpr": fpr_at_known_tpr(ks, us, 0.95),
            "aurc_open_set": open_set_aurc(ks, kg == kp, us),
            "cal_retention_achieved": float(np.mean(cal[m].to_numpy() >= thr)),
            **operating_point(thr, ks, kp, kg, us, up, labels, safe_labels),
        }
        per_method[m] = rec
    return {
        "n_cal": len(cal),
        "n_eval_known": len(kn),
        "n_eval_unknown": len(un),
        "retention_target": retention,
        "closed_set_known": closed,
        "methods": per_method,
    }


# -------------------------------------------------------------------- aggregation
def mean_std(values: list[float]) -> dict[str, Any]:
    """Mean and sample std (ddof=1; None when fewer than 2 values) plus the raw values."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else None,
        "n": int(len(v)),
        "values": [float(x) for x in v],
    }
