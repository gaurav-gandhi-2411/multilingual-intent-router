from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from scipy import stats as sps

from intent_router.train import SIBLING_PREFIX, parent_index

T = TypeVar("T")
CALL_TYPES = ("evaluation", "determinism_inference")


# ------------------------------------------------------------------ basic helpers
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row softmax of logits / temperature, computed in float64 and returned as float32."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def array_sha256(a: np.ndarray) -> str:
    """sha256 of the raw bytes of an array (C order); shape and dtype are folded in."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts matrix C[gold, pred], shape (n_classes, n_classes), int64."""
    idx = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _f1_from_confusion(
    conf: np.ndarray, present_only: bool, labels: np.ndarray | None = None
) -> float:
    """Macro-F1 from a confusion matrix.

    labels (class indices): mean over exactly these classes (a fixed set; classes with no
    support and no predictions score 0, like sklearn zero_division=0); overrides present_only.

    present_only=False: mean over all classes, absent classes score 0 (matches stats.macro_f1).
    present_only=True: mean over classes occurring in gold or pred (sklearn's default labels).
    """
    tp = np.diag(conf).astype(float)
    pred_n = conf.sum(axis=0).astype(float)
    true_n = conf.sum(axis=1).astype(float)
    denom = pred_n + true_n
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    if labels is not None:
        return float(f1[np.asarray(labels, dtype=np.int64)].mean())
    if present_only:
        mask = denom > 0
        return float(f1[mask].mean()) if mask.any() else 0.0
    return float(f1.mean())


def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over classes present in gold or predictions (used for slices/subsets)."""
    return _f1_from_confusion(confusion_matrix(y_true, y_pred, n_classes), True)


def macro_f1_gold_classes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over the classes present in the GOLD labels only.

    Predictions into classes absent from gold add no class; they only lower precision/recall of
    the gold classes. Equals sklearn f1_score(labels=sorted(unique(y_true)), average="macro",
    zero_division=0).
    """
    gold = np.unique(np.asarray(y_true))
    return _f1_from_confusion(confusion_matrix(y_true, y_pred, n_classes), False, gold)


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[dict[str, Any]]:
    """Per-class precision / recall / F1 / support (0 where undefined)."""
    conf = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(conf).astype(float)
    pred_n = conf.sum(axis=0).astype(float)
    true_n = conf.sum(axis=1).astype(float)
    prec = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros_like(tp), where=true_n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return [
        {
            "class_index": i,
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(true_n[i]),
        }
        for i in range(n_classes)
    ]


def row_normalise(conf: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix (rows with no support stay all-zero)."""
    tot = conf.sum(axis=1, keepdims=True).astype(float)
    return np.divide(conf, tot, out=np.zeros(conf.shape, dtype=float), where=tot > 0)


# ----------------------------------------------------------------------- bootstrap
def _resample_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, n_resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row-resampling indices (B, n) and the (B, K, K) confusions they produce."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    flat = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(y_pred, dtype=np.int64)
    confs = np.empty((n_resamples, n_classes, n_classes), dtype=np.int64)
    for b in range(n_resamples):
        confs[b] = np.bincount(flat[idx[b]], minlength=n_classes**2).reshape(n_classes, -1)
    return idx, confs


def _percentile_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_resamples: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
    present_only: bool = False,
    fixed_labels: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs for macro-F1 and accuracy by plain row resampling.

    Rows are resampled with replacement (not stratified): the CI then also carries the
    variability of the class mix, which is what a fresh test draw would have. Both metrics
    share the same resamples. present_only selects the macro-F1 convention (see
    _f1_from_confusion); the headline test number uses False. fixed_labels (class indices)
    fixes the macro-F1 label set once for the point estimate and every resample (slices pass
    the slice's gold classes).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _, confs = _resample_confusions(y_true, y_pred, n_classes, n_resamples, seed)
    f1s = np.array([_f1_from_confusion(c, present_only, fixed_labels) for c in confs])
    accs = np.trace(confs, axis1=1, axis2=2) / len(y_true)
    point_conf = confusion_matrix(y_true, y_pred, n_classes)
    out = {}
    for name, vals, point in (
        ("macro_f1", f1s, _f1_from_confusion(point_conf, present_only, fixed_labels)),
        ("accuracy", accs, float(np.mean(y_true == y_pred))),
    ):
        lo, hi = _percentile_ci(vals, level)
        out[name] = {"point": float(point), "lo": lo, "hi": hi}
    return out


def paired_bootstrap_delta_f1(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_classes: int,
    n_resamples: int = 10_000,
    seed: int = 42,
    level: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(A) - macro-F1(B): the same resampled rows score both."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    ka = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(pred_a, dtype=np.int64)
    kb = np.asarray(y_true, dtype=np.int64) * n_classes + np.asarray(pred_b, dtype=np.int64)
    deltas = np.empty(n_resamples)
    for b in range(n_resamples):
        ca = np.bincount(ka[idx[b]], minlength=n_classes**2).reshape(n_classes, -1)
        cb = np.bincount(kb[idx[b]], minlength=n_classes**2).reshape(n_classes, -1)
        deltas[b] = _f1_from_confusion(ca, False) - _f1_from_confusion(cb, False)
    lo, hi = _percentile_ci(deltas, level)
    point = _f1_from_confusion(confusion_matrix(y_true, pred_a, n_classes), False) - (
        _f1_from_confusion(confusion_matrix(y_true, pred_b, n_classes), False)
    )
    return {
        "delta": float(point),
        "lo": lo,
        "hi": hi,
        "frac_resamples_le_zero": float(np.mean(deltas <= 0)),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired correctness: two-sided binomial test on discordant pairs.

    b = A right & B wrong, c = A wrong & B right; under H0 each discordant pair is a fair coin.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    p = 1.0 if n == 0 else float(sps.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "n_discordant": n, "p_value": p}


# ------------------------------------------------------------------ hierarchy view
def family_mask(labels: list[str]) -> np.ndarray:
    """Boolean per class index: True for the shipment_information.* siblings."""
    return np.array([lab.startswith(SIBLING_PREFIX) for lab in labels])


def hierarchy_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Parent-level accuracy (siblings collapsed) and where shipment-family errors land."""
    parents = np.array(parent_index(labels))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fam = family_mask(labels)
    err = y_pred != y_true
    fam_err = err & fam[y_true]
    within = fam_err & fam[y_pred]
    return {
        "parent_accuracy": float(np.mean(parents[y_true] == parents[y_pred])),
        "n_parent_classes": int(parents.max() + 1),
        "n_errors": int(err.sum()),
        "n_family_gold_errors": int(fam_err.sum()),
        "n_family_errors_within_family": int(within.sum()),
        "share_family_errors_within_family": (
            float(within.sum() / fam_err.sum()) if fam_err.sum() else None
        ),
        "share_all_errors_within_family": float(within.sum() / err.sum()) if err.sum() else None,
    }


# ------------------------------------------------------------------- calibration
def ece_bins(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error on max-probability confidence, equal-width bins (lo, hi]."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y_true)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        row = {"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": cnt}
        if cnt:
            acc_b, conf_b = float(correct[m].mean()), float(conf[m].mean())
            ece += cnt / n * abs(acc_b - conf_b)
            row.update({"accuracy": acc_b, "confidence": conf_b})
        bins.append(row)
    return float(ece), bins


def nll(logits: np.ndarray, y_true: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold class under softmax(logits / T)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(z)), np.asarray(y_true)].mean())


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature minimising NLL of softmax(logits / T), by LBFGS on log T (T stays > 0)."""
    import torch

    z = torch.tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.tensor(np.asarray(y_true), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> Any:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


# ------------------------------------------------------------ selective prediction
def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: accept the k most confident rows (stable on ties), k = 1..n.

    Returns (coverage = k/n, risk = error rate among the accepted k).
    """
    conf = np.asarray(conf, dtype=float)
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(err) / k


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean selective risk over k = 1..n (lower is better)."""
    return float(risk_coverage(conf, correct)[1].mean())


def threshold_at_retention(scores: np.ndarray, retention: float = 0.95) -> float:
    """Largest threshold t such that mean(scores >= t) >= retention.

    With n known scores, keep ceil(retention * n) of them and set t to the smallest kept score
    (the ~5th percentile for 95%). Accept rule everywhere is `score >= t`, so ties at t are all
    accepted and the achieved retention can exceed the target; it is never below it. This is
    exact for any n (the single shared implementation; used by final, trackb and ood).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if len(s) == 0:
        raise ValueError("no scores to set a threshold on")
    keep = max(1, math.ceil(retention * len(s) - 1e-9))  # 1e-9: float noise in retention * n
    return float(s[keep - 1])


def selective_at_threshold(
    conf: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    n_classes: int,
) -> dict[str, float | int | None]:
    """Coverage, accuracy and macro-F1 at conf >= t.

    macro_f1_present: mean over the classes present in the GOLD labels of the accepted rows
    (fixed label set; wrong predictions into other classes add no class).
    """
    keep = np.asarray(conf) >= threshold
    n_keep = int(keep.sum())
    out: dict[str, float | int | None] = {
        "threshold": float(threshold),
        "n_accepted": n_keep,
        "coverage": float(n_keep / len(keep)),
        "accuracy": None,
        "macro_f1_present": None,
    }
    if n_keep:
        yt, yp = np.asarray(y_true)[keep], np.asarray(y_pred)[keep]
        out["accuracy"] = float(np.mean(yt == yp))
        out["macro_f1_present"] = macro_f1_gold_classes(yt, yp, n_classes)
    return out


# ------------------------------------------------------------------ test-once guard
def git_sha() -> str:
    """HEAD commit SHA (+ '-dirty' when the tree has changes); 'unknown' if git fails."""
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TestEvalRefused(RuntimeError):  # noqa: N818 - reads better as a refusal than *Error
    """Raised when a second test 'evaluation' of the same model fingerprint is attempted."""

    __test__ = False  # not a pytest class despite the Test* name


def read_test_log(log_path: Path) -> list[dict[str, Any]]:
    """Parse the jsonl test-evaluation log (missing file => empty)."""
    if not log_path.exists():
        return []
    return [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]


def evaluate_test_once(
    fingerprint: str,
    call_type: str,
    infer: Callable[[], T],
    log_path: Path,
    metrics: Callable[[T], dict[str, Any]] | None = None,
    sha: str | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[T, dict[str, Any] | None]:
    """The only door to test-split inference and metrics.

    Appends one jsonl line (timestamp, model fingerprint, call_type, git SHA) BEFORE running
    anything, so a crash after the test rows were read still counts as a use. An 'evaluation'
    for a fingerprint that already has one raises TestEvalRefused (nothing runs, nothing is
    logged). 'determinism_inference' may repeat and must not compute metrics.
    Returns (infer(), metrics(infer()) or None).
    """
    if call_type not in CALL_TYPES:
        raise ValueError(f"call_type must be one of {CALL_TYPES}, got {call_type!r}")
    if call_type == "evaluation" and metrics is None:
        raise ValueError("an 'evaluation' call must supply metrics")
    if call_type == "determinism_inference" and metrics is not None:
        raise ValueError("'determinism_inference' computes no metrics")
    prior = [
        e
        for e in read_test_log(log_path)
        if e["call_type"] == "evaluation" and e["model_fingerprint"] == fingerprint
    ]
    if call_type == "evaluation" and prior:
        raise TestEvalRefused(
            f"test split already evaluated for model {fingerprint[:12]} at "
            f"{prior[0]['timestamp']} (git {prior[0]['git_sha']}); refusing a second evaluation"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_fingerprint": fingerprint,
        "call_type": call_type,
        "git_sha": sha if sha is not None else git_sha(),
        **(extra or {}),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    result = infer()
    return result, (metrics(result) if metrics is not None else None)
