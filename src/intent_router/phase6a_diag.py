"""Open-set improvement round report-only diagnostics D1-D4 (pre-registered).

    python -m intent_router.phase6a_diag --config configs/phase6a_diag.yaml
        --stage <d1|d2|d3|d4|all> [--smoke]

Nothing here is a selection input. D1 is an ORACLE (it uses the held-out labels; every output and
file of it is labelled `oracle`); the training/selection module (phase6a.py) must not import this
module nor read results/phase6a/diag (tests/test_phase6a_diag.py pins both).

Stages (all resumable, deterministic, seed 42, sequential, exclusive GPU access):
  d1  ORACLE separability: logistic-regression probe known-vs-unknown, 5-fold StratifiedGroupKFold
      by dup_group, on the headline holdout and each DEV class; features = v1 Track B fine-tuned
      penultimate embeddings + FROZEN e5-base / e5-large / bge-m3 / LaBSE. Also writes
      n2_decision.json (frozen e5-large vs e5-base headline oracle AUROC, computed by code).
  d2  ID-neutral Track B for v1 (epoch 9), v3 = a1a3 (e* 6) and A1 (e* 11): headline (3 seeds),
      DEV (5), CONFIRM (5); AUROC and strict rejection@95, raw vs ID-neutral, paired bootstrap CIs,
      v3-minus-v1 deltas in both modes, ID-prefix counts, and the `ranking_reversal` flag.
  d3  threshold stability (v1 headline, 3 seeds): 2,000-bootstrap of the known-calibration rows vs
      the 5-fold cross-fitted OOF threshold.
  d4  learning curve (v1 headline): train fractions {0.25, 0.5, 0.75, 1.0} x seeds {42, 43, 44},
      least-squares slope vs log2(fraction) with a seed-bootstrap CI.

All stages share one run store (outputs/phase6a_diag/runs/<run_id>.npz + score CSVs under
results/phase6a/diag/): one trained Track B model yields raw AND ID-neutral features, so a model
trained for D1 is reused by D2/D3/D4 (v1 headline s42-44 and v1 DEV/CONFIRM LOCO s42 are the same
runs everywhere; D4's fraction 1.0 IS the v1 headline run).

Conservative readings of the rules fixed before the runs (also recorded in the output JSONs):
 1. N2 metric: the primary headline oracle AUROC is the MEAN OF FOLD AUROCs (pool = all 500 rows);
    the pooled out-of-fold variant is reported with an agreement flag, never silently preferred.
 2. Probe: L2-normalise -> in-fold standardise -> LogisticRegression(C=1, balanced), fixed a priori
    (tuning on oracle labels would itself leak). The pool is all rows (known classes + held-out
    class rows, incl. train rows). v1 features of known TRAIN rows are in-sample for the encoder
    (it was trained on them) while the unknown rows never were, which inflates v1's oracle AUROC; an
    extra pool `unseen_known_only` (drop known train rows) is reported next to the specified one.
 3. D1 v1 features come from the per-holdout v1 model with seed 42 (headline s42 / LOCO s42).
 4. D2: DEV and CONFIRM use ONE LOCO model seed (42, the 4e CONFIRM convention; 4e DEV used seeds
    0-2); headline seeds 42/43/44. v1 stops at its shipped epoch 9 (not its argmax 8). The
    neutralisation regex is the pre-registered `[A-Z]{2,4}-\\d+` verbatim (no word boundary): in
    neutral mode calibration, known-eval and unknown-eval texts are rewritten at scoring time, the
    scorer (class means, covariance, kNN bank) is fit on the UNCHANGED training features, the
    threshold is re-derived on the neutralised calibration rows. `ranking_reversal` is True when
    the v1-vs-v3 headline winner flips on EITHER strict rejection@95 or AUROC (per-metric flags and
    an "on both" flag are recorded); an exact tie is never counted as a flip.
 5. Known-side evaluation reads test rows exactly as the earlier open-set evaluations did; each run
 logs one
    `phase6a_diag` entry (before any test inference) in
    results/phase6a/diag/test_inference_log.jsonl covering the raw and the neutralised pass
    over the same test rows.
 6. D3 cross-fitting refits only the Gaussian scorer per fold (the encoder is fixed and was trained
    on the known TRAIN rows, so those scores are in-sample for the encoder); folds are grouped by
    dup_group. The bootstrap resamples calibration rows only; test/unknown rows stay fixed, so the
    spread is the threshold-induced spread.
 7. D4: nested per-class subsample (ceil(fraction * n_class), at least 1, fixed permutation per
    (seed, class)); the scorer is refit on the subsample; the slope is a least-squares fit over all
    fraction x seed points; the CI resamples the (few) seeds, so it is coarse and says so.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from intent_router import phase4c as p4
from intent_router import phase4e as p5
from intent_router import trackb_improve as ti
from intent_router.evaluate import git_sha, threshold_at_retention
from intent_router.models import predict, state_dict_sha256
from intent_router.ood import (
    auroc_unknown_positive,
    build_holdout_sets,
    fit_gaussian_lw,
    fit_scorer,
    l2_normalize,
    score_mahalanobis,
)
from intent_router.trackb import _scores_table, gpu_section
from intent_router.train import train_model

STAGES = ("d1", "d2", "d3", "d4", "all")
ALL_ORDER = ("d1", "d3", "d4", "d2")  # d2 is the longest; shared runs are trained once
ORACLE_LABEL = (
    "ORACLE: uses the held-out class labels (known-vs-unknown supervision). Report-only; never a "
    "selection input; never read by the training/selection module."
)
SMOKE_LABEL = "SMOKE: tiny run (1 epoch, no test rows), NOT a result"
SCORE_COLS = ("auroc", "strict_recall_95", "retention_95")  # keys of ti._unit_metrics @ 0.95
ID_PATTERN = r"[A-Z]{2,4}-\d+"  # the pre-registered pattern, verbatim
ID_PARTS_RE = re.compile(r"([A-Z]{2,4})-(\d+)")  # == ID_PATTERN with prefix and digits captured
CALL_TYPE = "phase6a_diag"
KNN_KS = (1, 5)  # fit_scorer default (kNN scores are stored in the tables, never reported here)


# ================================================================================ pure helpers
def neutralize_ids(text: str, prefix: str = "REF-") -> str:
    """Pre-registered ID neutralisation: every `[A-Z]{2,4}-\\d+` -> `REF-<digits>` (digits kept)."""
    return ID_PARTS_RE.sub(lambda m: f"{prefix}{m.group(2)}", text)


def id_prefix_counts(texts: Sequence[str]) -> dict[str, Any]:
    """Rows with an ID, per distinct prefix (a row counts once per prefix it contains)."""
    by_prefix: dict[str, int] = {}
    n_with, n_occ = 0, 0
    for t in texts:
        found = ID_PARTS_RE.findall(t)
        if found:
            n_with += 1
        n_occ += len(found)
        for p in {f[0] for f in found}:
            by_prefix[p] = by_prefix.get(p, 0) + 1
    return {
        "n_rows": len(texts),
        "n_rows_with_id": n_with,
        "n_id_occurrences": n_occ,
        "frac_rows_with_id": (n_with / len(texts)) if len(texts) else None,
        "rows_by_prefix": dict(sorted(by_prefix.items())),
    }


def auroc_pos(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) with ties 1/2 (rank formula); NaN if either side is empty."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def grouped_cv_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Fold id (0..n_splits-1) per row: StratifiedGroupKFold(shuffle, seed) on y with dup_group
    groups, so near-duplicates never straddle a train/test boundary."""
    folds = np.full(len(y), -1, dtype=int)
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups)):
        folds[te] = k
    if (folds < 0).any():
        raise AssertionError("grouped_cv_folds left rows unassigned")
    return folds


def probe_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    max_iter: int = 5000,
) -> np.ndarray:
    """Out-of-fold decision scores (high = unknown) of a logistic-regression probe.

    Per fold: L2-normalise rows, standardise with the TRAIN part's statistics, fit, score the
    held-out part. Deterministic (lbfgs, no randomness).
    """
    Xn = l2_normalize(np.asarray(X, dtype=np.float64))
    out = np.full(len(y), np.nan)
    for k in sorted(set(folds.tolist())):
        te = folds == k
        tr = ~te
        if len(set(y[tr].tolist())) < 2:
            raise ValueError(f"fold {k}: the training part has a single class")
        sc = StandardScaler().fit(Xn[tr])
        clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter)
        clf.fit(sc.transform(Xn[tr]), y[tr])
        out[te] = clf.decision_function(sc.transform(Xn[te]))
    return out


def _cmp_matrix(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """[n_pos, n_neg] with 1 where pos > neg, 0.5 on ties."""
    return (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (pos[:, None] == neg[None, :])


def group_bootstrap_auroc(
    y: np.ndarray,
    s: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, Any]:
    """Oracle AUROC (pooled OOF and mean of fold AUROCs) with a bootstrap CI over dup_groups.

    Groups are resampled with replacement (the unit of independence); a resample is represented by
    integer row weights, so each AUROC is a weighted pairwise-comparison sum (exact, vectorised).
    A fold whose resample has no positive or no negative weight is skipped in that resample's mean.
    Returns point estimates, percentile CIs and the per-fold AUROCs.
    """
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    uniq, gidx = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws]).astype(np.float64)
    W = counts[:, gidx]  # [B, n_rows] row multiplicities

    def weighted(p: np.ndarray, n: np.ndarray) -> np.ndarray:
        if len(p) == 0 or len(n) == 0:
            return np.full(n_boot, np.nan)
        C = _cmp_matrix(s[p], s[n])
        num = ((W[:, p] @ C) * W[:, n]).sum(axis=1)
        den = W[:, p].sum(axis=1) * W[:, n].sum(axis=1)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    pooled_b = weighted(pos, neg)
    fold_ids = sorted(set(folds.tolist()))
    per_fold_b = np.stack(
        [weighted(pos[folds[pos] == k], neg[folds[neg] == k]) for k in fold_ids], axis=1
    )
    mean_fold_b = np.nanmean(per_fold_b, axis=1)
    per_fold_pt = [auroc_pos(s[pos[folds[pos] == k]], s[neg[folds[neg] == k]]) for k in fold_ids]
    a = (1.0 - level) / 2.0

    def ci(v: np.ndarray) -> list[float]:
        v = v[np.isfinite(v)]
        return [float(x) for x in np.quantile(v, [a, 1 - a])] if len(v) else [float("nan")] * 2

    return {
        "auroc_pooled_oof": auroc_pos(s[pos], s[neg]),
        "auroc_pooled_oof_ci": ci(pooled_b),
        "auroc_mean_of_folds": float(np.nanmean(per_fold_pt)),
        "auroc_mean_of_folds_ci": ci(mean_fold_b),
        "per_fold_auroc": per_fold_pt,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "n_groups": int(len(uniq)),
        "n_boot": int(n_boot),
        "ci": f"percentile {level:.0%}, bootstrap over dup_groups",
    }


def n2_decision(
    base: Mapping[str, float], large: Mapping[str, float], margin: float = 0.02
) -> dict[str, Any]:
    """The N2 record from the headline oracle AUROCs of frozen e5-base / e5-large (computed).

    Primary = mean of fold AUROCs; the pooled out-of-fold variant is reported next to it and the
    disagreement (if any) is flagged, never resolved silently.
    """
    eps = 1e-12
    prim = float(large["auroc_mean_of_folds"] - base["auroc_mean_of_folds"])
    pool = float(large["auroc_pooled_oof"] - base["auroc_pooled_oof"])
    run = bool(prim >= margin - eps)
    run_pooled = bool(pool >= margin - eps)
    return {
        "oracle": True,
        "rule": "frozen e5-large headline oracle AUROC >= frozen e5-base + 0.02",
        "margin": margin,
        "metric": "headline holdout, pool = all rows, PRIMARY = mean of fold AUROCs",
        "values": {
            "e5_base_mean_of_folds": float(base["auroc_mean_of_folds"]),
            "e5_large_mean_of_folds": float(large["auroc_mean_of_folds"]),
            "diff_mean_of_folds": prim,
            "e5_base_pooled_oof": float(base["auroc_pooled_oof"]),
            "e5_large_pooled_oof": float(large["auroc_pooled_oof"]),
            "diff_pooled_oof": pool,
        },
        "run_n2": run,
        "run_n2_pooled_variant": run_pooled,
        "primary_and_pooled_agree": run == run_pooled,
        "label": ORACLE_LABEL,
    }


def ranking_reversal(
    raw: Mapping[str, Mapping[str, float]], neutral: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """Does the v1-vs-v3 headline ranking flip in neutral mode?

    raw / neutral: {"v1": {metric: value}, "v3": {...}} with metrics "strict_rej95" and "auroc"
    (headline holdout point estimates, higher = better). The winner of a metric is v3 / v1 / tie;
    a metric is "reversed" iff the raw and neutral winners differ and neither is a tie.
    """
    out: dict[str, Any] = {"by_metric": {}}
    for m in ("strict_rej95", "auroc"):
        w = {}
        for mode, d in (("raw", raw), ("neutral", neutral)):
            delta = float(d["v3"][m] - d["v1"][m])
            w[mode] = {
                "v3_minus_v1": delta,
                "winner": "v3" if delta > 0 else "v1" if delta < 0 else "tie",
            }
        rev = w["raw"]["winner"] != w["neutral"]["winner"] and "tie" not in (
            w["raw"]["winner"],
            w["neutral"]["winner"],
        )
        out["by_metric"][m] = {**w, "reversed": bool(rev)}
    flags = [v["reversed"] for v in out["by_metric"].values()]
    out["ranking_reversal"] = bool(any(flags))
    out["ranking_reversal_on_both_metrics"] = bool(all(flags))
    out["rule"] = "ranking_reversal = the v1-vs-v3 headline winner flips on strict rej@95 OR AUROC"
    return out


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares (slope, intercept) of y on x."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    return slope, float(y.mean() - slope * x.mean())


def slope_seed_bootstrap(
    values: np.ndarray, fractions: Sequence[float], n_boot: int, seed: int, level: float = 0.95
) -> dict[str, Any]:
    """Slope of the metric vs log2(fraction) with a CI from resampling SEEDS.

    values: [n_fractions, n_seeds]. Point = LS slope over all fraction x seed points (equal to the
    slope of the per-fraction means). Each resample draws n_seeds seeds with replacement (shared
    across fractions), averages per fraction, refits. With few seeds the bootstrap distribution is
    discrete and coarse; the per-seed slopes are returned alongside.
    """
    v = np.asarray(values, dtype=np.float64)
    x = np.log2(np.asarray(fractions, dtype=np.float64))
    n_f, n_s = v.shape
    point, intercept = fit_slope(np.repeat(x, n_s), v.reshape(-1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_s, size=(n_boot, n_s))
    means = v[:, idx].mean(axis=2)  # [n_f, B]
    xc = x - x.mean()
    slopes = (xc[:, None] * (means - means.mean(axis=0))).sum(axis=0) / (xc**2).sum()
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(slopes, [a, 1 - a])
    return {
        "slope_per_doubling": point,
        "intercept": intercept,
        "ci": [float(lo), float(hi)],
        "per_seed_slope": [fit_slope(x, v[:, j])[0] for j in range(n_s)],
        "n_boot": int(n_boot),
        "n_seeds": int(n_s),
        "ci_note": f"percentile {level:.0%}, bootstrap over {n_s} seeds (coarse for few seeds)",
    }


def subsample_train_rows(train: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Class-stratified, seeded, NESTED subsample of the training rows (frac in (0, 1]).

    Per class: a permutation fixed by (seed, class) and the first ceil(frac * n) ids (>= 1), so a
    smaller fraction is always a subset of a larger one for the same seed.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]: {frac}")
    if frac == 1.0:
        return train.sort_values("id").reset_index(drop=True)
    keep: list[str] = []
    for lab, g in train.groupby("label", sort=True):
        ids = sorted(g["id"].astype(str))
        h = hashlib.sha256(f"phase6a_d4|{seed}|{lab}".encode()).digest()
        perm = np.random.default_rng(int.from_bytes(h[:8], "little")).permutation(len(ids))
        k = max(1, math.ceil(frac * len(ids) - 1e-9))
        keep += [ids[i] for i in perm[:k]]
    sub = train[train["id"].astype(str).isin(set(keep))]
    return sub.sort_values("id").reset_index(drop=True)


def summarize(x: np.ndarray) -> dict[str, float]:
    """mean, sd (ddof=1), 2.5 / 97.5 percentiles."""
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        "p2_5": float(np.percentile(x, 2.5)),
        "p97_5": float(np.percentile(x, 97.5)),
    }


def threshold_bootstrap(
    cal: np.ndarray,
    known_eval: np.ndarray,
    unknown: np.ndarray,
    retention: float,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Per resample of the calibration-known rows: the 95%-retention threshold (ceil rule, as
    threshold_at_retention), achieved retention on known eval rows and rejection of unknown rows."""
    cal = np.asarray(cal, dtype=np.float64)
    n = len(cal)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    keep = max(1, math.ceil(retention * n - 1e-9))
    thr = np.sort(cal[idx], axis=1)[:, n - keep]  # keep-th largest of each resample
    ke, un = np.asarray(known_eval, dtype=np.float64), np.asarray(unknown, dtype=np.float64)
    return {
        "threshold": thr,
        "retention_known": (ke[None, :] >= thr[:, None]).mean(axis=1),
        "rejection_unknown": (un[None, :] < thr[:, None]).mean(axis=1),
    }


def crossfit_maha_scores(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int, n_splits: int, seed: int
) -> np.ndarray:
    """OOF Mahalanobis scores (higher = known): per fold, class means + Ledoit-Wolf covariance fit
    on the other folds' rows (folds stratified by class, grouped by dup_group), scored on the
    held-out fold. Only the Gaussian is refit; the encoder that produced `feats` is fixed."""
    folds = grouped_cv_folds(y, groups, n_splits, seed)
    out = np.full(len(y), np.nan)
    for k in range(n_splits):
        te = folds == k
        g = fit_gaussian_lw(feats[~te], y[~te], n_classes)
        out[te] = score_mahalanobis(feats[te], g)
    return out


def pool_hidden(hidden: Any, mask: Any, mode: str, dense: Any = None) -> Any:
    """Sentence embedding from last hidden states [B, T, H], L2-normalised (torch tensors).

    mean: attention-masked mean (e5); cls: first token (bge-m3); cls_dense_tanh: first token ->
    Dense(tanh) (LaBSE's sentence-transformers pipeline: CLS pooling -> Dense -> Normalize).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    if mode == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        e = (hidden * m).sum(1) / m.sum(1)
    elif mode == "cls":
        e = hidden[:, 0]
    elif mode == "cls_dense_tanh":
        if dense is None:
            raise ValueError("cls_dense_tanh needs the Dense layer")
        e = torch.tanh(dense(hidden[:, 0]))
    else:
        raise ValueError(f"unknown pooling {mode!r}")
    return F.normalize(e.float(), dim=-1)


# ================================================================================== context
@dataclass
class Diag:
    """Everything a stage needs."""

    cfg: dict[str, Any]
    env: p4.Env
    results: Path
    outputs: Path
    smoke: bool

    @property
    def ctx(self) -> ti.Ctx:
        return self.env.ctx

    @property
    def df_all(self) -> pd.DataFrame:
        return self.ctx.df_all

    @property
    def score(self) -> str:
        return str(self.cfg["trackb"]["score"])

    @property
    def retention(self) -> float:
        return float(self.cfg["trackb"]["retention"])

    @property
    def safe_labels(self) -> list[str]:
        return list(self.ctx.cfg["ood"]["safe_labels"])

    def n_boot(self, configured: int) -> int:
        return int(self.cfg["smoke"]["n_boot"]) if self.smoke else int(configured)

    def headline_seeds(self) -> list[int]:
        s = self.cfg["smoke"]["headline_seeds"] if self.smoke else self.cfg["seeds"]["headline"]
        return [int(x) for x in s]

    def loco_seed(self) -> int:
        seeds = self.cfg["seeds"]["loco"]
        if len(seeds) != 1:
            raise SystemExit("seeds.loco must hold exactly one model seed (see module docstring)")
        return int(seeds[0])

    @property
    def dev_classes(self) -> list[str]:
        return self.env.dev_classes  # smoke: first only

    @property
    def confirm_classes(self) -> list[str]:
        c = list(self.ctx.cfg["confirm_classes"])
        return c[:1] if self.smoke else c

    def epoch_of(self, recipe: str) -> int:
        return int(self.cfg["recipes"][recipe]["epoch"])


def make_diag(config_path: str, smoke: bool) -> Diag:
    """Build the p4.Env from the phase4e config, redirect every output to this module's dirs, and
    check the configured epochs against 4e's epoch.json files (fail closed)."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["neutralise"]["pattern"] != ID_PATTERN:
        raise SystemExit(f"neutralise.pattern must be the pre-registered {ID_PATTERN}")
    args = argparse.Namespace(
        config=cfg["phase4e_config"],
        smoke=smoke,
        no_wandb=True,
        smoke_adopt=None,
        smoke_root=cfg["smoke_root"],
    )
    env0 = p4.make_env(args)
    root = Path(cfg["smoke_root"])
    results = root / "results" if smoke else Path(cfg["results_dir"])
    outputs = root / "outputs" if smoke else Path(cfg["outputs_dir"])
    paths = ti.Paths(
        results, outputs / "unused_trackb_arrays", results / "test_inference_log.jsonl", smoke
    )
    ctx = replace(env0.ctx, P=paths)
    ctx.cfg["gpu_lock_expected_s"] = int(cfg["gpu_lock_expected_s"])
    env = p4.Env(env0.pcfg, p4.P4Paths(results, outputs, smoke), ctx, smoke, None, None)
    if not smoke:
        for key, rc in cfg["recipes"].items():
            ep = p4.read_json(Path(cfg["epoch_files"]) / key / "epoch.json")
            if ep is None or int(ep["e_star"]) != int(rc["epoch"]):
                raise SystemExit(
                    f"configured epoch of {key} ({rc['epoch']}) != "
                    f"{cfg['epoch_files']}/{key}/epoch.json "
                    f"({None if ep is None else ep['e_star']})"
                )
    return Diag(cfg, env, results, outputs, smoke)


# ===================================================================================== run store
@dataclass(frozen=True)
class RunSpec:
    """One Track B model of the shared store: recipe x holdout x seed x train fraction."""

    run_id: str
    recipe: str
    kind: str  # headline | loco
    holdout: tuple[str, ...]
    seed: int
    frac: float = 1.0


def make_run_id(
    recipe: str, kind: str, seed: int, cls: str | None = None, frac: float = 1.0
) -> str:
    """v1_headline_s42, a1a3_loco_orders_s42; a fraction < 1 adds `_f<pct>` (D4)."""
    base = f"{recipe}_headline" if kind == "headline" else f"{recipe}_loco_{cls}"
    if frac < 1.0:
        base += f"_f{round(frac * 100):03d}"
    return f"{base}_s{seed}"


def specs_for(
    dc: Diag, recipe: str, groups: Sequence[str], seeds: Sequence[int] | None = None
) -> list[RunSpec]:
    """Specs of a recipe for groups in {"headline", "dev", "confirm"} (headline seeds default to
    the configured ones; `seeds` overrides them)."""
    out: list[RunSpec] = []
    for g in groups:
        if g == "headline":
            for s in seeds or dc.headline_seeds():
                out.append(
                    RunSpec(
                        make_run_id(recipe, "headline", s),
                        recipe,
                        "headline",
                        dc.ctx.headline_holdout,
                        int(s),
                    )
                )
        else:
            classes = dc.dev_classes if g == "dev" else dc.confirm_classes
            for c in classes:
                s = dc.loco_seed()
                out.append(RunSpec(make_run_id(recipe, "loco", s, c), recipe, "loco", (c,), s))
    return out


def _paths(dc: Diag, spec: RunSpec) -> dict[str, Path]:
    return {
        "npz": dc.outputs / "runs" / f"{spec.run_id}.npz",
        "json": dc.results / "runs" / f"{spec.run_id}.json",
        "raw": dc.results / "scores_raw" / f"{spec.run_id}.csv",
        "neutral": dc.results / "scores_neutral" / f"{spec.run_id}.csv",
    }


def prepare_sets(dc: Diag, spec: RunSpec) -> Any:
    """Holdout sets of a run; a train fraction < 1 replaces `train` by the nested subsample."""
    sets = build_holdout_sets(dc.df_all, list(spec.holdout), dc.smoke)
    if spec.frac < 1.0:
        sets = replace(sets, train=subsample_train_rows(sets.train, spec.frac, spec.seed))
        if set(sets.train["label"]) != set(sets.labels):
            raise AssertionError("fraction subsample dropped a class")
    return sets


def _neutral(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.assign(text=frame["text"].map(lambda t: neutralize_ids(t, prefix)))


def _gather(
    ids: Sequence[str], sources: Sequence[tuple[Sequence[str], tuple[np.ndarray, np.ndarray]]]
) -> tuple[np.ndarray, np.ndarray]:
    """(logits, features) rows for `ids`, looked up in (ids, (logits, features)) sources."""
    where: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sids, (lg, ft) in sources:
        for n, i in enumerate(sids):
            where[i] = (lg[n], ft[n])
    return np.stack([where[i][0] for i in ids]), np.stack([where[i][1] for i in ids])


def train_and_extract(
    dc: Diag, spec: RunSpec, sets: Any, tcfg: Any
) -> tuple[Any, dict[str, np.ndarray], str, float]:
    """Train one model (exclusive GPU access); infer RAW and ID-NEUTRAL outputs before freeing it.

    Batching mirrors the earlier open-set evaluations (train, cal, non-test unknowns, then ONE call
    over all test
    rows) so raw scores can be bit-identical to the saved ones; the neutral pass repeats the same
    calls on neutralised texts. The test call is logged first (one `phase6a_diag` line per run).
    """
    ctx = dc.ctx
    bs = int(ctx.fcfg["predict_batch_size"])
    pre = str(dc.cfg["neutralise"]["replacement_prefix"])
    labels = sets.labels
    meta_run = {
        "run_id": spec.run_id,
        "track": "phase6a-diag",
        "kind": spec.kind,
        "holdout": list(spec.holdout),
        "n_labels": len(labels),
        "candidate": spec.recipe,
        "git_sha": git_sha(),
        "smoke": dc.smoke,
    }
    test_all = (
        None
        if dc.smoke
        else dc.df_all[dc.df_all["split"] == "test"].sort_values("id").reset_index(drop=True)
    )
    unk_nt = sets.eval_unknown[sets.eval_unknown["split"] != "test"]
    with gpu_section(
        int(dc.cfg["gpu_lock_expected_s"]) if not dc.smoke else 600,
        f"intent-router phase6a-diag {spec.run_id}",
        float(dc.cfg["gpu_lock_poll_s"]),
    ) as lock_info:
        res, model, tok = train_model(tcfg, sets.train, sets.cal, meta_run, labels=labels)
        try:
            if int(model.config.num_labels) != len(labels):
                raise AssertionError("model head size differs from the holdout label space")
            sha = state_dict_sha256(model)

            def infer(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                return predict(
                    model, tok, frame["text"].tolist(), tcfg.model_name, tcfg.max_len, bs
                )

            tr = infer(sets.train)
            cal, cal_n = infer(sets.cal), infer(_neutral(sets.cal, pre))
            unk_o, unk_o_n = infer(unk_nt), infer(_neutral(unk_nt, pre))
            test_o = test_o_n = None
            if test_all is not None:
                ti.log_test_call(
                    ctx.P.test_log,
                    f"{spec.run_id}:{sha}",
                    CALL_TYPE,
                    {
                        "run_id": spec.run_id,
                        "n_rows": len(test_all),
                        "role": "phase6a-diag",
                        "modes": ["raw", "neutral"],
                    },
                )
                test_o, test_o_n = infer(test_all), infer(_neutral(test_all, pre))
        finally:
            model = None
            ti._free_gpu()  # noqa: SLF001 - same helper trackb_improve uses between runs

    unk_ids = sets.eval_unknown["id"].tolist()
    ek_ids = sets.eval_known["id"].tolist()
    arrays: dict[str, np.ndarray] = {
        "train_ids": sets.train["id"].to_numpy(dtype=str),
        "train_logits": tr[0],
        "train_features": tr[1],
        "cal_ids": sets.cal["id"].to_numpy(dtype=str),
        "cal_logits": cal[0],
        "cal_features": cal[1],
        "cal_logits_n": cal_n[0],
        "cal_features_n": cal_n[1],
        "ek_ids": sets.eval_known["id"].to_numpy(dtype=str),
        "unk_ids": sets.eval_unknown["id"].to_numpy(dtype=str),
    }
    nt_ids = unk_nt["id"].tolist()
    for suf, unk_part, test_part, cal_part in (
        ("", unk_o, test_o, cal),
        ("_n", unk_o_n, test_o_n, cal_n),
    ):
        srcs: list[tuple[Sequence[str], tuple[np.ndarray, np.ndarray]]] = [(nt_ids, unk_part)]
        if test_part is not None and test_all is not None:
            srcs.append((test_all["id"].tolist(), test_part))
        arrays[f"unk_logits{suf}"], arrays[f"unk_features{suf}"] = _gather(unk_ids, srcs)
        if test_part is not None and test_all is not None:
            ek = _gather(ek_ids, [(test_all["id"].tolist(), test_part)])
        else:  # smoke: no test rows; the known side is the calibration rows
            ek = cal_part
        arrays[f"ek_logits{suf}"], arrays[f"ek_features{suf}"] = ek
    return res, arrays, sha, float(lock_info["waited_s"])


def _local_gold(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return frame["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()


def score_table(sets: Any, A: Mapping[str, np.ndarray], neutral: bool) -> pd.DataFrame:
    """Score table (trackb layout: id, split, set, is_unknown, gold, pred + one column per method).

    The scorer is fit on the UNCHANGED training features in both modes; neutral mode scores the
    neutralised calibration / known-eval / unknown rows and fits the temperature on neutralised
    calibration logits. Frozen-feature methods are not computed.
    """
    suf = "_n" if neutral else ""
    labels, K = sets.labels, len(sets.labels)
    for part, frame in (
        ("train", sets.train),
        ("cal", sets.cal),
        ("ek", sets.eval_known),
        ("unk", sets.eval_unknown),
    ):
        if list(A[f"{part}_ids"]) != frame["id"].astype(str).tolist():
            raise AssertionError(f"stored {part} ids differ from the holdout sets")
    scorer = fit_scorer(
        A[f"cal_logits{suf}"],
        _local_gold(sets.cal, labels),
        A["train_features"],
        _local_gold(sets.train, labels),
        K,
        None,
        KNN_KS,
    )

    def part(
        name: str, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
        lg, ft = A[f"{name}_logits{suf}"], A[f"{name}_features{suf}"]
        return frame, lg.argmax(1), scorer.score(lg, ft, None)

    parts = {
        "cal": part("cal", sets.cal),
        "eval_known": part("ek", sets.eval_known),
        "eval_unknown": part("unk", sets.eval_unknown),
    }
    return _scores_table(sets, parts, {"cal": False, "eval_known": False, "eval_unknown": True})


def ensure_run(dc: Diag, spec: RunSpec) -> None:
    """Train (if needed), store arrays + run json, then the raw and neutral score CSVs."""
    P = _paths(dc, spec)
    if P["neutral"].exists():
        return
    t0 = time.perf_counter()
    sets = prepare_sets(dc, spec)
    if P["npz"].exists() and P["json"].exists():
        A = dict(np.load(P["npz"]))
        print(f"[diag] {spec.run_id}: arrays exist, re-scoring only", flush=True)
    else:
        rec = p5.recipe_for(spec.recipe)
        cand = p4.candidate_for(dc.env, spec.recipe, dc.epoch_of(spec.recipe))
        tcfg = replace(
            ti.train_config(dc.ctx, cand, spec.seed), **p4.recipe_overrides(rec, dc.env.pcfg)
        )
        if tcfg.attn_implementation != "eager" or tcfg.epochs != 20:
            raise AssertionError(
                "Open-set improvement round diagnostics train with the final config "
                "(eager, 20 epochs)"
            )
        forbid = set(sets.eval_unknown["id"].astype(str)) | set(sets.cal["id"].astype(str))
        record: dict[str, Any] = {}

        def build(tr: pd.DataFrame, _ev: pd.DataFrame) -> p4.Extras:
            return p4.build_extras(
                rec, dc.env.inputs if rec.factors else None, dc.env.pcfg, tr, forbid, None
            )

        # This module's `train_model` name is wrapped for the call so A3 rows reach the trainer.
        with p4.patched_training([sys.modules[__name__]], build, record):
            res, A, sha, waited = train_and_extract(dc, spec, sets, tcfg)
        P["npz"].parent.mkdir(parents=True, exist_ok=True)
        np.savez(P["npz"], **A)
        p5.write_json(
            P["json"],
            {
                "run_id": spec.run_id,
                "recipe": spec.recipe,
                "kind": spec.kind,
                "holdout": list(spec.holdout),
                "seed": spec.seed,
                "train_fraction": spec.frac,
                "stop_epoch": tcfg.stop_epoch,
                "labels": sets.labels,
                "extras": record,
                "n_train": len(sets.train),
                "n_cal": len(sets.cal),
                "audit": sets.audit,
                "smoke": dc.smoke,
                "git_sha": git_sha(),
                "call_type": CALL_TYPE,
                "training": {
                    "config": res.config,
                    "epochs": res.epochs,
                    "wall_clock_s": res.wall_clock_s,
                    "load_s": res.load_s,
                    "peak_vram_mb": res.peak_vram_mb,
                    "gpu_exclusive": res.gpu_exclusive,
                    "gpu_foreign_seen": res.gpu_foreign_seen,
                    "nan_detected": res.nan_detected,
                    "fingerprint": sha,
                    "gpu_lock_waited_s": waited,
                    "run_total_s": time.perf_counter() - t0,
                },
            },
        )
    for mode in ("raw", "neutral"):  # neutral last: its CSV is the done marker
        ti._save_csv(score_table(sets, A, mode == "neutral"), P[mode])  # noqa: SLF001
    print(f"[diag] {spec.run_id}: done in {time.perf_counter() - t0:.0f}s", flush=True)


def ensure_runs(dc: Diag, specs: Sequence[RunSpec]) -> None:
    for s in specs:
        ensure_run(dc, s)


def read_table(dc: Diag, spec: RunSpec, mode: str) -> pd.DataFrame:
    return pd.read_csv(_paths(dc, spec)[mode])


def load_arrays(dc: Diag, spec: RunSpec) -> dict[str, np.ndarray]:
    return dict(np.load(_paths(dc, spec)["npz"]))


# =========================================================================================== D1
def encoder_embed(
    spec: Mapping[str, Any], texts: Sequence[str], max_len: int, batch_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """FROZEN sentence embeddings (fp16 on GPU), L2-normalised, with the model's prefix / pooling.

    Returns (float32 [n, d], info incl. parameter count, #rows truncated at max_len, snapshot size).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    name, prefix, pooling = str(spec["name"]), str(spec["prefix"]), str(spec["pooling"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=dtype).to(dev).eval()
    dense = None
    if pooling == "cls_dense_tanh":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        w = load_file(hf_hub_download(name, "2_Dense/model.safetensors"))
        dense = torch.nn.Linear(w["linear.weight"].shape[1], w["linear.weight"].shape[0])
        dense.load_state_dict({"weight": w["linear.weight"], "bias": w["linear.bias"]})
        dense = dense.to(
            dev
        ).eval()  # fp32: the tiny layer is applied to the fp16 CLS state cast up
    full = [prefix + t for t in texts]
    n_trunc = sum(
        len(x) > max_len for x in tok(full, add_special_tokens=True, truncation=False)["input_ids"]
    )
    parts = []
    with torch.no_grad():
        for s in range(0, len(full), batch_size):
            enc = tok(
                full[s : s + batch_size],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(dev)
            h = model(**enc).last_hidden_state
            if dense is not None:
                h = h.float()
            parts.append(pool_hidden(h, enc["attention_mask"], pooling, dense).cpu().numpy())
    n_params = int(sum(p.numel() for p in model.parameters()))
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    snap = None
    try:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(name, local_files_only=True))
        snap = int(sum(f.stat().st_size for f in root.rglob("*") if f.is_file()))
    except Exception:  # noqa: BLE001 - size reporting is best effort
        snap = None
    info = {
        "name": name,
        "pooling": pooling,
        "prefix": prefix,
        "dtype": str(dtype),
        "n_params": n_params,
        "peak_vram_mb": peak_mb,
        "n_rows_truncated_at_max_len": int(n_trunc),
        "max_len": max_len,
        "snapshot_bytes": snap,
    }
    return np.concatenate(parts).astype(np.float32), info


def ensure_frozen(dc: Diag, key: str, ids: list[str], texts: list[str]) -> np.ndarray:
    """Frozen features of `ids` for one encoder (cached npz, rows aligned to ids); exclusive GPU."""
    d1 = dc.cfg["d1"]
    spec = d1["encoders"][key]
    path = dc.outputs / "frozen" / f"{key}.npz"
    if path.exists():
        z = np.load(path)
        pos = {i: n for n, i in enumerate(z["ids"].tolist())}
        if all(i in pos for i in ids) and str(z["name"]) == spec["name"]:
            return np.asarray(z["feats"][[pos[i] for i in ids]], dtype=np.float64)
    with gpu_section(
        900, f"intent-router phase6a-diag frozen {key}", float(dc.cfg["gpu_lock_poll_s"])
    ):
        feats, info = encoder_embed(spec, texts, int(d1["max_len"]), int(d1["encode_batch_size"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, ids=np.array(ids, dtype=str), feats=feats, **{k: str(v) for k, v in info.items()}
    )
    p5.write_json(dc.results / "oracle_encoders" / f"{key}.json", info | {"oracle": True})
    print(f"[d1] frozen {key}: {feats.shape} ({info['n_rows_truncated_at_max_len']} truncated)")
    return feats.astype(np.float64)


def unsupervised_reference(
    dc: Diag, holdout_key: str, run_tables: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """v1 fine-tuned-Mahalanobis AUROC from the open-set evaluation results (results/trackb*) and
    recomputed
    from this run's own seed-42 score table (a cross-check, not the headline number)."""
    ref = Path(dc.cfg["d1"]["unsupervised_reference"])
    out: dict[str, Any] = {"source": str(ref), "method": dc.score}
    try:
        if holdout_key == "headline":
            h = p4.read_json(ref / "headline.json")
            m = h["methods"][dc.score]["auroc"] if h else None
            out["phase3_results"] = (
                None if m is None else {"mean_3_seeds": m["mean"], "values_by_seed": m["values"]}
            )
        else:
            lo = p4.read_json(ref / "loco.json")
            out["phase3_results"] = (
                None
                if lo is None
                else {"seed_42": lo["per_class"][holdout_key]["methods"][dc.score]["auroc"]}
            )
    except (KeyError, TypeError) as exc:
        out["phase3_results"] = None
        out["phase3_results_error"] = repr(exc)
    t = run_tables[holdout_key]
    out["recomputed_from_this_run_seed_42"] = ti.table_auroc(t, dc.score)
    return out


def stage_d1(dc: Diag) -> None:
    """ORACLE probes (+ n2_decision.json)."""
    d1 = dc.cfg["d1"]
    out_path = dc.results / "d1_oracle.json"
    if out_path.exists():
        print(f"[d1] {out_path} exists, skipping")
        return
    hs = dc.headline_seeds()[0]
    holdouts: dict[str, RunSpec] = {}
    for s in specs_for(dc, "v1", ["headline"], [hs]) + specs_for(dc, "v1", ["dev"]):
        holdouts["headline" if s.kind == "headline" else s.holdout[0]] = s
    ensure_runs(dc, list(holdouts.values()))
    frame = dc.df_all.sort_values("id").reset_index(drop=True)
    ids = frame["id"].astype(str).tolist()
    texts = frame["text"].tolist()
    groups = frame["dup_group"].to_numpy()
    feats_frozen = {k: ensure_frozen(dc, k, ids, texts) for k in d1["encoders"]}
    tables = {k: read_table(dc, s, "raw") for k, s in holdouts.items()}
    n_boot = dc.n_boot(d1["n_boot"])
    pr = d1["probe"]
    res: dict[str, Any] = {}
    for hk, spec in holdouts.items():
        A = load_arrays(dc, spec)
        where: dict[str, np.ndarray] = {}
        for part in ("train", "cal", "ek", "unk"):
            for i, f in zip(A[f"{part}_ids"].tolist(), A[f"{part}_features"], strict=True):
                where[i] = f
        v1_feats = np.stack([where[i] for i in ids])
        y = frame["label"].isin(spec.holdout).to_numpy().astype(int)
        known_train = ((frame["split"] == "train") & (y == 0)).to_numpy()
        sets_feats = {
            "v1_finetuned": v1_feats,
            **{f"frozen_{k}": v for k, v in feats_frozen.items()},
        }
        cell: dict[str, Any] = {}
        for fs, X in sets_feats.items():
            cell[fs] = {}
            for pool, mask in (
                ("all_rows", np.ones(len(y), bool)),
                ("unseen_known_only", ~known_train),
            ):
                folds = grouped_cv_folds(
                    y[mask], groups[mask], int(d1["n_splits"]), int(d1["seed"])
                )
                s = probe_oof_scores(
                    X[mask], y[mask], folds, float(pr["C"]), pr["class_weight"], int(pr["max_iter"])
                )
                cell[fs][pool] = group_bootstrap_auroc(
                    y[mask], s, folds, groups[mask], n_boot, int(d1["seed"]), float(d1["level"])
                ) | {"n_rows": int(mask.sum())}
        res[hk] = {
            "holdout": list(spec.holdout),
            "feature_sets": cell,
            "unsupervised_finetuned_mahalanobis": unsupervised_reference(dc, hk, tables),
        }
        print(
            f"[d1] ORACLE {hk}: "
            + ", ".join(
                f"{fs} {c['all_rows']['auroc_mean_of_folds']:.3f}" for fs, c in cell.items()
            )
        )
    head = res["headline"]["feature_sets"]
    n2 = n2_decision(
        head["frozen_e5_base"]["all_rows"],
        head["frozen_e5_large"]["all_rows"],
        float(d1["n2_margin"]),
    )
    n2 |= {"smoke": dc.smoke, "git_sha": git_sha()}
    if dc.smoke:
        n2["label"] = f"{n2['label']} [{SMOKE_LABEL}]"
    p5.write_json(dc.results / "n2_decision.json", n2)
    p5.write_json(
        out_path,
        {
            "oracle": True,
            "label": ORACLE_LABEL if not dc.smoke else f"{ORACLE_LABEL} [{SMOKE_LABEL}]",
            "probe": {
                "features": "L2-normalised, in-fold standardised",
                "model": "LogisticRegression",
                **pr,
                "cv": f"{d1['n_splits']}-fold StratifiedGroupKFold by dup_group, "
                f"shuffle, seed {d1['seed']}",
                "pool": "all rows (known + held-out classes)",
                "extra_pool": "unseen_known_only = drop known TRAIN rows (the v1 encoder was "
                "trained on them)",
                "v1_finetuned_model": "per-holdout v1 model, seed "
                f"{hs if holdouts else None} (headline) / LOCO seed {dc.loco_seed()}, epoch "
                f"{dc.epoch_of('v1')}",
            },
            "n2_decision": n2,
            "holdouts": res,
            "smoke": dc.smoke,
            "git_sha": git_sha(),
        },
    )
    print(f"[d1] ORACLE n2: run_n2={n2['run_n2']} diff={n2['values']['diff_mean_of_folds']:+.4f}")


# =========================================================================================== D2
def cells_of(res: Mapping[str, Any]) -> dict[str, Any]:
    """{metric: {point, lo, hi}} of ti.bootstrap_mean_ci for the reported metrics."""
    return {
        "auroc": {k: res[k]["auroc"] for k in ("point", "lo", "hi")},
        "strict_rej95": {k: res[k]["strict_recall_95"] for k in ("point", "lo", "hi")},
        "retention95": {k: res[k]["retention_95"] for k in ("point", "lo", "hi")},
    }


def paired_cells(a: Mapping[str, Any], b: Mapping[str, Any], level: float) -> dict[str, Any]:
    """a minus b, per metric: point difference + paired percentile CI (same resample indices)."""
    sa = {
        "auroc": a["samples"]["auroc"],
        "strict_rej95": a["samples"]["strict_recall_95"],
        "retention95": a["samples"]["retention_95"],
    }
    sb = {
        "auroc": b["samples"]["auroc"],
        "strict_rej95": b["samples"]["strict_recall_95"],
        "retention95": b["samples"]["retention_95"],
    }
    ci = ti.paired_delta_ci(sa, sb, level)
    pt = {
        "auroc": a["point"]["auroc"] - b["point"]["auroc"],
        "strict_rej95": a["point"]["strict_recall_95"] - b["point"]["strict_recall_95"],
        "retention95": a["point"]["retention_95"] - b["point"]["retention_95"],
    }
    return {
        m: {
            "delta": float(pt[m]),
            "lo": ci[m]["lo"],
            "hi": ci[m]["hi"],
            "ci_excludes_zero": bool(ci[m]["lo"] > 0 or ci[m]["hi"] < 0),
        }
        for m in pt
    }


def row_fingerprint(table: pd.DataFrame) -> str:
    """Hash of the (id, set) rows of a score table, order-free: pairs must share these rows."""
    keys = sorted(table[["id", "set"]].astype(str).agg("|".join, axis=1))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


def group_bootstrap(
    tables: Sequence[pd.DataFrame],
    is_headline: bool,
    score: str,
    retention: float,
    safe: Sequence[str],
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """ti.bootstrap_mean_ci over the units of one group (headline seeds: shared draws; classes:
    independent draws per class, identical for every (recipe, mode) because rows are identical)."""
    units = [ti.boot_unit(t, score, [retention], safe) for t in tables]
    res = ti.bootstrap_mean_ci(units, n_boot, seed, shared_draws=is_headline, level=level)
    res["per_unit"] = [
        {
            "auroc": auroc_unknown_positive(u.known, u.unknown),
            "strict_rej95": float(np.mean(u.unknown < u.thr[float(retention)])),
            "retention95": float(np.mean(u.known >= u.thr[float(retention)])),
        }
        for u in units
    ]
    return res


def reference_checks(dc: Diag, specs: Sequence[RunSpec]) -> dict[str, Any]:
    """Bitwise comparison of this stage's RAW maha_ft scores with saved score tables of the same
    recipe/run (where such a table exists)."""
    if dc.smoke:  # smoke tables have no test rows, so they can never align with the saved ones
        return {"skipped": "smoke"}
    out: dict[str, Any] = {}
    for recipe, ref_dir in dc.cfg["reference_scores"].items():
        rows = []
        for s in (x for x in specs if x.recipe == recipe and x.frac == 1.0):
            rid = s.run_id[len("v1_") :] if recipe == "v1" else s.run_id
            p = Path(ref_dir) / f"{rid}.csv"
            if not p.exists() or not _paths(dc, s)["raw"].exists():
                continue
            cmp = ti.compare_scores(read_table(dc, s, "raw"), pd.read_csv(p), [dc.score])
            rows.append(
                {
                    "run_id": s.run_id,
                    "ref": str(p),
                    "aligned": cmp["aligned"],
                    "pred_equal": cmp.get("pred_equal"),
                    "array_equal": cmp["methods"][dc.score]["array_equal"]
                    if cmp["aligned"]
                    else None,
                    "max_abs_diff": cmp["methods"][dc.score]["max_abs_diff"]
                    if cmp["aligned"]
                    else None,
                }
            )
        out[recipe] = {
            "n_compared": len(rows),
            "n_array_equal": sum(bool(r["array_equal"]) for r in rows),
            "max_abs_diff": max(
                (r["max_abs_diff"] for r in rows if r["max_abs_diff"] is not None), default=None
            ),
            "runs": rows,
        }
    return out


def id_prefix_report(dc: Diag, holdouts: Sequence[tuple[str, tuple[str, ...]]]) -> dict[str, Any]:
    """ID-prefix counts (rows with an ID, per prefix) in known-eval vs unknown rows per holdout."""
    out: dict[str, Any] = {}
    for key, hold in holdouts:
        sets = build_holdout_sets(dc.df_all, list(hold), dc.smoke)
        out[key] = {
            "holdout": list(hold),
            "known_eval": id_prefix_counts(sets.eval_known["text"].tolist()),
            "calibration_known": id_prefix_counts(sets.cal["text"].tolist()),
            "unknown": id_prefix_counts(sets.eval_unknown["text"].tolist()),
        }
    return out


def stage_d2(dc: Diag) -> None:
    """ID-neutral Track B: raw vs neutral, v1 / v3 / A1, headline + DEV + CONFIRM."""
    out_path = dc.results / "d2_id_neutral.json"
    if out_path.exists():
        print(f"[d2] {out_path} exists, skipping")
        return
    bs = dc.cfg["bootstrap"]
    n_boot, seed, level = dc.n_boot(bs["n_resamples"]), int(bs["seed"]), float(bs["level"])
    recipes = ["v1", "a1a3", "a1"]
    groups = ("headline", "dev", "confirm")
    all_specs = {r: {g: specs_for(dc, r, [g]) for g in groups} for r in recipes}
    for r in recipes:
        ensure_runs(dc, [s for g in groups for s in all_specs[r][g]])
    boot: dict[tuple[str, str, str], dict[str, Any]] = {}
    row_fp: dict[tuple[str, tuple[str, ...], int], str] = {}
    for r in recipes:
        for g in groups:
            sp = all_specs[r][g]
            for mode in ("raw", "neutral"):
                tabs = [read_table(dc, s, mode) for s in sp]
                for s, t in zip(sp, tabs, strict=True):
                    fp = row_fingerprint(t)
                    if row_fp.setdefault((g, s.holdout, s.seed), fp) != fp:
                        raise AssertionError(
                            f"{s.run_id} {mode}: eval rows differ across recipes/modes"
                        )
                boot[(r, mode, g)] = group_bootstrap(
                    tabs,
                    g == "headline",
                    dc.score,
                    dc.retention,
                    dc.safe_labels,
                    n_boot,
                    seed,
                    level,
                )
    results: dict[str, Any] = {
        r: {
            m: {
                g: cells_of(boot[(r, m, g)])
                | {
                    "n_units": len(boot[(r, m, g)]["per_unit"]),
                    "per_unit": boot[(r, m, g)]["per_unit"],
                }
                for g in groups
            }
            for m in ("raw", "neutral")
        }
        for r in recipes
    }
    neutral_minus_raw = {
        r: {g: paired_cells(boot[(r, "neutral", g)], boot[(r, "raw", g)], level) for g in groups}
        for r in recipes
    }
    deltas = {
        f"{cand}_minus_v1": {
            m: {g: paired_cells(boot[(cand, m, g)], boot[("v1", m, g)], level) for g in groups}
            for m in ("raw", "neutral")
        }
        for cand in ("a1a3", "a1")
    }
    head = {
        m: {
            r: {
                "strict_rej95": results[r][m]["headline"]["strict_rej95"]["point"],
                "auroc": results[r][m]["headline"]["auroc"]["point"],
            }
            for r in recipes
        }
        for m in ("raw", "neutral")
    }
    rev = ranking_reversal(
        {"v1": head["raw"]["v1"], "v3": head["raw"]["a1a3"]},
        {"v1": head["neutral"]["v1"], "v3": head["neutral"]["a1a3"]},
    )
    hold_keys = [("headline", tuple(dc.ctx.headline_holdout))] + [
        (c, (c,)) for c in [*dc.dev_classes, *dc.confirm_classes]
    ]
    flat_specs = [s for r in recipes for g in groups for s in all_specs[r][g]]
    p5.write_json(
        out_path,
        {
            "label": "measured, report-only" if not dc.smoke else SMOKE_LABEL,
            "disclosure": "report-only. CONFIRM and the headline holdout were already used in "
            "the earlier open-set rounds (adaptive reuse); this is a further look and selects "
            "nothing. DEV/CONFIRM use one LOCO model seed (42).",
            "design": {
                "recipes": {
                    "v1": "v1 (plain), epoch 9",
                    "v3": "a1a3 (A1+A3), e* 6",
                    "a1": "A1, e* 11",
                },
                "modes": "raw; neutral = every [A-Z]{2,4}-\\d+ -> REF-<digits> in calibration, "
                "known-eval and unknown texts at scoring time (training text unchanged; scorer fit "
                "on unchanged training features; threshold from neutralised calibration rows)",
                "score": dc.score,
                "retention": dc.retention,
                "groups": {
                    "headline": f"seeds {dc.headline_seeds()} (seed mean, shared draws)",
                    "dev": f"{dc.dev_classes} LOCO seed {dc.loco_seed()}",
                    "confirm": f"{dc.confirm_classes} LOCO seed {dc.loco_seed()}",
                },
                "bootstrap": {
                    "n_resamples": n_boot,
                    "seed": seed,
                    "level": level,
                    "kind": "rows resampled within each holdout, stratified known/unknown; "
                    "pairs share draws (identical rows)",
                },
            },
            "results": results,
            "neutral_minus_raw": neutral_minus_raw,
            "v3_minus_v1": deltas["a1a3_minus_v1"],
            "a1_minus_v1": deltas["a1_minus_v1"],
            "ranking_reversal": rev["ranking_reversal"],
            "ranking_reversal_detail": rev,
            "id_prefix_counts": id_prefix_report(dc, hold_keys),
            "reference_checks_raw_vs_saved": reference_checks(dc, flat_specs),
            "smoke": dc.smoke,
            "git_sha": git_sha(),
        },
    )
    print(f"[d2] wrote {out_path}; ranking_reversal={rev['ranking_reversal']}")


# =========================================================================================== D3
def stage_d3(dc: Diag) -> None:
    """Threshold stability of v1's headline operating point: bootstrap vs cross-fitted threshold."""
    c = dc.cfg["d3"]
    out_path = dc.results / "d3_threshold_stability.json"
    if out_path.exists():
        print(f"[d3] {out_path} exists, skipping")
        return
    specs = specs_for(dc, "v1", ["headline"])
    ensure_runs(dc, specs)
    n_boot = dc.n_boot(c["n_boot"])
    per_seed: dict[str, Any] = {}
    pooled: dict[str, list[np.ndarray]] = {
        "threshold": [],
        "retention_known": [],
        "rejection_unknown": [],
    }
    for s in specs:
        t = read_table(dc, s, "raw")
        cal = t[t["set"] == "cal"][dc.score].to_numpy(np.float64)
        ev = t[t["set"] == "eval"]
        unk = ev["is_unknown"].astype(bool)
        ek, uk = (
            ev.loc[~unk, dc.score].to_numpy(np.float64),
            ev.loc[unk, dc.score].to_numpy(np.float64),
        )
        thr0 = threshold_at_retention(cal, dc.retention)
        boot = threshold_bootstrap(cal, ek, uk, dc.retention, n_boot, int(c["seed"]))
        for k in pooled:
            pooled[k].append(boot[k])
        # cross-fitted OOF threshold over the known train+val rows (Gaussian refit per fold)
        sets = prepare_sets(dc, s)
        A = load_arrays(dc, s)
        feats = np.concatenate([A["train_features"], A["cal_features"]]).astype(np.float64)
        known = pd.concat([sets.train, sets.cal], ignore_index=True)
        y = _local_gold(known, sets.labels)
        oof = crossfit_maha_scores(
            feats,
            y,
            known["dup_group"].to_numpy(),
            len(sets.labels),
            int(c["n_splits"]),
            int(c["seed"]),
        )
        thr_cf = threshold_at_retention(oof, dc.retention)

        def at(th: float, ek: np.ndarray = ek, uk: np.ndarray = uk) -> dict[str, float]:
            return {
                "threshold": float(th),
                "retention_known_eval": float(np.mean(ek >= th)),
                "rejection_unknown": float(np.mean(uk < th)),
            }

        per_seed[str(s.seed)] = {
            "n_cal": len(cal),
            "n_known_eval": len(ek),
            "n_unknown": len(uk),
            "calibration_threshold": at(thr0),
            "bootstrap": {k: summarize(v) for k, v in boot.items()},
            "bootstrap_binomial_sd_retention_finite_test_set": float(
                math.sqrt(dc.retention * (1 - dc.retention) / len(ek))
            ),
            "crossfit_oof_threshold": at(thr_cf)
            | {
                "n_oof_rows": int(len(oof)),
                "oof_retention_at_threshold": float(np.mean(oof >= thr_cf)),
            },
        }
    p5.write_json(
        out_path,
        {
            "label": "measured, report-only" if not dc.smoke else SMOKE_LABEL,
            "design": {
                "model": f"v1 headline, seeds {dc.headline_seeds()}, epoch {dc.epoch_of('v1')}",
                "score": dc.score,
                "retention": dc.retention,
                "bootstrap": f"{n_boot} resamples of the known-calibration (val) rows; threshold = "
                "ceil rule on each resample; retention on known test rows and rejection of unknown "
                "rows evaluated at that threshold (eval rows fixed: threshold-induced spread only)",
                "crossfit": f"{c['n_splits']}-fold StratifiedGroupKFold (class, dup_group) over "
                "known train+val rows; Gaussian refit per fold, encoder fixed (train rows are "
                "in-sample for the encoder); threshold = ceil-rule 95% of the OOF scores",
            },
            "per_seed": per_seed,
            "pooled_over_seeds": {k: summarize(np.concatenate(v)) for k, v in pooled.items()},
            "smoke": dc.smoke,
            "git_sha": git_sha(),
        },
    )
    print(f"[d3] wrote {out_path}")


# =========================================================================================== D4
def stage_d4(dc: Diag) -> None:
    """Learning curve of v1 on the headline holdout; slope vs log2(fraction)."""
    c = dc.cfg["d4"]
    out_path = dc.results / "d4_learning_curve.json"
    if out_path.exists():
        print(f"[d4] {out_path} exists, skipping")
        return
    fracs = [float(f) for f in (dc.cfg["smoke"]["d4_fractions"] if dc.smoke else c["fractions"])]
    seeds = [int(s) for s in (dc.cfg["smoke"]["d4_seeds"] if dc.smoke else c["seeds"])]
    specs: dict[tuple[float, int], RunSpec] = {}
    for f in fracs:
        for s in seeds:
            specs[(f, s)] = RunSpec(
                make_run_id("v1", "headline", s, None, f),
                "v1",
                "headline",
                tuple(dc.ctx.headline_holdout),
                s,
                f,
            )
    ensure_runs(dc, list(specs.values()))
    auroc = np.zeros((len(fracs), len(seeds)))
    rej = np.zeros_like(auroc)
    n_train: dict[str, int] = {}
    for i, f in enumerate(fracs):
        for j, s in enumerate(seeds):
            u = ti.boot_unit(
                read_table(dc, specs[(f, s)], "raw"), dc.score, [dc.retention], dc.safe_labels
            )
            auroc[i, j] = auroc_unknown_positive(u.known, u.unknown)
            rej[i, j] = float(np.mean(u.unknown < u.thr[float(dc.retention)]))
            n_train[f"{f}|{s}"] = len(prepare_sets(dc, specs[(f, s)]).train)
    nb = dc.n_boot(c["n_boot"])
    p5.write_json(
        out_path,
        {
            "label": "measured, report-only" if not dc.smoke else SMOKE_LABEL,
            "design": {
                "model": f"v1 recipe, headline holdout, stop epoch {dc.epoch_of('v1')}",
                "fractions": fracs,
                "seeds": seeds,
                "score": dc.score,
                "subsample": "class-stratified, seeded, nested; scorer refit on the subsample; "
                "calibration/eval rows unchanged",
            },
            "n_train_rows": n_train,
            "auroc": {
                "by_fraction_seed": auroc.tolist(),
                "mean_by_fraction": auroc.mean(axis=1).tolist(),
                "slope_vs_log2_fraction": slope_seed_bootstrap(
                    auroc, fracs, nb, int(c["seed"]), float(c["level"])
                ),
            },
            "strict_rejection_95": {
                "by_fraction_seed": rej.tolist(),
                "mean_by_fraction": rej.mean(axis=1).tolist(),
                "slope_vs_log2_fraction": slope_seed_bootstrap(
                    rej, fracs, nb, int(c["seed"]), float(c["level"])
                ),
            },
            "smoke": dc.smoke,
            "git_sha": git_sha(),
        },
    )
    print(f"[d4] wrote {out_path}")


# ========================================================================================= main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Open-set improvement round diagnostics D1-D4 (report-only)"
    )
    ap.add_argument("--config", default="configs/phase6a_diag.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny: 1 epoch, no test rows; writes only under outputs/phase6a_diag_smoke",
    )
    args = ap.parse_args(argv)
    dc = make_diag(args.config, args.smoke)
    fns = {"d1": stage_d1, "d2": stage_d2, "d3": stage_d3, "d4": stage_d4}
    for st in ALL_ORDER if args.stage == "all" else (args.stage,):
        print(f"[diag] stage {st} start", flush=True)
        fns[st](dc)


if __name__ == "__main__":
    main()
