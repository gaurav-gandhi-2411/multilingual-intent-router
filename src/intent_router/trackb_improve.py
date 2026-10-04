from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from intent_router import data as data_mod
from intent_router.evaluate import (
    git_sha,
    macro_f1_present,
    softmax,
    threshold_at_retention,
)
from intent_router.final import write_json
from intent_router.models import QUERY_PREFIX, build_model, predict_hidden, state_dict_sha256
from intent_router.ood import (
    GaussianModel,
    HoldoutSets,
    auroc_unknown_positive,
    build_holdout_sets,
    fit_gaussian_lw,
    fit_scorer,
    method_names,
    run_metrics,
    score_mahalanobis,
    score_msp,
    score_neg_energy,
)
from intent_router.trackb import _scores_table, frozen_lookup, gpu_section, plan_runs
from intent_router.train import TrainConfig, train_fold, train_model

STAGES = (
    "reproduce",
    "c2_guard",
    "c2_dev",
    "c3_fit_check",
    "c3_guard",
    "c3_dev",
    "c3_latency",
    "c4_dev_confirm_headline",
    "b1_ood",
    "select",
    "confirm",
    "business",
    "final_ood",
)
C1_METHODS = ("maha_ml", "fuse_maha_energy")
CURRENT_KEY = "base/maha_ft"  # the shipped open-set evaluation method on the reproduced base models
PARTS = ("train", "cal", "ek", "unk")  # npz row groups: train / calibration / eval known / unknown
LABEL_BOOTSTRAP = "estimate: row bootstrap within each holdout, stratified by known/unknown"


# ======================================================================= pure scoring (C1)
@dataclass
class MultiLayerMaha:
    """One Ledoit-Wolf Mahalanobis model per layer; layer scores are standardised then summed.

    mu / sd are the mean / population std (ddof=0) of each layer's raw score on the
    CALIBRATION-KNOWN rows only; train rows fit the Gaussians, no other row touches mu / sd.
    """

    gaussians: list[GaussianModel]
    mu: np.ndarray
    sd: np.ndarray

    def layer_scores(self, hidden: np.ndarray) -> np.ndarray:
        """Raw per-layer scores [n, L] (minus the smallest class Mahalanobis distance)."""
        return np.stack(
            [score_mahalanobis(hidden[:, i], g) for i, g in enumerate(self.gaussians)], axis=1
        )

    def score(self, hidden: np.ndarray) -> np.ndarray:
        """Sum over layers of (raw - mu) / sd; higher = known."""
        return ((self.layer_scores(hidden) - self.mu) / self.sd).sum(axis=1)


def fit_multilayer_maha(
    train_hidden: np.ndarray, y_train: np.ndarray, n_classes: int, cal_hidden: np.ndarray
) -> MultiLayerMaha:
    """Fit per-layer Gaussians on train-known pooled states; standardise on cal-known rows."""
    n_layers = train_hidden.shape[1]
    gaussians = [fit_gaussian_lw(train_hidden[:, i], y_train, n_classes) for i in range(n_layers)]
    raw = np.stack(
        [score_mahalanobis(cal_hidden[:, i], g) for i, g in enumerate(gaussians)], axis=1
    )
    sd = raw.std(axis=0)
    return MultiLayerMaha(gaussians, raw.mean(axis=0), np.where(sd > 0, sd, 1.0))


@dataclass
class RankFusion:
    """Mean of calibration-known empirical-CDF ranks of several scores (all higher = known)."""

    sorted_cal: list[np.ndarray]

    def transform(self, components: Sequence[np.ndarray]) -> np.ndarray:
        """Per component: share of cal-known scores <= x (in [0, 1]); then the mean."""
        if len(components) != len(self.sorted_cal):
            raise ValueError("number of components differs from the fitted fusion")
        ranks = [
            np.searchsorted(s, np.asarray(x, dtype=np.float64), side="right") / len(s)
            for s, x in zip(self.sorted_cal, components, strict=True)
        ]
        return np.mean(ranks, axis=0)


def fit_rank_fusion(cal_components: Sequence[np.ndarray]) -> RankFusion:
    """Fit the empirical CDFs on the calibration-known scores of each component."""
    return RankFusion([np.sort(np.asarray(c, dtype=np.float64)) for c in cal_components])


# ===================================================================== pure selection (DEV)
def select_on_dev(
    dev_auroc: Mapping[str, Mapping[str, float]],
    eligible: Sequence[str],
    current: str,
    dev_classes: Sequence[str],
) -> dict[str, Any]:
    """Pre-registered decision rule, on DEV-class AUROCs only.

    dev_auroc: key -> {class: AUROC}; every key must cover exactly dev_classes (a CONFIRM class,
    or a missing DEV class, raises). Winner = highest DEV-mean AUROC among `eligible` (ties go to
    the earlier key). Ship iff winner_DEV - current_DEV > std_ddof1(current's DEV per-class
    AUROCs) / sqrt(n_dev); otherwise keep the current method.
    """
    dev = set(dev_classes)
    for key, per_class in dev_auroc.items():
        got = set(per_class)
        if got != dev:
            raise ValueError(
                f"{key}: class set must equal the DEV classes; extra {sorted(got - dev)}, "
                f"missing {sorted(dev - got)}"
            )
    if current not in dev_auroc:
        raise KeyError(f"current method {current!r} missing from the DEV table")
    unknown_keys = [k for k in eligible if k not in dev_auroc]
    if unknown_keys:
        raise KeyError(f"eligible keys without DEV AUROCs: {unknown_keys}")
    means = {k: float(np.mean([v[c] for c in sorted(dev)])) for k, v in dev_auroc.items()}
    cur_vals = np.array([dev_auroc[current][c] for c in sorted(dev)])
    margin = float(cur_vals.std(ddof=1) / math.sqrt(len(cur_vals)))
    out: dict[str, Any] = {
        "dev_classes": sorted(dev),
        "dev_mean_auroc": means,
        "current": current,
        "current_dev_mean": means[current],
        "margin_std_ddof1_over_sqrt_n": margin,
        "rule": "ship iff winner DEV mean - current DEV mean > std_ddof1(current DEV)/sqrt(5)",
        "winner": None,
        "winner_dev_mean": None,
        "winner_minus_current": None,
        "ship": False,
        "shipped": current,
    }
    if eligible:
        winner = max(eligible, key=lambda k: (means[k], -list(eligible).index(k)))
        diff = means[winner] - means[current]
        ship = bool(diff > margin)
        out |= {
            "winner": winner,
            "winner_dev_mean": means[winner],
            "winner_minus_current": diff,
            "ship": ship,
            "shipped": winner if ship else current,
        }
    return out


# ============================================================ pure metrics: curves, bootstrap
def rejection_retention_curve(
    cal: np.ndarray, known: np.ndarray, unknown: np.ndarray, targets: Sequence[float]
) -> dict[str, list[float]]:
    """Per retention target: measured known retention and strict rejection recall (cal threshold).

    The threshold keeps ceil(r * n) calibration-known rows; accept = score >= t (unchanged rule).
    """
    thr = [threshold_at_retention(cal, r) for r in targets]
    return {
        "target": [float(r) for r in targets],
        "retention_known": [float(np.mean(known >= t)) for t in thr],
        "strict_rejection_recall": [float(np.mean(unknown < t)) for t in thr],
    }


@dataclass
class BootUnit:
    """One holdout's eval scores for one method, plus its calibration-fixed thresholds."""

    known: np.ndarray
    unknown: np.ndarray
    unk_safe: np.ndarray  # bool per unknown row: predicted label is in safe_labels
    thr: dict[float, float]  # retention -> threshold (from calibration rows, never resampled)


def _tag(r: float) -> str:
    return f"{round(r * 100)}"


def _unit_metrics(u: BootUnit, k: np.ndarray, un: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    """Metrics for B resamples at once: k [B, n_k], un [B, n_u], s [B, n_u] (bool safe-pred)."""
    n_k, n_u = k.shape[1], un.shape[1]
    ranks = rankdata(np.concatenate([k, un], axis=1), axis=1)
    out: dict[str, Any] = {
        "auroc": (ranks[:, :n_k].sum(axis=1) - n_k * (n_k + 1) / 2) / (n_k * n_u)
    }
    keep = max(1, math.ceil(0.95 * n_k - 1e-9))
    t95 = np.sort(k, axis=1)[:, n_k - keep]  # fpr@95tpr: threshold re-derived on the resample
    out["fpr_at_95tpr"] = (un >= t95[:, None]).mean(axis=1)
    for r, t in u.thr.items():
        out[f"retention_{_tag(r)}"] = (k >= t).mean(axis=1)
        out[f"strict_recall_{_tag(r)}"] = (un < t).mean(axis=1)
        out[f"lenient_recall_{_tag(r)}"] = ((un < t) | s).mean(axis=1)
    return out


def bootstrap_mean_ci(
    units: Sequence[BootUnit],
    n_resamples: int,
    seed: int,
    shared_draws: bool = False,
    level: float = 0.95,
) -> dict[str, Any]:
    """Row bootstrap within each holdout, stratified by known/unknown; mean over units per resample.

    For every unit the known rows and the unknown rows are resampled separately with replacement
    (n_known and n_unknown fixed). Calibration thresholds stay fixed. shared_draws=True reuses one
    index draw for all units (units that share the same rows, e.g. the headline seeds).
    Returns {"point", "lo", "hi", "samples"}: samples[metric] is the per-resample mean over units.
    """
    rng = np.random.default_rng(seed)
    per_unit: list[dict[str, Any]] = []
    draws: tuple[np.ndarray, np.ndarray] | None = None
    for u in units:
        n_k, n_u = len(u.known), len(u.unknown)
        if shared_draws and draws is not None:
            if draws[0].shape[1] != n_k or draws[1].shape[1] != n_u:
                raise ValueError("shared_draws needs units with identical row counts")
            ik, iu = draws
        else:
            ik = rng.integers(0, n_k, size=(n_resamples, n_k))
            iu = rng.integers(0, n_u, size=(n_resamples, n_u))
            draws = (ik, iu)
        per_unit.append(_unit_metrics(u, u.known[ik], u.unknown[iu], u.unk_safe[iu]))
    points = [  # B=1 "resample" that is the identity: the point estimate, same code path
        _unit_metrics(u, u.known[None, :], u.unknown[None, :], u.unk_safe[None, :]) for u in units
    ]
    samples = {m: np.mean([p[m] for p in per_unit], axis=0) for m in per_unit[0]}
    alpha = (1.0 - level) / 2.0
    lo_hi = {m: np.quantile(v, [alpha, 1.0 - alpha]) for m, v in samples.items()}
    return {
        "point": {m: float(np.mean([p[m][0] for p in points])) for m in points[0]},
        "lo": {m: float(lo_hi[m][0]) for m in samples},
        "hi": {m: float(lo_hi[m][1]) for m in samples},
        "samples": samples,
    }


def paired_delta_ci(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], level: float = 0.95
) -> dict[str, Any]:
    """CI of mean(a) - mean(b) from per-resample samples drawn with the same indices."""
    alpha = (1.0 - level) / 2.0
    out: dict[str, Any] = {}
    for m in a:
        d = np.asarray(a[m]) - np.asarray(b[m])
        lo, hi = np.quantile(d, [alpha, 1.0 - alpha])
        out[m] = {"mean_delta": float(d.mean()), "lo": float(lo), "hi": float(hi)}
    return out


# =============================================================== pure business arithmetic
def business_row(
    n_messages: int, prevalence: float, retention: float, strict: float, lenient: float
) -> dict[str, float]:
    """Per n_messages at unknown prevalence pi, from MEASURED retention and rejection recall.

    known wrongly abstained = n(1-pi)(1-retention); unknowns caught = n*pi*recall; unknowns
    misrouted = n*pi*(1-recall) (strict), and the same with the lenient recall.
    """
    n_unk, n_known = n_messages * prevalence, n_messages * (1.0 - prevalence)
    return {
        "prevalence": prevalence,
        "retention_measured": retention,
        "known_wrongly_abstained": n_known * (1.0 - retention),
        "unknowns_caught_strict": n_unk * strict,
        "unknowns_misrouted_strict": n_unk * (1.0 - strict),
        "unknowns_caught_lenient": n_unk * lenient,
        "unknowns_misrouted_lenient": n_unk * (1.0 - lenient),
    }


# ============================================================ pure reproduction comparison
def compare_scores(mine: pd.DataFrame, ref: pd.DataFrame, methods: Sequence[str]) -> dict[str, Any]:
    """Bitwise comparison of per-method score columns of two score tables, aligned by id/set.

    Both tables are expected to be CSV round-trips (the same %.9g formatting), so equality is at
    the stored precision. Reports array_equal and the max absolute difference per method.
    """
    key = ["id", "set"]
    if len(mine) != len(ref) or mine.duplicated(key).any() or ref.duplicated(key).any():
        return {"aligned": False, "n_mine": len(mine), "n_ref": len(ref)}
    a = mine.set_index(key).sort_index()
    b = ref.set_index(key).sort_index()
    if not a.index.equals(b.index):
        return {"aligned": False, "n_mine": len(mine), "n_ref": len(ref)}
    out: dict[str, Any] = {
        "aligned": True,
        "n_rows": len(a),
        "pred_equal": bool((a["pred"] == b["pred"]).all()),
        "methods": {},
    }
    for m in methods:
        x, y = a[m].to_numpy(np.float64), b[m].to_numpy(np.float64)
        out["methods"][m] = {
            "array_equal": bool(np.array_equal(x, y)),
            "max_abs_diff": float(np.max(np.abs(x - y))) if len(x) else 0.0,
        }
    return out


# ================================================================================= context
@dataclass(frozen=True)
class Paths:
    """Where one invocation writes; smoke runs are fully redirected under outputs/."""

    results: Path
    outputs: Path
    test_log: Path
    smoke: bool

    @property
    def scores(self) -> Path:
        return self.results / "scores"

    @property
    def runs(self) -> Path:
        return self.results / "runs"


@dataclass
class Ctx:
    """Everything a stage needs; built once in main()."""

    cfg: dict[str, Any]
    fcfg: dict[str, Any]
    tb: dict[str, Any]
    P: Paths
    df_all: pd.DataFrame
    use_wandb: bool
    smoke: bool
    fz: dict[str, np.ndarray] | None = None

    @property
    def base_model(self) -> str:
        return str(self.fcfg["train"]["model_name"])

    @property
    def ks(self) -> tuple[int, ...]:
        return tuple(int(k) for k in self.tb["ood"]["knn_k"])

    @property
    def all_methods(self) -> list[str]:
        return [*method_names(self.ks), *C1_METHODS]

    @property
    def headline_holdout(self) -> tuple[str, ...]:
        return tuple(sorted(self.tb["headline"]["holdout"]))

    @property
    def loco_classes(self) -> list[str]:
        labels = list(data_mod.LABELS)
        return [s.holdout[0] for s in plan_runs(self.tb, labels, "loco")]

    def frozen_of(self) -> dict[str, np.ndarray]:
        """
        id -> frozen e5-base feature row (cache of open-set evaluation; only requested ids are
        read).
        """
        if self.fz is None:
            ids = self.df_all["id"].astype(str).tolist()
            mx = int(self.fcfg["train"]["max_len"])
            arr = frozen_lookup(Path(self.cfg["frozen_cache"]), ids, self.base_model, mx)
            if arr is None:
                raise RuntimeError(f"frozen cache {self.cfg['frozen_cache']} missing/incomplete")
            self.fz = dict(zip(ids, arr, strict=True))
        return self.fz


def make_paths(cfg: dict[str, Any], smoke: bool) -> Paths:
    """Real paths from the config, or a sandbox under outputs/trackb_improve_smoke."""
    if smoke:
        root = Path("outputs/trackb_improve_smoke")
        return Paths(root / "results", root / "outputs", root / "test_inference_log.jsonl", True)
    return Paths(
        Path(cfg["results_dir"]), Path(cfg["outputs_dir"]), Path(cfg["test_inference_log"]), False
    )


# ---------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    """A model recipe: key for tables, cid prefix for run ids, training overrides."""

    key: str
    cid: str
    family: str  # base | c2 | c3
    model_name: str
    lr: float
    supcon_lambda: float | None
    group: str
    stop_epoch: int | None = None  # deployed epoch (None => final.yaml stop_epoch)
    grad_accum: int = 1


def _grp(ctx: Ctx, name: str) -> str:
    return f"{ctx.cfg['wandb']['group_prefix']}-{name}"


def base_candidate(ctx: Ctx, group: str = "c1") -> Candidate:
    t = ctx.fcfg["train"]
    return Candidate("base", "", "base", t["model_name"], float(t["lr"]), None, _grp(ctx, group))


def guard_path(ctx: Ctx, key: str) -> Path:
    return ctx.P.results / "guard" / f"{key}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def c2_candidates(ctx: Ctx) -> list[Candidate]:
    out = []
    for lam in ctx.cfg["c2"]["lambdas"]:
        key = f"c2_l{lam:g}"
        g = _read_json(guard_path(ctx, key))
        t = ctx.fcfg["train"]
        out.append(
            Candidate(
                key,
                key.replace(".", "p") + "_",
                "c2",
                t["model_name"],
                float(t["lr"]),
                float(lam),
                _grp(ctx, "c2"),
                int(g["deployed_epoch"]) if g else None,
            )  # fmt: skip
        )
    return out


def c3_accum(ctx: Ctx) -> int:
    """Gradient-accumulation steps chosen by c3_fit_check; stops the stage if C3 was aborted."""
    chk = _read_json(ctx.P.results / "c3_fit_check.json")
    if chk is None:
        raise SystemExit("run --stage c3_fit_check first")
    if chk["aborted"]:
        raise SystemExit("C3 aborted by c3_fit_check (see c3_fit_check.json): no C3 runs")
    return int(chk["chosen"]["accum"])


def c3_candidates(ctx: Ctx) -> list[Candidate]:
    c3 = ctx.cfg["c3"]
    acc = c3_accum(ctx)
    out = []
    for lr in c3["lrs"]:
        key = f"c3_lr{lr:g}"
        g = _read_json(guard_path(ctx, key))
        out.append(
            Candidate(
                key,
                key + "_",
                "c3",
                c3["model_name"],
                float(lr),
                None,
                _grp(ctx, "c3"),
                int(g["deployed_epoch"]) if g else None,
                acc,
            )  # fmt: skip
        )
    return out


def train_config(ctx: Ctx, cand: Candidate, seed: int, group: str | None = None) -> TrainConfig:
    """The final model's train block with only the candidate's overrides applied."""
    tr = dict(ctx.fcfg["train"])
    tr.pop("query_prefix")
    if QUERY_PREFIX.get(ctx.base_model, "") != ctx.fcfg["train"]["query_prefix"]:
        raise ValueError("final.yaml query_prefix differs from models.QUERY_PREFIX")
    tr |= {
        "model_name": cand.model_name, "lr": cand.lr, "model_seed": seed, "label_smoothing": 0.0,
        "supcon_lambda": cand.supcon_lambda,
        "supcon_temperature": float(ctx.cfg["c2"]["temperature"]),
        "grad_accum": cand.grad_accum,
    }  # fmt: skip
    if cand.family == "c3":
        tr["batch_size"] = int(ctx.cfg["c3"]["batch_size"])
    if cand.stop_epoch is not None:
        tr["stop_epoch"] = cand.stop_epoch
    if ctx.smoke:
        tr["stop_epoch"] = 1
    wb = {"wandb_project": ctx.cfg["wandb"]["project"], "wandb_group": group or cand.group}
    return TrainConfig.from_dict({**tr, **wb})


# --------------------------------------------------------------------------------- runs
@dataclass(frozen=True)
class ImpRun:
    """One model to train and score: candidate x holdout x seed."""

    run_id: str
    cand: Candidate
    kind: str  # headline | loco
    holdout: tuple[str, ...]
    seed: int
    call_type: str


def run_id_for(cand: Candidate, kind: str, seed: int, cls: str | None = None) -> str:
    if kind == "headline":
        return f"{cand.cid}headline_s{seed}"
    return f"{cand.cid}loco_{cls}_s{seed}"


def plan(
    ctx: Ctx,
    cand: Candidate,
    classes: Sequence[str],
    headline_seeds: Sequence[int],
    loco_seeds: Sequence[int],
    call_type: str,
) -> list[ImpRun]:
    """Runs for the given LOCO classes x loco_seeds, then the headline holdout x its seeds."""
    runs = [
        ImpRun(run_id_for(cand, "loco", s, c), cand, "loco", (c,), s, call_type)
        for s in loco_seeds
        for c in classes
    ]
    runs += [
        ImpRun(
            run_id_for(cand, "headline", s), cand, "headline", ctx.headline_holdout, s, call_type
        )
        for s in headline_seeds
    ]
    return runs


def _local_gold(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return frame["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()


def log_test_call(path: Path, fingerprint: str, call_type: str, extra: dict[str, Any]) -> int:
    """Append one jsonl line BEFORE any test-split inference; returns the attempt number.

    A crash-resume re-trains the same deterministic model and repeats the call: it is allowed
    (resumability) but shows up as attempt > 1 so a repeated test use is never silent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = 0
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                e = json.loads(ln)
                prior += int(
                    e.get("model_fingerprint") == fingerprint and e.get("call_type") == call_type
                )
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_fingerprint": fingerprint,
        "call_type": call_type,
        "git_sha": git_sha(),
        "attempt": prior + 1,
        **extra,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return prior + 1


def score_run(
    ctx: Ctx, sets: HoldoutSets, A: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All method scores of one run from its saved arrays; fits use train-known / cal-known only."""
    labels, K = sets.labels, len(sets.labels)
    fz_of = ctx.frozen_of()

    def fz(frame: pd.DataFrame) -> np.ndarray:
        return np.stack([fz_of[i] for i in frame["id"]])

    y_tr, y_cal = _local_gold(sets.train, labels), _local_gold(sets.cal, labels)
    scorer = fit_scorer(
        A["cal_logits"], y_cal, A["train_features"], y_tr, K, fz(sets.train), ctx.ks
    )
    ml = fit_multilayer_maha(A["train_hidden"], y_tr, K, A["cal_hidden"])
    fusion_names = list(ctx.cfg["c1"]["fusion_components"])
    cal_base = scorer.score(A["cal_logits"], A["cal_features"], fz(sets.cal))
    fusion = fit_rank_fusion([cal_base[n] for n in fusion_names])

    def scores(part: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        s = scorer.score(A[f"{part}_logits"], A[f"{part}_features"], fz(frame))
        s["maha_ml"] = ml.score(A[f"{part}_hidden"])
        s["fuse_maha_energy"] = fusion.transform([s[n] for n in fusion_names])
        return s

    parts = {
        "cal": (sets.cal, A["cal_logits"].argmax(1), scores("cal", sets.cal)),
        "eval_known": (sets.eval_known, A["ek_logits"].argmax(1), scores("ek", sets.eval_known)),
        "eval_unknown": (
            sets.eval_unknown, A["unk_logits"].argmax(1), scores("unk", sets.eval_unknown),
        ),
    }  # fmt: skip
    table = _scores_table(sets, parts, {"cal": False, "eval_known": False, "eval_unknown": True})
    info = {
        "temperature": scorer.temperature,
        "ledoit_wolf_shrinkage": {
            "fine_tuned": scorer.ft_gauss.shrinkage,
            "multi_layer": [g.shrinkage for g in ml.gaussians],
        },
        "multi_layer_cal_mu": ml.mu.tolist(),
        "multi_layer_cal_sd": ml.sd.tolist(),
    }
    return table, info


def _save_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    table.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, path)  # CSV existence == run complete (resume rule)


def _free_gpu() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_extract(
    ctx: Ctx, run: ImpRun, sets: HoldoutSets, tcfg: TrainConfig
) -> tuple[Any, dict[str, np.ndarray], str, float]:
    """Train one model (exclusive GPU access), then extract logits / head features / pooled states.

    Inference batching mirrors open-set evaluation exactly (train, cal, non-test unknowns, then ONE
    call over
    all test rows) so the reproduction can be bitwise. Test inference is logged first.
    Returns (FoldResult, arrays, fingerprint, lock wait seconds).
    """
    bs = int(ctx.fcfg["predict_batch_size"])
    layers = [int(x) for x in ctx.cfg["c1"]["layers"]]
    labels = sets.labels
    meta_run = {
        "run_id": run.run_id, "track": "B-improve", "kind": run.kind,
        "holdout": list(run.holdout), "n_labels": len(labels), "candidate": run.cand.key,
        "git_sha": git_sha(), "smoke": ctx.P.smoke,
    }  # fmt: skip

    def infer(model: Any, tok: Any, frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        return predict_hidden(
            model, tok, frame["text"].tolist(), tcfg.model_name, tcfg.max_len, bs, layers
        )

    test_all = (
        None
        if ctx.smoke
        else ctx.df_all[ctx.df_all["split"] == "test"].sort_values("id").reset_index(drop=True)
    )
    expected = int(ctx.cfg["gpu_lock_expected_s"]) * (3 if run.cand.family == "c3" else 1)
    with gpu_section(
        expected if not ctx.smoke else 600,
        f"intent-router trackB-improve {run.run_id}",
        float(ctx.cfg["gpu_lock_poll_s"]),
    ) as lock_info:
        res, model, tok = train_model(tcfg, sets.train, sets.cal, meta_run, labels=labels)
        try:
            if int(model.config.num_labels) != len(labels):
                raise AssertionError("model head size differs from the holdout label space")
            sha = state_dict_sha256(model)
            tr = infer(model, tok, sets.train)
            cal = infer(model, tok, sets.cal)
            unk_nt = sets.eval_unknown[sets.eval_unknown["split"] != "test"]
            unk_nt_out = infer(model, tok, unk_nt)
            test_out: tuple[np.ndarray, ...] | None = None
            if not ctx.smoke:
                assert test_all is not None
                log_test_call(
                    ctx.P.test_log, f"{run.run_id}:{sha}", run.call_type,
                    {"run_id": run.run_id, "n_rows": len(test_all), "role": "trackB-improve"},
                )  # fmt: skip
                test_out = infer(model, tok, test_all)
        finally:
            model = None
            _free_gpu()

    unk_at: dict[str, list[np.ndarray]] = {
        i: [x[n] for x in unk_nt_out] for n, i in enumerate(unk_nt["id"])
    }
    if test_out is not None:
        assert test_all is not None
        pos = {i: n for n, i in enumerate(test_all["id"])}
        ek_idx = [pos[i] for i in sets.eval_known["id"]]
        ek = tuple(x[ek_idx] for x in test_out)
        for i in sets.eval_unknown["id"]:
            if i in pos:
                unk_at[i] = [x[pos[i]] for x in test_out]
    else:  # smoke: no test inference; the known side is the calibration rows
        ek = cal
    unk = tuple(np.stack([unk_at[i][j] for i in sets.eval_unknown["id"]]) for j in range(3))
    arrays: dict[str, np.ndarray] = {}
    for part, frame, vals in (
        ("train", sets.train, tr),
        ("cal", sets.cal, cal),
        ("ek", sets.eval_known, ek),
        ("unk", sets.eval_unknown, unk),
    ):
        arrays[f"{part}_ids"] = frame["id"].to_numpy(dtype=str)
        arrays[f"{part}_logits"], arrays[f"{part}_features"], arrays[f"{part}_hidden"] = vals
    return res, arrays, sha, float(lock_info["waited_s"])


def run_one(ctx: Ctx, run: ImpRun) -> None:
    """Train, save arrays + run json, score every method, write the score CSV last."""
    csv_path = ctx.P.scores / f"{run.run_id}.csv"
    if csv_path.exists():
        print(f"[improve] {run.run_id}: score CSV exists, skipping", flush=True)
        return
    t0 = time.perf_counter()
    sets = build_holdout_sets(ctx.df_all, list(run.holdout), ctx.P.smoke)
    tcfg = train_config(ctx, run.cand, run.seed)
    if tcfg.attn_implementation != "eager" or tcfg.epochs != 20:
        raise AssertionError(
            "Open-set scorer comparison trains with the final config (eager, 20-epoch schedule)"
        )
    res, arrays, sha, waited = train_and_extract(ctx, run, sets, tcfg)
    out_dir = ctx.P.outputs / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{run.run_id}.npz", **arrays)
    table, info = score_run(ctx, sets, arrays)
    meta = {
        "run_id": run.run_id, "candidate": run.cand.key, "kind": run.kind,
        "holdout": list(run.holdout), "seed": run.seed, "labels": sets.labels,
        "call_type": run.call_type, "git_sha": git_sha(), "smoke": ctx.P.smoke,
        "audit": sets.audit | info,
        "training": {
            "config": res.config, "epochs": res.epochs, "wall_clock_s": res.wall_clock_s,
            "load_s": res.load_s, "peak_vram_mb": res.peak_vram_mb,
            "gpu_exclusive": res.gpu_exclusive, "gpu_foreign_seen": res.gpu_foreign_seen,
            "nan_detected": res.nan_detected, "fingerprint": sha, "gpu_lock_waited_s": waited,
            "wandb_url": res.wandb_url, "run_total_s": time.perf_counter() - t0,
        },
    }  # fmt: skip
    write_json(ctx.P.runs / f"{run.run_id}.json", meta)
    _save_csv(table, csv_path)
    print(
        f"[improve] {run.run_id}: train {res.wall_clock_s:.0f}s total "
        f"{time.perf_counter() - t0:.0f}s gpu_exclusive={res.gpu_exclusive}",
        flush=True,
    )


def run_all(ctx: Ctx, runs: Sequence[ImpRun]) -> None:
    for r in runs:
        run_one(ctx, r)


# ============================================================================== table access
def read_scores(ctx: Ctx, run_id: str) -> pd.DataFrame:
    path = ctx.P.scores / f"{run_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing score CSV {path} (run the earlier stage first)")
    return pd.read_csv(path)


def labels_for(ctx: Ctx, holdout: Sequence[str]) -> list[str]:
    return [c for c in sorted(ctx.df_all["label"].unique()) if c not in set(holdout)]


def table_auroc(table: pd.DataFrame, method: str) -> float:
    ev = table[table["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool)
    return auroc_unknown_positive(ev.loc[~unk, method].to_numpy(), ev.loc[unk, method].to_numpy())


def boot_unit(
    table: pd.DataFrame, method: str, retentions: Sequence[float], safe_labels: Sequence[str]
) -> BootUnit:
    cal = table[table["set"] == "cal"][method].to_numpy()
    ev = table[table["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool)
    return BootUnit(
        ev.loc[~unk, method].to_numpy(np.float64),
        ev.loc[unk, method].to_numpy(np.float64),
        ev.loc[unk, "pred"].isin(list(safe_labels)).to_numpy(),
        {float(r): threshold_at_retention(cal, r) for r in retentions},
    )


# ================================================================================ stage: reproduce
def stage_reproduce(ctx: Ctx) -> None:
    """
    Re-create the open-set evaluation base models by deterministic retraining; compare MSP bitwise.
    """
    base = base_candidate(ctx, "c1")
    classes = ctx.loco_classes
    if ctx.smoke:
        classes = [ctx.cfg["dev_classes"][0]]
        runs = plan(ctx, base, classes, [], [int(ctx.cfg["loco_seed"])], "trackb_reproduce")
    else:
        runs = plan(
            ctx, base, classes, ctx.cfg["headline_seeds"], [int(ctx.cfg["loco_seed"])],
            "trackb_reproduce",
        )  # fmt: skip
    run_all(ctx, runs)
    if ctx.smoke:
        _print_smoke_scores(ctx, runs[0].run_id)
        return
    ref_dir = Path(ctx.cfg["phase3_dir"])
    per_run: dict[str, Any] = {}
    for r in runs:
        mine = read_scores(ctx, r.run_id)
        ref = pd.read_csv(ref_dir / "scores" / f"{r.run_id}.csv")
        cmp = compare_scores(mine, ref, method_names(ctx.ks))
        meta_mine = json.loads((ctx.P.runs / f"{r.run_id}.json").read_text(encoding="utf-8"))
        meta_ref = json.loads((ref_dir / "runs" / f"{r.run_id}.json").read_text(encoding="utf-8"))
        fp_mine = meta_mine["training"]["fingerprint"]
        fp_ref = meta_ref["training"]["fingerprint"]
        cmp["state_dict_sha256_equal"] = fp_mine == fp_ref
        cmp["state_dict_sha256_phase3"] = fp_ref
        cmp["state_dict_sha256_reproduced"] = fp_mine
        per_run[r.run_id] = cmp
    msp = {k: v["methods"]["msp"] for k, v in per_run.items() if v.get("aligned")}
    out = {
        "label": "measured: open-set evaluation base models re-created by deterministic retraining "
        "(eager)",
        "n_runs": len(per_run),
        "all_aligned": all(v.get("aligned") for v in per_run.values()),
        "all_msp_bitwise_equal": bool(msp) and all(v["array_equal"] for v in msp.values()),
        "max_abs_diff_msp": max((v["max_abs_diff"] for v in msp.values()), default=None),
        "all_methods_bitwise_equal": all(
            m["array_equal"]
            for v in per_run.values()
            if v.get("aligned")
            for m in v["methods"].values()
        ),  # fmt: skip
        "all_state_dicts_equal": all(v["state_dict_sha256_equal"] for v in per_run.values()),
        "comparison_note": "CSV round-trip (%.9g) on both sides; cal + eval rows aligned by id/set",
        "runs": per_run,
    }
    write_json(ctx.P.results / "reproduce.json", out)
    print(
        f"[reproduce] msp bitwise equal: {out['all_msp_bitwise_equal']}, max abs diff "
        f"{out['max_abs_diff_msp']}, state dicts equal: {out['all_state_dicts_equal']}"
    )


def _print_smoke_scores(ctx: Ctx, run_id: str) -> None:
    t = read_scores(ctx, run_id)
    print(f"[smoke] {run_id}: stop_epoch 1, known side = calibration rows, NOT a result")
    for m in ctx.all_methods:
        print(f"  {m:18s} AUROC {table_auroc(t, m):.4f}")


# ================================================================================ stages: guards
def run_guard(ctx: Ctx, cand: Candidate) -> dict[str, Any]:
    """5-fold CV (train+val only), fold-seed s0, model seed 0, 20-epoch schedule, eager."""
    from intent_router.cv import fold_split, load_cv_frame

    g = ctx.cfg["guard"]
    if abs(float(g["threshold"]) - (float(g["comparator"]) - float(g["tolerance"]))) > 1e-9:
        raise AssertionError("guard threshold != comparator - tolerance")
    frame = load_cv_frame()
    folds = [0] if ctx.smoke else list(range(int(g["n_folds"])))
    gdir = ctx.P.results / "guard"
    tcfg = replace(
        train_config(ctx, replace(cand, stop_epoch=None), int(g["model_seed"])),
        stop_epoch=1 if ctx.smoke else None,
    )
    curves: list[list[float]] = []
    for f in folds:
        fp = gdir / f"{cand.key}_f{f}.json"
        if fp.exists():
            rec = json.loads(fp.read_text(encoding="utf-8"))
        else:
            tr, ev = fold_split(frame, int(g["fold_seed_idx"]), f)
            meta = {"run_id": f"guard_{cand.key}_f{f}", "fold": f, "candidate": cand.key}
            with gpu_section(
                int(ctx.cfg["gpu_lock_expected_s"]) * (3 if cand.family == "c3" else 1),
                f"intent-router trackB-improve guard {cand.key} f{f}",
                float(ctx.cfg["gpu_lock_poll_s"]),
            ):
                res = train_fold(tcfg, tr, ev, meta)
            _free_gpu()
            rec = {
                "candidate": cand.key, "fold": f,
                "macro_f1": [e["macro_f1"] for e in res.epochs],
                "wall_clock_s": res.wall_clock_s, "peak_vram_mb": res.peak_vram_mb,
                "wandb_url": res.wandb_url, "nan_detected": res.nan_detected,
                "gpu_exclusive": res.gpu_exclusive, "config": res.config,
            }  # fmt: skip
            write_json(fp, rec)
            print(f"[guard] {cand.key} fold {f}: best-epoch F1 {max(rec['macro_f1']):.4f}")
        curves.append(rec["macro_f1"])
    mean_curve = np.mean(np.array(curves), axis=0)
    epoch = int(np.argmax(mean_curve)) + 1  # 1-indexed; ties go to the earliest epoch
    value = float(mean_curve[epoch - 1])
    thr = float(g["threshold"])
    out = {
        "candidate": cand.key, "label": "measured, CV s0 5-fold, model seed 0, eager",
        "n_folds": len(folds), "smoke": ctx.smoke,
        "mean_curve": mean_curve.tolist(), "deployed_epoch": epoch, "macro_f1_at_deployed": value,
        "threshold": thr, "comparator": float(g["comparator"]),
        "passed": bool(value >= thr),
        "why": None if value >= thr else f"macro-F1 {value:.4f} < guard {thr:.4f} at epoch {epoch}",
    }  # fmt: skip
    if not ctx.smoke:
        write_json(guard_path(ctx, cand.key), out)
    print(f"[guard] {cand.key}: epoch {epoch} macro-F1 {value:.4f} passed={out['passed']}")
    return out


def stage_guard(ctx: Ctx, cands: list[Candidate]) -> None:
    for c in cands:
        if not ctx.smoke and guard_path(ctx, c.key).exists():
            print(f"[guard] {c.key}: summary exists, skipping")
            continue
        run_guard(ctx, c)


def passing(ctx: Ctx, cands: list[Candidate]) -> list[Candidate]:
    """Candidates whose guard summary exists and passed (a missing summary stops the stage)."""
    out = []
    for c in cands:
        g = _read_json(guard_path(ctx, c.key))
        if g is None:
            raise SystemExit(f"no guard summary for {c.key}: run its guard stage first")
        if g["passed"]:
            out.append(c)
        else:
            print(f"[guard] {c.key} FAILED the Track A guard ({g['why']}): no Track B runs")
    return out


def stage_dev(ctx: Ctx, cands: list[Candidate]) -> None:
    """LOCO models for the 5 DEV classes (seed 42, deployed epoch) of every guard-passing cand."""
    if ctx.smoke:
        c = replace(cands[-1], stop_epoch=1)
        runs = plan(ctx, c, [ctx.cfg["dev_classes"][0]], [], [42], "trackb_improve")
        run_all(ctx, runs)
        _print_smoke_scores(ctx, runs[0].run_id)
        return
    for c in passing(ctx, cands):
        runs = plan(
            ctx, c, ctx.cfg["dev_classes"], [], [int(ctx.cfg["loco_seed"])], "trackb_improve"
        )
        run_all(ctx, runs)


# =============================================================================== C3 fit check
def _fit_attempt(
    ctx: Ctx, micro: int, accum: int, texts: list[str], y: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """3 real AdamW steps of e5-large at batch micro*accum; OOM or spill-to-host => not fitting."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    c3 = ctx.cfg["c3"]
    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    rec: dict[str, Any] = {"micro_batch": micro, "accum": accum, "step_s": []}
    model = optim = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        tok, model = build_model(c3["model_name"], labels, "eager")
        model.to(dev).train()
        optim = torch.optim.AdamW(model.parameters(), lr=float(c3["fit_check_lr"]))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        bs = micro * accum
        for step in range(int(c3["fit_check_steps"])):
            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            lo = step * bs
            for a in range(accum):
                s = lo + a * micro
                enc = tok(
                    [QUERY_PREFIX.get(c3["model_name"], "") + t for t in texts[s : s + micro]],
                    padding=True, truncation=True, max_length=int(c3["max_len"]),
                    return_tensors="pt",
                ).to(dev)  # fmt: skip
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits
                loss = (
                    F.cross_entropy(logits.float(), torch.as_tensor(y[s : s + micro], device=dev))
                    / accum
                )
                scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()
            rec["step_s"].append(time.perf_counter() - t0)
        rec["outcome"] = "completed"
    except torch.cuda.OutOfMemoryError as exc:
        rec["outcome"] = "oom"
        rec["error"] = str(exc)[:600]
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        rec["outcome"] = "oom"
        rec["error"] = str(exc)[:600]
    finally:
        rec["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        rec["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        del model, optim
        _free_gpu()
    # Windows' "CUDA sysmem fallback" can spill past VRAM instead of raising: allocated > total
    # means the step did not fit in device memory whatever the outcome says.
    rec["spilled_past_vram"] = bool(rec["max_memory_allocated_mb"] * 2**20 > total)
    rec["fits"] = rec["outcome"] == "completed" and not rec["spilled_past_vram"]
    return rec


def stage_c3_fit_check(ctx: Ctx) -> None:
    """Pre-registered abort check: does one e5-large training step fit in 8 GB? Writes the json."""
    import torch

    c3 = ctx.cfg["c3"]
    out_path = ctx.P.results / "c3_fit_check.json"
    labels = list(data_mod.LABELS)
    train = ctx.df_all[ctx.df_all["split"] == "train"].sort_values("id").reset_index(drop=True)
    need = int(c3["fit_check_steps"]) * int(c3["batch_size"])
    texts = train["text"].tolist()[:need]
    y = _local_gold(train, labels)[:need]
    attempts: list[dict[str, Any]] = []
    with gpu_section(900, "intent-router trackB-improve c3_fit_check", 30):
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory / 2**20
        for micro, accum in c3["accum_fallbacks"]:
            if micro * accum != int(c3["batch_size"]):
                raise ValueError(f"micro*accum must equal batch_size: {micro}x{accum}")
            att = _fit_attempt(ctx, int(micro), int(accum), texts, y, labels)
            attempts.append(att)
            print(f"[c3_fit_check] {micro}x{accum}: {att['outcome']} fits={att['fits']} "
                  f"peak alloc {att['max_memory_allocated_mb']:.0f} MB", flush=True)  # fmt: skip
            if att["fits"]:
                break
        _, probe = build_model(c3["model_name"], labels, "eager")
        n_params = int(sum(p.numel() for p in probe.parameters()))
        del probe
    chosen = next((a for a in attempts if a["fits"]), None)
    mb = 2**20
    arith = {
        "n_params": n_params,
        "weights_fp32_mb": 4 * n_params / mb,
        "grads_fp32_mb": 4 * n_params / mb,
        "adamw_state_fp32_mb": 8 * n_params / mb,
        "static_total_mb": 16 * n_params / mb,
        "device_total_mb": total_mb,
        "note": "static = weights + grads + 2 AdamW moments, all fp32; gradient accumulation "
        "shrinks activations only, NOT these; fp16 autocast adds weight-cast copies on top",
    }
    out = {
        "label": "measured, 3 real AdamW steps, fp32 weights + fp16 autocast, max_len 64",
        "model_name": c3["model_name"], "gpu": props.name, "vram_budget_gb": c3["vram_budget_gb"],
        "arithmetic": arith, "attempts": attempts,
        "chosen": None if chosen is None else {"micro_batch": chosen["micro_batch"],
                                                "accum": chosen["accum"]},
        "aborted": chosen is None,
        "decision": "C3 aborted per the rule fixed beforehand (a step does not fit in 8 GB)"
        if chosen is None else "fits: C3 may proceed",
    }  # fmt: skip
    write_json(out_path, out)
    print(f"[c3_fit_check] aborted={out['aborted']} -> {out_path}")


# ================================================================================= C3 latency
def stage_c3_latency(ctx: Ctx) -> None:
    """CPU batch-1 latency p50/p95 over the val texts (1 thread and default threads) + resources.

    Latency depends on architecture, not weight values, so freshly initialised 12-way heads on the
    pretrained encoders are used; training wall-clock / VRAM come from the trained run jsons.
    """
    import torch

    c3_accum(ctx)  # stops if C3 was aborted
    c3 = ctx.cfg["c3"]
    labels = list(data_mod.LABELS)
    val = ctx.df_all[ctx.df_all["split"] == "val"].sort_values("id")["text"].tolist()
    default_threads = torch.get_num_threads()
    out: dict[str, Any] = {
        "label": "measured, CPU, batch 1, fp32, eager, val texts",
        "n_texts": len(val),
        "default_threads": default_threads,
        "models": {},
    }
    for name in (ctx.base_model, c3["model_name"]):
        tok, model = build_model(name, labels, "eager")
        model.eval()
        per: dict[str, Any] = {}
        for tag, n_thr in (("1_thread", 1), ("default_threads", default_threads)):
            torch.set_num_threads(n_thr)
            lat: list[float] = []
            with torch.no_grad():
                for _ in range(int(c3["latency_repeats"])):
                    for t in val:
                        enc = tok(QUERY_PREFIX.get(name, "") + t, return_tensors="pt",
                                  truncation=True, max_length=int(c3["max_len"]))  # fmt: skip
                        t0 = time.perf_counter()
                        model(**enc)
                        lat.append((time.perf_counter() - t0) * 1000)
            per[tag] = {"threads": n_thr, "p50_ms": float(np.percentile(lat, 50)),
                        "p95_ms": float(np.percentile(lat, 95))}  # fmt: skip
        torch.set_num_threads(default_threads)
        out["models"][name] = {"cpu_latency": per}
        del model
        gc.collect()
    runs_dir = ctx.P.runs
    for cand_name, prefix in [
        (ctx.base_model, "loco_"),
        *[(c.key, c.cid) for c in c3_candidates(ctx)],
    ]:
        ms = []
        for p in sorted(runs_dir.glob(f"{prefix}*_s42.json")):
            ms.append(json.loads(p.read_text(encoding="utf-8"))["training"])
        if ms:
            out.setdefault("training_resources", {})[cand_name] = {
                "n_runs": len(ms),
                "median_wall_clock_s": float(np.median([m["wall_clock_s"] for m in ms])),
                "max_peak_vram_mb": float(max(m["peak_vram_mb"] for m in ms)),
            }
    write_json(ctx.P.results / "c3_resources.json", out)
    print(json.dumps(out["models"], indent=1))


# ========================================================================== C4 ensemble + B1
def ensemble_table(
    tables: Sequence[pd.DataFrame], arrs: Sequence[Mapping[str, np.ndarray]], labels: list[str]
) -> pd.DataFrame:
    """ens_maha = mean of per-seed maha_ft; ens_entropy = -entropy of the mean softmax."""
    first = tables[0]
    key = ["id", "set", "is_unknown", "gold"]
    for t in tables[1:]:
        if not t[key].equals(first[key]):
            raise AssertionError("per-seed score tables are not row-aligned")
    ids = np.concatenate([arrs[0][f"{p}_ids"] for p in ("cal", "ek", "unk")])
    if not np.array_equal(ids, first["id"].to_numpy(dtype=str)):
        raise AssertionError("npz rows differ from score-table rows")
    probs = np.mean(
        [
            softmax(np.concatenate([a[f"{p}_logits"] for p in ("cal", "ek", "unk")])).astype(
                np.float64
            )
            for a in arrs
        ],
        axis=0,
    )
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
    out = first[["id", "split", "set", "is_unknown", "gold"]].copy()
    out["pred"] = [labels[i] for i in probs.argmax(1)]
    out["ens_maha"] = np.mean([t["maha_ft"].to_numpy(np.float64) for t in tables], axis=0)
    out["ens_entropy"] = -ent
    return out


def stage_c4(ctx: Ctx) -> None:
    """Base model seeds 42-46 for every LOCO class + the headline holdout; write ensemble CSVs."""
    base = base_candidate(ctx, "c4")
    seeds = [int(s) for s in ctx.cfg["c4"]["seeds"]]
    p3 = [int(ctx.cfg["loco_seed"])]
    need = [run_id_for(base, "loco", p3[0], c) for c in ctx.loco_classes] + [
        run_id_for(base, "headline", s, None) for s in ctx.cfg["headline_seeds"]
    ]
    missing = [r for r in need if not (ctx.P.scores / f"{r}.csv").exists()]
    if missing:
        raise SystemExit(
            f"run --stage reproduce first (seed-42 / headline arrays reused): {missing}"
        )
    loco_extra = [s for s in seeds if s not in p3]
    head_extra = [s for s in seeds if s not in ctx.cfg["headline_seeds"]]
    run_all(ctx, plan(ctx, base, ctx.loco_classes, head_extra, loco_extra, "trackb_improve"))
    units: list[tuple[str, tuple[str, ...], list[str]]] = [
        (f"c4_loco_{c}_ens", (c,), [run_id_for(base, "loco", s, c) for s in seeds])
        for c in ctx.loco_classes
    ]
    units.append(("c4_headline_ens", ctx.headline_holdout,
                  [run_id_for(base, "headline", s) for s in seeds]))  # fmt: skip
    for out_id, holdout, rids in units:
        path = ctx.P.scores / f"{out_id}.csv"
        if path.exists():
            continue
        tables = [read_scores(ctx, r) for r in rids]
        arrs = [dict(np.load(ctx.P.outputs / "runs" / f"{r}.npz")) for r in rids]
        _save_csv(ensemble_table(tables, arrs, labels_for(ctx, holdout)), path)
        print(f"[c4] wrote {path.name} from seeds {seeds}")


def stage_b1(ctx: Ctx) -> None:
    """Frozen e5 + class-balanced LR fitted on train-known rows: MSP and energy per holdout."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    from intent_router.baselines import _macro_f1_score

    bc = ctx.fcfg["baselines"]
    fz_of = ctx.frozen_of()
    specs = [("b1_headline", ctx.headline_holdout)] + [
        (f"b1_loco_{c}", (c,))
        for c in (ctx.cfg["dev_classes"][:1] if ctx.smoke else ctx.loco_classes)
    ]
    for run_id, holdout in specs:
        path = ctx.P.scores / f"{run_id}.csv"
        if path.exists():
            continue
        sets = build_holdout_sets(ctx.df_all, list(holdout), ctx.smoke)
        labels, K = sets.labels, len(sets.labels)

        def fz(frame: pd.DataFrame) -> np.ndarray:
            return np.stack([fz_of[i] for i in frame["id"]])

        gs = GridSearchCV(
            LogisticRegression(class_weight="balanced", max_iter=5000),
            {"C": list(bc["c_grid"])},
            scoring=make_scorer(_macro_f1_score, n_classes=K),
            cv=StratifiedGroupKFold(int(bc["inner_folds"]), shuffle=True, random_state=42),
            n_jobs=1, refit=True,
        )  # fmt: skip
        gs.fit(fz(sets.train), _local_gold(sets.train, labels), groups=sets.train["dup_group"])
        clf = gs.best_estimator_
        log_test_call(
            ctx.P.test_log, f"{run_id}:{gs.best_params_['C']}", "trackb_b1",
            {"run_id": run_id, "role": "b1-lr-on-frozen-e5", "n_rows": len(sets.eval_known)},
        )  # fmt: skip

        def part(
            frame: pd.DataFrame, clf: Any = clf
        ) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
            dec = clf.decision_function(fz(frame))  # multinomial logits
            return (
                frame,
                dec.argmax(1),
                {"b1_msp": score_msp(dec), "b1_neg_energy": score_neg_energy(dec)},
            )

        parts = {"cal": part(sets.cal), "eval_known": part(sets.eval_known),
                 "eval_unknown": part(sets.eval_unknown)}  # fmt: skip
        table = _scores_table(
            sets, parts, {"cal": False, "eval_known": False, "eval_unknown": True}
        )
        _save_csv(table, path)
        print(f"[b1] {run_id}: C={gs.best_params_['C']} "
              f"AUROC msp {table_auroc(table, 'b1_msp'):.4f}", flush=True)  # fmt: skip


# ===================================================================================== select
def _candidate_tables(ctx: Ctx, cand: Candidate, classes: Sequence[str]) -> dict[str, pd.DataFrame]:
    return {
        c: read_scores(ctx, run_id_for(cand, "loco", int(ctx.cfg["loco_seed"]), c)) for c in classes
    }


def stage_select(ctx: Ctx) -> None:
    """DEV-only selection. Only DEV class files are ever opened here."""
    dev = list(ctx.cfg["dev_classes"])
    if set(dev) & set(ctx.cfg["confirm_classes"]):
        raise AssertionError("DEV and CONFIRM classes overlap")
    dev_auroc: dict[str, dict[str, float]] = {}
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    base_tabs = _candidate_tables(ctx, base_candidate(ctx), dev)
    for m in ctx.all_methods:
        k = f"base/{m}"
        dev_auroc[k] = {c: table_auroc(t, m) for c, t in base_tabs.items()}
        if m in C1_METHODS:
            eligible.append(k)
    for cand in [*c2_candidates(ctx), *(c3_candidates(ctx) if _c3_alive(ctx) else [])]:
        g = _read_json(guard_path(ctx, cand.key))
        if g is None:
            raise SystemExit(f"no guard summary for {cand.key}")
        if not g["passed"]:
            excluded[cand.key] = g["why"]
            continue
        tabs = _candidate_tables(ctx, cand, dev)
        for m in ctx.all_methods:
            k = f"{cand.key}/{m}"
            dev_auroc[k] = {c: table_auroc(t, m) for c, t in tabs.items()}
            eligible.append(k)
    report_only: list[str] = []
    for tag, methods in (("c4", ("ens_maha", "ens_entropy")), ("b1", ("b1_msp", "b1_neg_energy"))):
        for m in methods:
            files = {
                c: ctx.P.scores / (f"c4_loco_{c}_ens.csv" if tag == "c4" else f"b1_loco_{c}.csv")
                for c in dev
            }
            if all(p.exists() for p in files.values()):
                dev_auroc[f"{tag}/{m}"] = {
                    c: table_auroc(pd.read_csv(p), m) for c, p in files.items()
                }
                report_only.append(f"{tag}/{m}")
    out = select_on_dev(dev_auroc, eligible, CURRENT_KEY, dev)
    out |= {
        "label": "measured on DEV classes only",
        "dev_per_class_auroc": dev_auroc,
        "eligible": eligible,
        "report_only_not_eligible": report_only,
        "excluded_by_guard": excluded,
        "not_eligible_note": "C4 (evidence only), B1 (fairness fix) and open-set evaluation "
        "methods on base "
        "are reported, never selected",
    }
    write_json(ctx.P.results / "select.json", out)
    print(
        f"[select] winner {out['winner']} DEV {out['winner_dev_mean']:.4f} vs current "
        f"{out['current_dev_mean']:.4f}; margin {out['margin_std_ddof1_over_sqrt_n']:.4f}; "
        f"ship={out['ship']} -> shipped {out['shipped']}"
    )


def _c3_alive(ctx: Ctx) -> bool:
    chk = _read_json(ctx.P.results / "c3_fit_check.json")
    return bool(chk) and not chk["aborted"]


# ==================================================================================== confirm
def _cand_by_key(ctx: Ctx, key: str) -> Candidate:
    if key == "base":
        return base_candidate(ctx)
    pool = [*c2_candidates(ctx), *(c3_candidates(ctx) if _c3_alive(ctx) else [])]
    hit = [c for c in pool if c.key == key]
    if not hit:
        raise KeyError(f"unknown candidate {key!r}")
    return hit[0]


def _unit_summary(table: pd.DataFrame, method: str, labels: list[str], ctx: Ctx) -> dict[str, Any]:
    ood = ctx.cfg["ood"]
    safe = tuple(ood["safe_labels"])
    rec: dict[str, Any] = {}
    for r in ood["op_retentions"]:
        m = run_metrics(table, [method], labels, float(r), safe)["methods"][method]
        tag = _tag(r)
        if not rec:
            rec |= {"auroc": m["auroc"], "aupr": m["aupr"], "fpr_at_95tpr": m["fpr_at_95tpr"]}
        rec[f"op{tag}"] = {
            "strict_rejection_recall": m["strict_rejection_recall"],
            "lenient_rejection_recall": m["lenient_rejection_recall"],
            "retention_known": m["retention_known"],
            "cal_retention_achieved": m["cal_retention_achieved"],
        }
    return rec


def _mean_rec(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in recs[0]:
        if isinstance(recs[0][k], dict):
            out[k] = _mean_rec([r[k] for r in recs])
        else:
            out[k] = float(np.mean([r[k] for r in recs]))
    return out


def confirm_entry(ctx: Ctx, cand: Candidate, method: str) -> dict[str, Any]:
    """CONFIRM LOCO classes + headline seeds for one (candidate, method): metrics, curves, CIs."""
    cfg = ctx.cfg
    conf = list(cfg["confirm_classes"])
    ood, bs = cfg["ood"], cfg["bootstrap"]
    rets = [float(r) for r in ood["op_retentions"]]
    cv = ood["curve"]
    targets = [round(float(x), 2) for x in np.arange(cv["lo"], cv["hi"] + 1e-9, cv["step"])]
    seed = int(cfg["loco_seed"])
    entry: dict[str, Any] = {"candidate": cand.key, "method": method}
    for name, tabs, labs, shared in (
        ("confirm", {c: read_scores(ctx, run_id_for(cand, "loco", seed, c)) for c in conf},
         {c: labels_for(ctx, (c,)) for c in conf}, False),
        ("headline", {f"s{s}": read_scores(ctx, run_id_for(cand, "headline", s))
                      for s in cfg["headline_seeds"]},
         {f"s{s}": labels_for(ctx, ctx.headline_holdout) for s in cfg["headline_seeds"]}, True),
    ):  # fmt: skip
        per = {k: _unit_summary(t, method, labs[k], ctx) for k, t in tabs.items()}
        curves = []
        for t in tabs.values():
            cal = t[t["set"] == "cal"][method].to_numpy()
            ev = t[t["set"] == "eval"]
            unk = ev["is_unknown"].astype(bool)
            curves.append(rejection_retention_curve(
                cal, ev.loc[~unk, method].to_numpy(), ev.loc[unk, method].to_numpy(), targets
            ))  # fmt: skip
        boot = bootstrap_mean_ci(
            [boot_unit(t, method, rets, ood["safe_labels"]) for t in tabs.values()],
            int(bs["n_resamples"]), int(bs["seed"]), shared, float(bs["level"]),
        )  # fmt: skip
        entry[name] = {
            "per_unit": per,
            "mean": _mean_rec(list(per.values())),
            "ci95": {m: {"point": boot["point"][m], "lo": boot["lo"][m], "hi": boot["hi"][m]}
                     for m in boot["point"]},
            "curve_mean": {
                "target": targets,
                "retention_known": np.mean([c["retention_known"] for c in curves], axis=0).tolist(),
                "strict_rejection_recall": np.mean(
                    [c["strict_rejection_recall"] for c in curves], axis=0
                ).tolist(),
            },
            "bootstrap": LABEL_BOOTSTRAP + f"; {bs['n_resamples']} resamples, seed {bs['seed']}",
        }  # fmt: skip
        entry[name]["_samples"] = boot["samples"]
    return entry


def stage_confirm(ctx: Ctx) -> None:
    """CONFIRM + headline report for the DEV winner and the current method."""
    sel = _read_json(ctx.P.results / "select.json")
    if sel is None:
        raise SystemExit("run --stage select first")
    cfg = ctx.cfg
    entries_spec = {CURRENT_KEY: ("base", "maha_ft")}
    if sel["winner"] is not None:
        cand_key, method = sel["winner"].split("/", 1)
        entries_spec[sel["winner"]] = (cand_key, method)
        if cand_key != "base":  # needs its own CONFIRM LOCO models + headline seeds
            cand = _cand_by_key(ctx, cand_key)
            run_all(
                ctx,
                plan(ctx, cand, cfg["confirm_classes"], cfg["headline_seeds"],
                     [int(cfg["loco_seed"])], "trackb_improve"),
            )  # fmt: skip
    entries: dict[str, Any] = {}
    for key, (ck, m) in entries_spec.items():
        entries[key] = confirm_entry(ctx, _cand_by_key(ctx, ck), m)
    deltas: dict[str, Any] = {}
    if sel["winner"] is not None and sel["winner"] != CURRENT_KEY:
        for part in ("confirm", "headline"):
            deltas[part] = paired_delta_ci(
                entries[sel["winner"]][part]["_samples"],
                entries[CURRENT_KEY][part]["_samples"],
                float(cfg["bootstrap"]["level"]),
            )
    for e in entries.values():
        for part in ("confirm", "headline"):
            e[part].pop("_samples")
    report_only = _confirm_report_only(ctx)
    out = {
        "label": "measured, CONFIRM LOCO classes (5, seed 42) and headline holdout (seeds "
        f"{cfg['headline_seeds']}); thresholds from calibration-known rows",
        "confirm_classes": cfg["confirm_classes"],
        "select_summary": {
            k: sel[k]
            for k in (
                "winner",
                "ship",
                "shipped",
                "winner_dev_mean",
                "current_dev_mean",
                "winner_minus_current",
                "margin_std_ddof1_over_sqrt_n",
            )
        },  # fmt: skip
        "shipped": sel["shipped"],
        "winner_note": "winner = DEV-best eligible; it is SHIPPED only if select.ship is true",
        "entries": entries,
        "paired_delta_winner_minus_current": deltas,
        "report_only": report_only,
        "ci_note": "AUPR has a point estimate only; resampled: AUROC, FPR@95TPR, recall, retention",
    }
    write_json(ctx.P.results / "confirm.json", out)
    _confirm_figure(ctx, out)
    print(f"[confirm] wrote confirm.json; shipped = {sel['shipped']}")


def _confirm_report_only(ctx: Ctx) -> dict[str, Any]:
    """Mean CONFIRM AUROC of every other method and of C4 / B1 (no claims, no CIs)."""
    conf = list(ctx.cfg["confirm_classes"])
    out: dict[str, Any] = {}
    base_tabs = _candidate_tables(ctx, base_candidate(ctx), conf)
    for m in ctx.all_methods:
        out[f"base/{m}"] = float(np.mean([table_auroc(t, m) for t in base_tabs.values()]))
    for tag, methods in (("c4", ("ens_maha", "ens_entropy")), ("b1", ("b1_msp", "b1_neg_energy"))):
        for m in methods:
            ps = [
                ctx.P.scores / (f"c4_loco_{c}_ens.csv" if tag == "c4" else f"b1_loco_{c}.csv")
                for c in conf
            ]
            if all(p.exists() for p in ps):
                out[f"{tag}/{m}"] = float(np.mean([table_auroc(pd.read_csv(p), m) for p in ps]))
    return {"confirm_mean_auroc": out, "label": "report only (C4 / B1 are not eligible)"}


def _confirm_figure(ctx: Ctx, out: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, part in zip(axes, ("confirm", "headline"), strict=True):
        for key, e in out["entries"].items():
            c = e[part]["curve_mean"]
            ax.plot(c["retention_known"], c["strict_rejection_recall"], marker=".", label=key)
        ax.set_title(f"{part} (mean over {'classes' if part == 'confirm' else 'seeds'})")
        ax.set_xlabel("known retention (measured)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("strict rejection recall")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path = ctx.P.results / "confirm_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =================================================================================== business
def stage_business(ctx: Ctx) -> None:
    """Per-1,000-message estimate from the SHIPPED method's measured CONFIRM-mean rates."""
    conf = _read_json(ctx.P.results / "confirm.json")
    if conf is None:
        raise SystemExit("run --stage confirm first")
    b = ctx.cfg["business"]
    shipped = conf["shipped"]
    entry = conf["entries"][shipped]["confirm"]["mean"]
    rows = []
    for r in b["retention"]:
        op = entry[f"op{_tag(float(r))}"]
        for pi in b["prevalence"]:
            row = business_row(
                int(b["messages"]), float(pi), op["retention_known"],
                op["strict_rejection_recall"], op["lenient_rejection_recall"],
            )  # fmt: skip
            rows.append({"calibration_retention_target": float(r), **row})
    out = {
        "label": "estimate from measured CONFIRM rates",
        "shipped_method": shipped,
        "per_messages": int(b["messages"]),
        "formulas": {
            "known_wrongly_abstained": "N(1-pi)(1-retention_measured)",
            "unknowns_caught": "N*pi*rejection_recall",
            "unknowns_misrouted": "N*pi*(1-rejection_recall)",
        },
        "measured_confirm_mean": {k: v for k, v in entry.items() if k.startswith("op")},
        "rows": rows,
    }
    write_json(ctx.P.results / "business.json", out)
    print(f"[business] {len(rows)} rows for {shipped}")


# =================================================================================== final_ood
def stage_final_ood(ctx: Ctx) -> None:
    """Apply the shipped C1 method to the final 12-class model (one logged feature extraction)."""
    out_path = ctx.P.results / "final_ood.json"
    sel = _read_json(ctx.P.results / "select.json")
    if sel is None:
        raise SystemExit("run --stage select first")
    shipped = sel["shipped"]
    cand_key, method = shipped.split("/", 1)
    if cand_key != "base" or method not in C1_METHODS:
        todo = (
            "TODO(orchestrator): the shipped method changes model weights; train the final model "
            "once with this recipe, run its single logged test evaluation and report both models' "
            "test results with disclosure"
            if cand_key != "base"
            else None
        )
        write_json(out_path, {
            "status": "todo" if todo else "not_applicable",
            "shipped": shipped,
            "note": todo or "shipped method is unchanged (open-set evaluation maha_ft): nothing to "
                "extract",
        })  # fmt: skip
        print(f"[final_ood] {shipped}: wrote status only")
        return
    if out_path.exists():
        print("[final_ood] final_ood.json exists, skipping")
        return
    cfg, fcfg = ctx.cfg, ctx.fcfg
    log_path = Path(cfg["final_test_log"])
    if log_path.exists() and any(
        json.loads(x).get("call_type") == "ood_feature_extraction"
        for x in log_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ):
        raise SystemExit("an ood_feature_extraction call is already logged: refusing a second one")
    labels = list(data_mod.LABELS)
    train = ctx.df_all[ctx.df_all["split"] == "train"].sort_values("id").reset_index(drop=True)
    val = ctx.df_all[ctx.df_all["split"] == "val"].sort_values("id").reset_index(drop=True)
    test = ctx.df_all[ctx.df_all["split"] == "test"].sort_values("id").reset_index(drop=True)
    layers = [int(x) for x in cfg["c1"]["layers"]]
    bs, ml = int(fcfg["predict_batch_size"]), int(fcfg["train"]["max_len"])
    with gpu_section(900, "intent-router trackB-improve final_ood", float(cfg["gpu_lock_poll_s"])):
        tok, model = build_model(str(cfg["final_model_dir"]), labels, "eager")
        import torch

        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        fp = state_dict_sha256(model)
        tr = predict_hidden(model, tok, train["text"].tolist(), ctx.base_model, ml, bs, layers)
        va = predict_hidden(model, tok, val["text"].tolist(), ctx.base_model, ml, bs, layers)
        log_test_call(
            log_path, fp, "ood_feature_extraction",
            {"role": "trackB-improve final_ood", "method": method, "n_rows": len(test)},
        )  # fmt: skip
        te = predict_hidden(model, tok, test["text"].tolist(), ctx.base_model, ml, bs, layers)
        del model
        _free_gpu()
    ts_path = cfg["final_artifacts"].get("train_summary")
    summary = _read_json(Path(ts_path)) if ts_path else None
    saved_pred = pd.read_csv(cfg["final_artifacts"]["test_predictions"])
    same_ids = saved_pred["id"].tolist() == test["id"].tolist()
    pred_equal = bool(same_ids and np.array_equal(saved_pred["pred"].to_numpy(), te[0].argmax(1)))
    if not pred_equal:
        raise AssertionError("test predictions differ from results/final/test_predictions.csv")
    z = np.load(cfg["final_artifacts"]["features_logits"])
    l2i = {lab: i for i, lab in enumerate(labels)}
    y_tr, y_val, y_te = (_local_gold(f, labels) for f in (train, val, test))
    ml_model = fit_multilayer_maha(tr[2], y_tr, len(labels), va[2])
    g_ft = fit_gaussian_lw(tr[1], y_tr, len(labels))
    names = list(cfg["c1"]["fusion_components"])

    def comps(lg: np.ndarray, ft: np.ndarray) -> dict[str, np.ndarray]:
        return {"maha_ft": score_mahalanobis(ft, g_ft), "neg_energy": score_neg_energy(lg)}

    val_c, te_c = comps(va[0], va[1]), comps(te[0], te[1])
    fusion = fit_rank_fusion([val_c[n] for n in names])
    val_s = {"maha_ml": ml_model.score(va[2]),
             "fuse_maha_energy": fusion.transform([val_c[n] for n in names])}  # fmt: skip
    te_s = {"maha_ml": ml_model.score(te[2]),
            "fuse_maha_energy": fusion.transform([te_c[n] for n in names])}  # fmt: skip
    retention = float(ctx.tb["ood"]["retention"])
    thr = threshold_at_retention(val_s[method], retention)
    acc = te_s[method] >= thr
    te_pred = te[0].argmax(1)
    out = {
        "status": "done", "method": method, "label": "final 12-class model, OOD method only; "
        "no change to test predictions (asserted equal to results/final/test_predictions.csv)",
        "model_fingerprint": fp,
        "fingerprint_matches_final_train_summary": None
        if summary is None
        else summary.get("model_fingerprint") == fp, "test_predictions_match_saved_csv": pred_equal,
        "max_abs_diff_vs_saved": {
            "val_logits": float(np.max(np.abs(va[0] - z["val_logits"]))),
            "val_features": float(np.max(np.abs(va[1] - z["val_features"]))),
            "train_features": float(np.max(np.abs(tr[1] - z["train_features"]))),
        },
        "layers": layers, "threshold": thr,
        "threshold_source": f"{retention:.0%} retention on val (all 12 classes known)",
        "val_retention_achieved": float(np.mean(val_s[method] >= thr)),
        "n_test": len(y_te), "n_test_accepted": int(acc.sum()),
        "test_coverage_at_threshold": float(acc.mean()),
        "test_accepted_macro_f1_present": macro_f1_present(y_te[acc], te_pred[acc], len(labels))
        if acc.any() else None,
        "test_accepted_accuracy": float(np.mean(y_te[acc] == te_pred[acc])) if acc.any() else None,
        "test_all_accuracy": float(np.mean(y_te == te_pred)),
        "test_call": "ONE logged call, call_type ood_feature_extraction, in " + str(log_path),
        "n_labels": len(l2i),
        "y_val_n": len(y_val),
    }  # fmt: skip
    write_json(out_path, out)
    print(
        f"[final_ood] {method}: thr {thr:.4f} test coverage {out['test_coverage_at_threshold']:.3f}"
    )


# ======================================================================================= main
def load_ctx(args: argparse.Namespace) -> Ctx:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fcfg = yaml.safe_load(Path(cfg["final_config"]).read_text(encoding="utf-8"))
    tb = yaml.safe_load(Path(cfg["trackb_config"]).read_text(encoding="utf-8"))
    if args.poll_s is not None:
        cfg["gpu_lock_poll_s"] = args.poll_s
    use_wandb = bool(cfg["wandb"]["enabled"]) and not args.no_wandb and not args.smoke
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    P = make_paths(cfg, args.smoke)
    data_mod.load_data(fcfg["data_path"])
    df = data_mod.get_frame(["train", "val", "test"], fcfg["data_path"], fcfg["splits_path"])
    if args.smoke:
        df = df[df["split"] != "test"].reset_index(drop=True)
        if (df["split"] == "test").any():
            raise AssertionError("smoke frame must not contain test rows")
    ctx = Ctx(cfg, fcfg, tb, P, df, use_wandb, args.smoke)
    expected = sorted(ctx.loco_classes)
    if sorted([*cfg["dev_classes"], *cfg["confirm_classes"]]) != expected:
        raise AssertionError("dev_classes + confirm_classes must be exactly the 10 LOCO classes")
    return ctx


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Open-set scorer comparison: Track B improvement (pre-registered)"
    )
    ap.add_argument("--config", default="configs/trackb_improve.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--smoke", action="store_true", help="tiny: stop_epoch 1, no test rows")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--poll-s", type=float, default=None)
    args = ap.parse_args(argv)
    if args.smoke and args.stage not in ("reproduce", "c2_guard", "c2_dev", "b1_ood"):
        raise SystemExit("--smoke supports reproduce, c2_guard, c2_dev, b1_ood only")
    ctx = load_ctx(args)
    st = args.stage
    if st == "reproduce":
        stage_reproduce(ctx)
    elif st == "c2_guard":
        stage_guard(ctx, c2_candidates(ctx))
    elif st == "c2_dev":
        stage_dev(ctx, c2_candidates(ctx))
    elif st == "c3_fit_check":
        stage_c3_fit_check(ctx)
    elif st == "c3_guard":
        stage_guard(ctx, c3_candidates(ctx))
    elif st == "c3_dev":
        stage_dev(ctx, c3_candidates(ctx))
    elif st == "c3_latency":
        stage_c3_latency(ctx)
    elif st == "c4_dev_confirm_headline":
        stage_c4(ctx)
    elif st == "b1_ood":
        stage_b1(ctx)
    elif st == "select":
        stage_select(ctx)
    elif st == "confirm":
        stage_confirm(ctx)
    elif st == "business":
        stage_business(ctx)
    elif st == "final_ood":
        stage_final_ood(ctx)


if __name__ == "__main__":
    main()
