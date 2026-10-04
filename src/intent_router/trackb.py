from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.baselines import embed_e5, embed_texts
from intent_router.evaluate import (
    evaluate_test_once,
    git_sha,
    macro_f1_present,
    risk_coverage,
)
from intent_router.final import write_json
from intent_router.models import QUERY_PREFIX, predict, prepare_text, state_dict_sha256
from intent_router.ood import (
    NON_LOCO_LABELS,
    HoldoutSets,
    OodScorer,
    build_holdout_sets,
    fit_scorer,
    mean_std,
    method_names,
    run_metrics,
    threshold_at_retention,
)
from intent_router.train import TrainConfig, train_model

STAGES = ("headline", "loco", "ls", "shipped", "report", "all")
SELECTION_NOTE = "a design choice made across holdout experiments"
SMOKE_LABEL = "SMOKE: known side evaluated on calibration rows, NOT a test result"
# Metrics aggregated across runs (all are per-method keys of ood.run_metrics()).
AGG_KEYS = (
    "auroc",
    "aupr",
    "fpr_at_95tpr",
    "aurc_open_set",
    "retention_known",
    "strict_rejection_recall",
    "lenient_rejection_recall",
    "known_accepted_routed_to_safe_frac_of_known",
    "known_accepted_routed_to_safe_gold_not_safe_frac_of_known",
    "known_accepted_macro_f1_present",
    "known_accepted_accuracy",
    "cal_retention_achieved",
)
LS_KEYS = (
    "auroc",
    "aupr",
    "fpr_at_95tpr",
    "strict_rejection_recall",
    "lenient_rejection_recall",
    "retention_known",
)
FIT_SOURCES = {
    "model_weights": "train split minus held-out classes; per-epoch curves on the calibration "
    "rows; no epoch/checkpoint selection (stop_epoch fixed by the bake-off CV)",
    "temperature": "calibration rows (val split minus held-out classes): NLL, LBFGS",
    "threshold_per_method": "calibration rows (known only), 95% retention",
    "mahalanobis_ft": "class means + Ledoit-Wolf covariance: train minus held-out, fine-tuned "
    "penultimate features",
    "knn_ft_bank": "train minus held-out, fine-tuned penultimate features",
    "mahalanobis_frozen": "class means + Ledoit-Wolf covariance: train minus held-out, frozen "
    "e5 features",
    "knn_frozen_bank": "train minus held-out, frozen e5 features",
    "knn_k": "fixed {1, 5}, reported, not tuned",
    "design_note": "lr / stop_epoch / model were chosen in bake-off by 12-class CV on train+val "
    "rows, which includes the held-out classes as KNOWN classes; no unknown-class data is used "
    "by any fitted quantity of a Track B run, but those design choices pre-date the holdouts",
}


# ------------------------------------------------------------------------- paths
@dataclass(frozen=True)
class Paths:
    """Where one invocation writes; smoke runs are fully redirected under outputs/."""

    results: Path
    figures: Path
    outputs: Path
    frozen_cache: Path
    test_log: Path
    smoke: bool


def make_paths(cfg: dict[str, Any], smoke: bool) -> Paths:
    """Real paths from the config, or a self-contained sandbox under outputs/trackb_smoke."""
    if smoke:
        root = Path("outputs/trackb_smoke")
        return Paths(
            root / "results", root / "figures", root / "outputs", root / "frozen_e5.npz",
            root / "test_inference_log.jsonl", True,
        )  # fmt: skip
    return Paths(
        Path(cfg["results_dir"]),
        Path(cfg["figures_dir"]),
        Path(cfg["outputs_dir"]),
        Path(cfg["frozen_cache"]),
        Path(cfg["test_inference_log"]),
        False,
    )


# ---------------------------------------------------------------------- run plan
@dataclass(frozen=True)
class RunSpec:
    """One Track B model: which classes are held out, seed, label smoothing, W&B group."""

    run_id: str
    kind: str  # headline | loco | ls
    holdout: tuple[str, ...]
    seed: int
    label_smoothing: float
    group: str


def plan_runs(cfg: dict[str, Any], labels: list[str], kind: str) -> list[RunSpec]:
    """The runs of one stage: headline (3 seeds), loco (10 classes, 1 seed), ls (3 seeds)."""
    wb = cfg["wandb"]
    head = tuple(sorted(cfg["headline"]["holdout"]))
    if kind == "headline":
        return [
            RunSpec(f"headline_s{s}", kind, head, s, 0.0, wb["group_headline"])
            for s in cfg["headline"]["seeds"]
        ]
    if kind == "ls":
        ls = float(cfg["ls"]["label_smoothing"])
        return [
            RunSpec(f"ls{ls}_s{s}".replace(".", "p"), kind, head, s, ls, wb["group_ls"])
            for s in cfg["ls"]["seeds"]
        ]
    if kind == "loco":
        excl = set(cfg["loco"]["excluded_classes"]) | set(NON_LOCO_LABELS)
        seed = int(cfg["loco"]["seed"])
        return [
            RunSpec(f"loco_{c}_s{seed}", kind, (c,), seed, 0.0, wb["group_loco"])
            for c in labels
            if c not in excl
        ]
    raise ValueError(f"unknown run kind {kind!r}")


def train_dict(fcfg: dict[str, Any], spec: RunSpec, smoke: bool, wandb_project: str) -> TrainConfig:
    """The final model's train block with only seed / label smoothing (and smoke) overridden."""
    tr = dict(fcfg["train"])
    prefix = tr.pop("query_prefix")
    if QUERY_PREFIX.get(tr["model_name"], "") != prefix:
        raise ValueError("final.yaml query_prefix differs from models.QUERY_PREFIX")
    tr["model_seed"] = spec.seed
    tr["label_smoothing"] = spec.label_smoothing
    if smoke:
        tr["stop_epoch"] = 1
    return TrainConfig.from_dict({**tr, "wandb_project": wandb_project, "wandb_group": spec.group})


# -------------------------------------------------------------- frozen embeddings
def embed_frozen_rows(
    texts: list[str], model: Any, tok: Any, model_name: str, max_len: int, dev: Any
) -> np.ndarray:
    """Frozen features of raw texts: model query prefix, masked mean pooling, L2-normalised.

    Uses the same pooling code as the B1 baseline (baselines.embed_texts) on a loaded encoder.
    """
    return embed_texts(model, tok, [prepare_text(t, model_name) for t in texts], max_len, dev)


def save_frozen(
    path: Path, ids: list[str], feats: np.ndarray, model_name: str, max_len: int
) -> None:
    """Cache frozen features (ids aligned to rows) as npz; the path is gitignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.array(ids, dtype=str),
        feats=feats.astype(np.float32),
        model_name=model_name,
        max_len=max_len,
    )


def frozen_lookup(path: Path, ids: list[str], model_name: str, max_len: int) -> np.ndarray | None:
    """Rows of the cache for `ids` (in that order), or None if absent / incomplete / mismatched."""
    if not path.exists():
        return None
    z = np.load(path)
    if str(z["model_name"]) != model_name or int(z["max_len"]) != max_len:
        return None
    pos = {i: n for n, i in enumerate(z["ids"].tolist())}
    if any(i not in pos for i in ids):
        return None
    return np.asarray(z["feats"][[pos[i] for i in ids]], dtype=np.float64)


def ensure_frozen(
    cache: Path, frame: pd.DataFrame, model_name: str, max_len: int, expected_s: int, poll_s: float
) -> None:
    """Compute frozen e5 features for every row of `frame` once (exclusive GPU) unless cached."""
    ids = frame["id"].astype(str).tolist()
    if frozen_lookup(cache, ids, model_name, max_len) is not None:
        return
    rows = frame.sort_values("id")
    texts = [prepare_text(t, model_name) for t in rows["text"]]
    with gpu_section(expected_s, "intent-router trackB frozen e5", poll_s):
        emb = embed_e5(texts, model_name, max_len)  # prefix already applied above
    save_frozen(cache, rows["id"].astype(str).tolist(), emb, model_name, max_len)
    print(f"[trackb] frozen e5 features cached: {cache} ({len(rows)} rows)")


def gpu_section(expected_s: int, command: str, poll_s: float) -> Any:
    """Exclusive GPU access when a CUDA device is visible; a no-op on CPU-only runs.

    CPU-only runs (CUDA_VISIBLE_DEVICES="") never touch the GPU, so they neither need nor
    should hold it (exclusive access would also wait for unrelated GPU processes).
    """
    import torch

    if not torch.cuda.is_available():
        return contextlib.nullcontext({"waited_s": 0.0, "lock": "skipped: no CUDA device"})
    return gpu_lock.gpu_exclusive(expected_s, command, poll_s=poll_s)


# ------------------------------------------------------------------- one run
def _local_gold(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return frame["label"].map({lab: i for i, lab in enumerate(labels)}).to_numpy()


def _scores_table(
    sets: HoldoutSets,
    parts: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]],
    unknown_flag: dict[str, bool],
) -> pd.DataFrame:
    """Long score table: id, split, set, is_unknown, gold, pred (names) + one column per method."""
    frames = []
    for name, (rows, pred_idx, scores) in parts.items():
        t = pd.DataFrame(
            {
                "id": rows["id"].to_numpy(),
                "split": rows["split"].to_numpy(),
                "set": "cal" if name == "cal" else "eval",
                "is_unknown": unknown_flag[name],
                "gold": rows["label"].to_numpy(),
                "pred": [sets.labels[i] for i in pred_idx],
            }
        )
        for m, v in scores.items():
            t[m] = v
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _infer(
    model: Any, tok: Any, frame: pd.DataFrame, tcfg: TrainConfig, bs: int
) -> tuple[np.ndarray, np.ndarray]:
    return predict(model, tok, frame["text"].tolist(), tcfg.model_name, tcfg.max_len, bs)


def _wandb_log_ood(
    url: str | None, project: str, summary: dict[str, Any], figs: dict[str, Path]
) -> None:
    """Re-open the finished training run (id taken from its URL) and add OOD summary/images.

    Best effort: any failure only prints. Images are score plots, never dataset text.
    """
    if not url:
        return
    try:
        import wandb

        run = wandb.init(
            project=project, id=url.rstrip("/").split("/")[-1], resume="allow", reinit=True
        )
        run.summary.update(summary)
        if figs:
            run.log({f"fig/{k}": wandb.Image(str(p)) for k, p in figs.items()})
        run.finish()
    except Exception as exc:  # noqa: BLE001 - tracking must never kill the pipeline
        print(f"[wandb] OOD logging failed: {exc!r}")


def run_one(
    spec: RunSpec,
    cfg: dict[str, Any],
    fcfg: dict[str, Any],
    P: Paths,
    df_all: pd.DataFrame,
    use_wandb: bool,
) -> None:
    """Train one holdout model, score every row, write run meta json then the score CSV."""
    import torch

    csv_path = P.results / "scores" / f"{spec.run_id}.csv"
    if csv_path.exists():
        print(f"[trackb] {spec.run_id}: score CSV exists, skipping")
        return
    ood_cfg = cfg["ood"]
    ks = tuple(int(k) for k in ood_cfg["knn_k"])
    sets = build_holdout_sets(df_all, list(spec.holdout), P.smoke)
    labels = sets.labels
    tcfg = train_dict(fcfg, spec, P.smoke, cfg["wandb"]["project"])
    bs = int(fcfg["predict_batch_size"])
    if tcfg.attn_implementation != "eager" or tcfg.epochs != 20 or tcfg.model_seed != spec.seed:
        raise AssertionError("Track B must train with the final config (eager, 20-epoch schedule)")
    all_frame = pd.concat([sets.train, sets.cal, sets.eval_unknown], ignore_index=True)
    ids_needed = df_all["id"].astype(str).tolist() if not P.smoke else all_frame["id"].tolist()
    fz_all = frozen_lookup(P.frozen_cache, ids_needed, tcfg.model_name, tcfg.max_len)
    if fz_all is None:
        raise RuntimeError(f"frozen cache {P.frozen_cache} missing or incomplete (ensure_frozen)")
    fz_of = dict(zip(ids_needed, fz_all, strict=True))

    def fz(frame: pd.DataFrame) -> np.ndarray:
        return np.stack([fz_of[i] for i in frame["id"]])

    meta_run = {
        "run_id": spec.run_id, "track": "B", "kind": spec.kind, "holdout": list(spec.holdout),
        "n_labels": len(labels), "git_sha": git_sha(), "smoke": P.smoke,
    }  # fmt: skip
    t_run = time.perf_counter()
    test_all = (
        None
        if P.smoke
        else df_all[df_all["split"] == "test"].sort_values("id").reset_index(drop=True)
    )
    with gpu_section(
        int(cfg["gpu_lock_expected_s"]) if not P.smoke else 600,
        f"intent-router trackB {spec.run_id}",
        float(cfg["gpu_lock_poll_s"]),
    ) as lock_info:
        res, model, tok = train_model(tcfg, sets.train, sets.cal, meta_run, labels=labels)
        try:
            if int(model.config.num_labels) != len(labels):
                raise AssertionError("model head size differs from the holdout label space")
            sha = state_dict_sha256(model)
            tr_lg, tr_ft = _infer(model, tok, sets.train, tcfg, bs)
            cal_lg, cal_ft = _infer(model, tok, sets.cal, tcfg, bs)
            unk_nt = sets.eval_unknown[sets.eval_unknown["split"] != "test"]
            unk_nt_lg, unk_nt_ft = _infer(model, tok, unk_nt, tcfg, bs)
            test_out: dict[str, Any] | None = None
            if not P.smoke:
                assert test_all is not None

                def infer_test() -> dict[str, Any]:
                    lg, ft = _infer(model, tok, test_all, tcfg, bs)
                    return {"ids": test_all["id"].tolist(), "logits": lg, "features": ft}

                # The one guarded test-split inference of this model (refused if repeated).
                test_out, _ = evaluate_test_once(
                    f"{spec.run_id}:{sha}", "evaluation", infer_test, P.test_log,
                    lambda o: {"n_rows": len(o["ids"])},
                    extra={"run_id": spec.run_id, "n_rows": len(test_all), "role": "trackB"},
                )  # fmt: skip
        finally:
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- everything below is CPU-only and uses saved arrays
    y_tr, y_cal = _local_gold(sets.train, labels), _local_gold(sets.cal, labels)
    scorer: OodScorer = fit_scorer(cal_lg, y_cal, tr_ft, y_tr, len(labels), fz(sets.train), ks)
    unk_logits: dict[str, np.ndarray] = dict(zip(unk_nt["id"], unk_nt_lg, strict=True))
    unk_feats: dict[str, np.ndarray] = dict(zip(unk_nt["id"], unk_nt_ft, strict=True))
    if test_out is not None:
        pos = {i: n for n, i in enumerate(test_out["ids"])}
        known_idx = [pos[i] for i in sets.eval_known["id"]]
        ek_lg, ek_ft = test_out["logits"][known_idx], test_out["features"][known_idx]
        for i in sets.eval_unknown["id"]:
            if i in pos:
                unk_logits[i] = test_out["logits"][pos[i]]
                unk_feats[i] = test_out["features"][pos[i]]
    else:  # smoke: no test inference; the known side is the calibration rows
        ek_lg, ek_ft = cal_lg, cal_ft
    uk = sets.eval_unknown
    uk_lg = np.stack([unk_logits[i] for i in uk["id"]])
    uk_ft = np.stack([unk_feats[i] for i in uk["id"]])
    parts = {
        "cal": (sets.cal, cal_lg.argmax(1), scorer.score(cal_lg, cal_ft, fz(sets.cal))),
        "eval_known": (
            sets.eval_known, ek_lg.argmax(1), scorer.score(ek_lg, ek_ft, fz(sets.eval_known)),
        ),
        "eval_unknown": (uk, uk_lg.argmax(1), scorer.score(uk_lg, uk_ft, fz(uk))),
    }  # fmt: skip
    table = _scores_table(sets, parts, {"cal": False, "eval_known": False, "eval_unknown": True})
    methods = method_names(ks)
    metrics = run_metrics(
        table, methods, labels, float(ood_cfg["retention"]), tuple(ood_cfg["safe_labels"])
    )

    out_dir = P.outputs / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{spec.run_id}.npz",
        train_ids=sets.train["id"].to_numpy(dtype=str), train_logits=tr_lg, train_features=tr_ft,
        cal_ids=sets.cal["id"].to_numpy(dtype=str), cal_logits=cal_lg, cal_features=cal_ft,
        unknown_ids=uk["id"].to_numpy(dtype=str), unknown_logits=uk_lg, unknown_features=uk_ft,
        eval_known_ids=sets.eval_known["id"].to_numpy(dtype=str),
        eval_known_logits=ek_lg, eval_known_features=ek_ft,
    )  # fmt: skip
    figs: dict[str, Path] = {}
    if spec.kind == "headline" and not P.smoke:
        figs = make_figures(table, labels, methods, metrics, P.outputs / "figs" / spec.run_id, "")
    summary = {
        f"ood/{m}/{k}": rec[k]
        for m, rec in metrics["methods"].items()
        for k in ("auroc", "aupr", "fpr_at_95tpr", "strict_rejection_recall",
                  "lenient_rejection_recall", "retention_known")
    }  # fmt: skip
    summary |= {
        "closed_set/macro_f1": metrics["closed_set_known"]["macro_f1"],
        "closed_set/accuracy": metrics["closed_set_known"]["accuracy"],
        "temperature": scorer.temperature,
    }
    if use_wandb and not P.smoke:
        _wandb_log_ood(res.wandb_url, cfg["wandb"]["project"], summary, figs)

    meta = {
        **meta_run,
        "seed": spec.seed,
        "label_smoothing": spec.label_smoothing,
        "group": spec.group,
        "labels": labels,
        "label2id": {lab: i for i, lab in enumerate(labels)},
        "smoke_label": SMOKE_LABEL if P.smoke else None,
        "audit": {
            **sets.audit,
            "fit_sources": FIT_SOURCES,
            "temperature": scorer.temperature,
            "ledoit_wolf_shrinkage": {
                "fine_tuned": scorer.ft_gauss.shrinkage,
                "frozen": scorer.fz_gauss.shrinkage if scorer.fz_gauss else None,
            },
            "test_split_inference": "none (smoke)" if P.smoke else "one guarded call",
        },
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
            "gpu_lock_waited_s": lock_info["waited_s"],
            "wandb_url": res.wandb_url,
            "run_total_s": time.perf_counter() - t_run,
        },  # fmt: skip
        "metrics_at_run_time": metrics,
    }
    write_json(P.results / "runs" / f"{spec.run_id}.json", meta)
    P.results.joinpath("scores").mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".csv.tmp")
    table.to_csv(tmp, index=False, float_format="%.9g")
    os.replace(tmp, csv_path)  # CSV existence == run complete (resume rule)
    print(
        f"[trackb] {spec.run_id}: train {res.wall_clock_s:.0f}s, total "
        f"{time.perf_counter() - t_run:.0f}s, gpu_exclusive={res.gpu_exclusive}"
    )


# ----------------------------------------------------------------------- figures
def make_figures(
    table: pd.DataFrame,
    labels: list[str],
    methods: list[str],
    metrics: dict[str, Any],
    out_dir: Path,
    name_prefix: str,
    hist_methods: list[str] | None = None,
) -> dict[str, Path]:
    """Score histograms (known vs unknown), ROC curves and open-set risk-coverage, from a table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    ev = table[table["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool).to_numpy()
    l2i = {lab: i for i, lab in enumerate(labels)}
    correct = (ev["gold"].map(l2i).to_numpy() == ev["pred"].map(l2i).to_numpy()) & ~unk
    figs: dict[str, Path] = {}
    for m in hist_methods or methods:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        s = ev[m].to_numpy()
        bins = np.histogram_bin_edges(s, bins=25)
        ax.hist(s[~unk], bins=bins, alpha=0.6, label=f"known (n={int((~unk).sum())})")
        ax.hist(s[unk], bins=bins, alpha=0.6, label=f"unknown (n={int(unk.sum())})")
        thr = metrics["methods"][m]["threshold"]
        ax.axvline(thr, color="k", linestyle=":", label="95%-retention threshold (cal)")
        ax.set_title(f"{m}: AUROC {metrics['methods'][m]['auroc']:.3f}", fontsize=10)
        ax.set_xlabel("score (higher = known)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"{name_prefix}hist_{m}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs[f"hist_{m}"] = p
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in methods:
        fpr, tpr, _ = roc_curve(~unk, ev[m].to_numpy())  # known positive
        ax.plot(fpr, tpr, label=f"{m} ({metrics['methods'][m]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR (unknown accepted)")
    ax.set_ylabel("TPR (known accepted)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    figs["roc"] = out_dir / f"{name_prefix}roc.png"
    fig.savefig(figs["roc"], dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in methods:
        cov, risk = risk_coverage(ev[m].to_numpy(), correct)
        ax.plot(cov, risk, label=m)
    ax.set_xlabel("coverage (all eval rows)")
    ax.set_ylabel("risk (accepted unknown or misclassified known)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    figs["risk_coverage"] = out_dir / f"{name_prefix}risk_coverage.png"
    fig.savefig(figs["risk_coverage"], dpi=130)
    plt.close(fig)
    return figs


# ----------------------------------------------------------------------- loading
def load_run(P: Paths, run_id: str, ood_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Meta json + metrics RECOMPUTED from the score CSV; None if the run is not complete."""
    csv_path, meta_path = (
        P.results / "scores" / f"{run_id}.csv",
        P.results / "runs" / f"{run_id}.json",
    )
    if not (csv_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    table = pd.read_csv(csv_path)
    ks = tuple(int(k) for k in ood_cfg["knn_k"])
    methods = method_names(ks)
    metrics = run_metrics(
        table, methods, meta["labels"], float(ood_cfg["retention"]), tuple(ood_cfg["safe_labels"])
    )
    return {"meta": meta, "table": table, "metrics": metrics, "methods": methods}


def _collect(P: Paths, specs: list[RunSpec], ood_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = {}
    for s in specs:
        r = load_run(P, s.run_id, ood_cfg)
        if r is not None:
            runs[s.run_id] = r
    return runs


def _agg(runs: list[dict[str, Any]], method: str, keys: tuple[str, ...]) -> dict[str, Any]:
    """mean/std/values of each key for one method across runs (None values skipped)."""
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r["metrics"]["methods"][method][k] for r in runs]
        vals = [v for v in vals if v is not None]
        out[k] = mean_std(vals) if vals else None
    return out


def _sum_counts(runs: list[dict[str, Any]], method: str) -> dict[str, int]:
    tot: dict[str, int] = {}
    for r in runs:
        for lab, c in r["metrics"]["methods"][method]["unknown_not_rejected_by_pred_label"].items():
            tot[lab] = tot.get(lab, 0) + c
    return dict(sorted(tot.items()))


# ---------------------------------------------------------------------- aggregates
def headline_report(runs: dict[str, dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Mean +- std (ddof=1) over the headline seeds, per method."""
    rs = list(runs.values())
    methods = rs[0]["methods"]
    smoke = rs[0]["meta"]["smoke"]
    return {
        "label": SMOKE_LABEL if smoke else "measured, headline holdout, model seeds " + str(seeds),
        "holdout": rs[0]["meta"]["holdout"],
        "n_runs": len(rs),
        "seeds": [r["meta"]["seed"] for r in rs],
        "std_convention": "sample std (ddof=1) across model seeds",
        "n_eval_known": rs[0]["metrics"]["n_eval_known"],
        "n_eval_unknown": rs[0]["metrics"]["n_eval_unknown"],
        "closed_set_known": {
            k: mean_std([r["metrics"]["closed_set_known"][k] for r in rs])
            for k in ("macro_f1", "accuracy")
        },
        "methods": {
            m: {
                **_agg(rs, m, AGG_KEYS),
                "unknown_not_rejected_by_pred_label_summed_over_seeds": _sum_counts(rs, m),
            }
            for m in methods
        },
        "runs": {
            rid: {
                "seed": r["meta"]["seed"],
                "temperature": r["meta"]["audit"]["temperature"],
                "wall_clock_s": r["meta"]["training"]["wall_clock_s"],
                "gpu_exclusive": r["meta"]["training"]["gpu_exclusive"],
                "wandb_url": r["meta"]["training"]["wandb_url"],
                "per_method": {m: r["metrics"]["methods"][m] for m in methods},
            }
            for rid, r in runs.items()
        },
    }


def select_method(loco_runs: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    """Best mean LOCO AUROC (strict mode); ties go to the earlier method in the fixed order."""
    means = {
        m: float(np.mean([r["metrics"]["methods"][m]["auroc"] for r in loco_runs])) for m in methods
    }
    best = max(methods, key=lambda m: (means[m], -methods.index(m)))
    return {
        "rule": "best mean LOCO AUROC (strict mode) across the leave-one-class-out runs",
        "note": SELECTION_NOTE,
        "n_loco_runs": len(loco_runs),
        "mean_auroc_by_method": means,
        "ranking": sorted(methods, key=lambda m: -means[m]),
        "shipped_method": best,
    }


def loco_report(runs: dict[str, dict[str, Any]], expected: int) -> dict[str, Any]:
    """Per held-out class x method table (AUROC, strict/lenient recall, retention) + means."""
    rs = list(runs.values())
    methods = rs[0]["methods"]
    per_class: dict[str, Any] = {}
    for r in sorted(rs, key=lambda r: r["meta"]["holdout"][0]):
        cls = r["meta"]["holdout"][0]
        per_class[cls] = {
            "n_eval_unknown": r["metrics"]["n_eval_unknown"],
            "n_eval_known": r["metrics"]["n_eval_known"],
            "closed_set_macro_f1": r["metrics"]["closed_set_known"]["macro_f1"],
            "methods": {
                m: {
                    k: r["metrics"]["methods"][m][k]
                    for k in (
                        "auroc",
                        "strict_rejection_recall",
                        "lenient_rejection_recall",
                        "retention_known",
                    )
                }
                for m in methods
            },
        }
    out: dict[str, Any] = {
        "label": SMOKE_LABEL if rs[0]["meta"]["smoke"] else "measured, 1 model seed per class",
        "complete": len(rs) == expected,
        "n_runs": len(rs),
        "expected_runs": expected,
        "per_class": per_class,
        "mean_across_classes": {
            m: _agg(
                rs,
                m,
                (
                    "auroc",
                    "strict_rejection_recall",
                    "lenient_rejection_recall",
                    "retention_known",
                    "aupr",
                    "fpr_at_95tpr",
                ),
            )
            for m in methods
        },  # fmt: skip
    }
    if len(rs) == expected:
        out["selection"] = select_method(rs, methods)
    return out


def ls_report(base: dict[str, dict[str, Any]], ls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """LS 0.1 vs LS 0 on the headline holdout: per-method metric deltas paired by model seed."""
    b = {r["meta"]["seed"]: r for r in base.values()}
    a = {r["meta"]["seed"]: r for r in ls.values()}
    seeds = sorted(set(b) & set(a))
    methods = next(iter(base.values()))["methods"]
    out: dict[str, Any] = {
        "label": "measured, headline holdout; delta = label_smoothing 0.1 minus 0, paired by seed",
        "seeds": seeds,
        "std_convention": "sample std (ddof=1) across model seeds",
        "closed_set_known": {
            k: {
                "ls0": mean_std([b[s]["metrics"]["closed_set_known"][k] for s in seeds]),
                "ls0p1": mean_std([a[s]["metrics"]["closed_set_known"][k] for s in seeds]),
                "delta": mean_std(
                    [
                        a[s]["metrics"]["closed_set_known"][k]
                        - b[s]["metrics"]["closed_set_known"][k]
                        for s in seeds
                    ]
                ),
            }
            for k in ("macro_f1", "accuracy")
        },
        "methods": {},
    }
    for m in methods:
        out["methods"][m] = {
            k: {
                "ls0": mean_std([b[s]["metrics"]["methods"][m][k] for s in seeds]),
                "ls0p1": mean_std([a[s]["metrics"]["methods"][m][k] for s in seeds]),
                "delta": mean_std(
                    [
                        a[s]["metrics"]["methods"][m][k] - b[s]["metrics"]["methods"][m][k]
                        for s in seeds
                    ]
                ),
            }
            for k in LS_KEYS
        }
    return out


def audit_report(runs: dict[str, dict[str, Any]], P: Paths, smoke: bool) -> dict[str, Any]:
    """Leakage audit: per run sizes, assertion results, fitted-quantity sources, test-log count."""
    log_lines = (
        [json.loads(x) for x in P.test_log.read_text().splitlines() if x.strip()]
        if P.test_log.exists()
        else []
    )
    per_run = []
    for rid, r in sorted(runs.items()):
        a = r["meta"]["audit"]
        n_log = sum(1 for e in log_lines if e.get("run_id") == rid)
        per_run.append(
            {
                "run_id": rid,
                "holdout": a["holdout"],
                "seed": r["meta"]["seed"],
                "n_train": a["n_train"],
                "n_cal": a["n_cal"],
                "n_eval_known": a["n_eval_known"],
                "n_eval_unknown": a["n_eval_unknown"],
                "assertions": a["assertions"],
                "all_assertions_passed": all(a["assertions"].values()),
                "ids_sha256": a["ids_sha256"],
                "fit_sources": a["fit_sources"],
                "test_inference_log_lines": n_log,
                "test_inference_log_ok": smoke or n_log == 1,
            }
        )
    return {
        "smoke": smoke,
        "n_runs": len(per_run),
        "all_assertions_passed": all(p["all_assertions_passed"] for p in per_run),
        "all_test_logs_exactly_one": all(p["test_inference_log_ok"] for p in per_run),
        "test_inference_log": str(P.test_log),
        "frozen_features_note": "frozen e5 features of all 500 rows (incl. test) are computed once "
        "by a fixed pretrained encoder; nothing is fitted on them except from train-known rows",
        "runs": per_run,
    }


# --------------------------------------------------------------------- shipped
def stage_shipped(
    cfg: dict[str, Any], fcfg: dict[str, Any], P: Paths, df_all: pd.DataFrame
) -> None:
    """Apply the best-LOCO method to the saved 12-class final model (no model inference at all)."""
    ood_cfg = cfg["ood"]
    fa = cfg["final_artifacts"]
    npz_path = Path(fa["features_logits"])
    if not npz_path.exists():
        print(f"[trackb] shipped: SKIPPED, final-model artifacts not found: {npz_path}")
        return
    labels = list(data_mod.LABELS)
    specs = plan_runs(cfg, labels, "loco")
    runs = _collect(P, specs, ood_cfg)
    if len(runs) != len(specs):
        have = sorted(runs)
        print(f"[trackb] shipped: SKIPPED, LOCO incomplete ({len(runs)}/{len(specs)}): {have}")
        return
    methods = method_names(tuple(int(k) for k in ood_cfg["knn_k"]))
    selection = select_method(list(runs.values()), methods)
    method = selection["shipped_method"]
    z = np.load(npz_path)
    l2i = {lab: i for i, lab in enumerate(labels)}
    label_of = dict(zip(df_all["id"], df_all["label"], strict=True))

    def gold(ids: np.ndarray) -> np.ndarray:
        return np.array([l2i[label_of[i]] for i in ids.tolist()])

    y_tr, y_val, y_te = gold(z["train_ids"]), gold(z["val_ids"]), gold(z["test_ids"])
    tcfg = fcfg["train"]
    fz = {}
    if method.endswith("_frozen"):
        for part in ("train", "val", "test"):
            f = frozen_lookup(
                P.frozen_cache, z[f"{part}_ids"].tolist(), tcfg["model_name"], int(tcfg["max_len"])
            )
            if f is None:
                print(f"[trackb] shipped: SKIPPED, frozen cache missing for {part} ids")
                return
            fz[part] = f
    ks = tuple(int(k) for k in ood_cfg["knn_k"])
    scorer = fit_scorer(
        z["val_logits"], y_val, z["train_features"], y_tr, len(labels), fz.get("train"), ks
    )
    val_s = scorer.score(z["val_logits"], z["val_features"], fz.get("val"))[method]
    te_s = scorer.score(z["test_logits"], z["test_features"], fz.get("test"))[method]
    thr = threshold_at_retention(val_s, float(ood_cfg["retention"]))
    te_pred = z["test_logits"].argmax(axis=1)
    pred_csv = Path(fa["test_predictions"])
    consistent = None
    if pred_csv.exists():
        saved = pd.read_csv(pred_csv)
        consistent = bool(
            saved["id"].tolist() == z["test_ids"].tolist()
            and np.array_equal(saved["pred"].to_numpy(), te_pred)
        )
    acc = te_s >= thr
    m = re.fullmatch(r"knn(\d+)_(ft|frozen)", method)
    out = {
        "method": method,
        "k": int(m.group(1)) if m else None,
        "selection": selection,
        "threshold": thr,
        "threshold_source": f"{ood_cfg['retention']:.0%} retention on val (all 12 classes known)",
        "tie_handling": "accept = score >= threshold; ties at the threshold are accepted, so "
        "achieved retention is >= the target",
        "temperature": scorer.temperature,
        "temperature_source": "val NLL, LBFGS",
        "n_val": len(y_val),
        "val_retention_achieved": float(np.mean(val_s >= thr)),
        "n_test": len(y_te),
        "n_test_accepted": int(acc.sum()),
        "test_coverage_at_threshold": float(acc.mean()),
        "test_accepted_macro_f1_present": macro_f1_present(y_te[acc], te_pred[acc], len(labels))
        if acc.any()
        else None,
        "test_accepted_accuracy": float(np.mean(y_te[acc] == te_pred[acc])) if acc.any() else None,
        "test_all_accuracy": float(np.mean(y_te == te_pred)),
        "test_predictions_match_saved_csv": consistent,
        "sources": {
            "arrays": str(npz_path),
            "test_inference": "none: saved test logits/features from final.py",
        },
    }
    write_json(P.results / "shipped.json", out)
    print(
        f"[trackb] shipped {method}: thr {thr:.4f} val-ret {out['val_retention_achieved']:.3f} "
        f"test-cov {out['test_coverage_at_threshold']:.3f}"
    )


# ----------------------------------------------------------------------- report
def stage_report(cfg: dict[str, Any], P: Paths, labels: list[str]) -> None:
    """Aggregate finished runs into headline/loco/ls/audit json and the headline figures."""
    ood_cfg = cfg["ood"]
    head_specs = plan_runs(cfg, labels, "headline")
    loco_specs = plan_runs(cfg, labels, "loco")
    ls_specs = plan_runs(cfg, labels, "ls")
    head = _collect(P, head_specs, ood_cfg)
    loco = _collect(P, loco_specs, ood_cfg)
    ls = _collect(P, ls_specs, ood_cfg)
    all_runs = {**head, **loco, **ls}
    if head:
        h = headline_report(head, [s.seed for s in head_specs])
        h["complete"] = len(head) == len(head_specs)
        write_json(P.results / "headline.json", h)
        _headline_figures(head, h, P)
    if loco:
        write_json(P.results / "loco.json", loco_report(loco, len(loco_specs)))
    if ls and head:
        write_json(P.results / "ls_ablation.json", ls_report(head, ls))
    if all_runs:
        write_json(P.results / "audit.json", audit_report(all_runs, P, P.smoke))
    print(f"[trackb] report: headline {len(head)}, loco {len(loco)}, ls {len(ls)} runs aggregated")


def _headline_figures(head: dict[str, dict[str, Any]], h: dict[str, Any], P: Paths) -> None:
    """Histograms for MSP, the best fine-tuned and best frozen method (mean AUROC), ROC, RC."""
    first = head[sorted(head)[0]]  # lowest seed run: plots show one model, not a pooled score
    methods = first["methods"]
    mean_auc = {m: h["methods"][m]["auroc"]["mean"] for m in methods}
    best_ft = max((m for m in methods if m.endswith("_ft")), key=mean_auc.get)  # type: ignore[arg-type]
    best_fz = max((m for m in methods if m.endswith("_frozen")), key=mean_auc.get)  # type: ignore[arg-type]
    figs = make_figures(
        first["table"], first["meta"]["labels"], methods, first["metrics"], P.figures,
        "trackb_headline_", ["msp", best_ft, best_fz],
    )  # fmt: skip
    h_path = P.results / "headline.json"
    h["figures"] = {k: str(v) for k, v in figs.items()}
    h["figure_note"] = f"figures from run {sorted(head)[0]} (one model seed)"
    h["best_finetuned_by_mean_auroc"], h["best_frozen_by_mean_auroc"] = best_ft, best_fz
    write_json(h_path, h)


# ------------------------------------------------------------------------ main
def _frame(fcfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    """All dataset rows with split info; smoke drops the test split before anything else."""
    df = data_mod.get_frame(["train", "val", "test"], fcfg["data_path"], fcfg["splits_path"])
    if smoke:
        df = df[df["split"] != "test"].reset_index(drop=True)
        if (df["split"] == "test").any():
            raise AssertionError("smoke frame must not contain test rows")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Track B: open-set experiments")
    ap.add_argument("--config", default="configs/trackb.yaml")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, no test inference")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--poll-s", type=float, default=None, help="override gpu_lock_poll_s")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fcfg = yaml.safe_load(Path(cfg["final_config"]).read_text(encoding="utf-8"))
    if args.poll_s is not None:
        cfg["gpu_lock_poll_s"] = args.poll_s
    if args.smoke and args.stage not in ("headline", "all"):
        raise SystemExit("--smoke only runs the headline stage")
    use_wandb = bool(cfg["wandb"]["enabled"]) and not args.no_wandb and not args.smoke
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    P = make_paths(cfg, args.smoke)
    data_mod.load_data(fcfg["data_path"])
    labels = list(data_mod.LABELS)
    df_all = _frame(fcfg, args.smoke)
    tcfg = fcfg["train"]
    stages = (
        ["headline", "loco", "ls", "shipped", "report"] if args.stage == "all" else [args.stage]
    )
    if args.smoke:
        stages = ["headline"]
        cfg["headline"]["seeds"] = cfg["headline"]["seeds"][:1]
    needs_training = [s for s in stages if s in ("headline", "loco", "ls")]
    if needs_training:
        ensure_frozen(
            P.frozen_cache, df_all, tcfg["model_name"], int(tcfg["max_len"]),
            int(cfg["gpu_lock_expected_s"]), float(cfg["gpu_lock_poll_s"]),
        )  # fmt: skip
    for st in stages:
        if st in ("headline", "loco", "ls"):
            for spec in plan_runs(cfg, labels, st):
                run_one(spec, cfg, fcfg, P, df_all, use_wandb)
        elif st == "shipped":
            stage_shipped(cfg, fcfg, P, df_all)
        elif st == "report":
            stage_report(cfg, P, labels)
    if args.smoke:
        r = load_run(P, plan_runs(cfg, labels, "headline")[0].run_id, cfg["ood"])
        assert r is not None
        write_json(P.results / "smoke_metrics.json", r["metrics"])
        print(f"[trackb] {SMOKE_LABEL}")
        for m, rec in r["metrics"]["methods"].items():
            print(
                f"  {m:12s} AUROC {rec['auroc']:.4f}  AUPR {rec['aupr']:.4f}  "
                f"FPR@95TPR {rec['fpr_at_95tpr']:.4f}  strict-recall "
                f"{rec['strict_rejection_recall']:.3f}  retention {rec['retention_known']:.3f}"
            )


if __name__ == "__main__":
    main()
