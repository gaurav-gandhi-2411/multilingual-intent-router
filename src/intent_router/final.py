from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import parallel_backend

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.analysis import (
    COMBINED_LABEL,
    SLICE_MACRO_F1_DEFINITION,
    load_oof_mean,
    ranked_errors,
    shipment_family_hypothesis,
    slice_report,
    top_confusions,
)
from intent_router.data import get_frame
from intent_router.evaluate import (
    array_sha256,
    aurc,
    bootstrap_metrics_ci,
    confusion_matrix,
    ece_bins,
    evaluate_test_once,
    fit_temperature,
    git_sha,
    hierarchy_summary,
    macro_f1_present,
    mcnemar_exact,
    nll,
    paired_bootstrap_delta_f1,
    per_class_report,
    risk_coverage,
    row_normalise,
    selective_at_threshold,
    softmax,
    threshold_at_retention,
)
from intent_router.models import QUERY_PREFIX, predict, state_dict_sha256
from intent_router.stats import accuracy, macro_f1
from intent_router.train import TrainConfig, train_model

STAGES = ("train", "evaluate", "all", "analysis")
TEST_LABEL = "unbiased (single test evaluation)"
CV_LABEL = "post-selection (optimistic) CV estimate"
SMOKE_LABEL = "SMOKE: val stand-in for the test split, NOT a test score"


@dataclass(frozen=True)
class Paths:
    """Where one invocation writes; smoke runs are fully redirected under outputs/."""

    results: Path
    figures: Path
    outputs: Path
    model_dir: Path
    test_log: Path
    smoke: bool


def make_paths(cfg: dict[str, Any], smoke: bool) -> Paths:
    """Real paths from the config, or a self-contained sandbox under outputs/final_smoke."""
    if smoke:
        root = Path("outputs/final_smoke")
        return Paths(
            root / "results", root / "figures", root / "outputs", root / "model",
            root / "test_eval_log.jsonl", True,
        )  # fmt: skip
    return Paths(
        Path(cfg["results_dir"]),
        Path(cfg["figures_dir"]),
        Path(cfg["outputs_dir"]),
        Path(cfg["model_dir"]),
        Path(cfg["test_eval_log"]),
        False,
    )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: Any) -> None:
    """Write indented JSON (numpy scalars/arrays converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def eval_frame(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    """The split evaluated by the guarded call: test normally, VAL ONLY in smoke (never test)."""
    split = "val" if smoke else "test"
    assert not (smoke and split == "test"), "smoke runs must never load test rows"
    df = get_frame([split], cfg["data_path"], cfg["splits_path"])
    return df.sort_values("id").reset_index(drop=True)


def prediction_frame(
    ids: list[str], gold: np.ndarray, probs: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """ids + gold/pred (index and name) + raw MSP confidence + the 12 probabilities. No text."""
    pred = probs.argmax(axis=1)
    df = pd.DataFrame(
        {
            "id": ids,
            "gold": gold,
            "pred": pred,
            "gold_label": [labels[i] for i in gold],
            "pred_label": [labels[i] for i in pred],
            "confidence": probs.max(axis=1),
        }
    )
    for i in range(probs.shape[1]):
        df[f"prob_{i}"] = probs[:, i]
    return df


def save_pred_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV with float32-exact probabilities (9 significant digits round-trip float32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.9g")


# ---------------------------------------------------------------------- train stage
def _headline(out: dict[str, Any]) -> dict[str, Any]:
    pred = out["logits"].argmax(axis=1)
    return {
        "macro_f1": macro_f1(out["gold"], pred, out["logits"].shape[1]),
        "accuracy": accuracy(out["gold"], pred),
        "n": len(pred),
    }


def _run_summary(res: Any) -> dict[str, Any]:
    return {
        "config": res.config,
        "epochs": res.epochs,
        "wall_clock_s": res.wall_clock_s,
        "load_s": res.load_s,
        "train_epoch_s": res.train_epoch_s,
        "eval_epoch_s": res.eval_epoch_s,
        "peak_vram_mb": res.peak_vram_mb,
        "gpu_exclusive": res.gpu_exclusive,
        "gpu_foreign_seen": res.gpu_foreign_seen,
        "gpu_snapshot_start": res.gpu_snapshot_start,
        "gpu_snapshot_end": res.gpu_snapshot_end,
        "nan_detected": res.nan_detected,
        "precision": res.precision,
        "wandb_url": res.wandb_url,
        "wandb_logged": res.wandb_logged,
    }


def _time_inference(
    model: Any, tok: Any, texts: list[str], name: str, max_len: int, bs: int, repeats: int
) -> float:
    """Median ms/row of predict() over `repeats` timed calls after one warm-up call."""
    import torch

    predict(model, tok, texts, name, max_len, bs)
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(model, tok, texts, name, max_len, bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / len(texts) * 1000.0


def save_final_model(model: Any, tok: Any, model_dir: Path, tcfg: TrainConfig) -> None:
    """save_pretrained + tokenizer; config.json gains query_prefix and max_length."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    cfg_path = model_dir / "config.json"
    d = json.loads(cfg_path.read_text(encoding="utf-8"))
    d["query_prefix"] = QUERY_PREFIX.get(tcfg.model_name, "")
    d["max_length"] = tcfg.max_len
    cfg_path.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _run_one(
    name: str,
    role: str,
    tcfg: TrainConfig,
    cfg: dict[str, Any],
    P: Paths,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train once, then infer on val (and train/test as the role requires), then free the GPU.

    role: primary (saved model + the single guarded 'evaluation'), repeat (guarded
    'determinism_inference'), sdpa (timing + val only; never touches the test split).
    """
    import torch

    labels = list(data_mod.LABELS)
    l2i = data_mod.label2id()
    bs, ml = int(cfg["predict_batch_size"]), tcfg.max_len
    res, model, tok = train_model(tcfg, train_df, val_df, {"run_id": name, "role": role})
    out: dict[str, Any] = {"name": name, "role": role, "attn": tcfg.attn_implementation}
    try:
        out["fingerprint"] = state_dict_sha256(model)
        val_texts = val_df["text"].tolist()
        out["val_logits"], out["val_features"] = predict(
            model, tok, val_texts, tcfg.model_name, ml, bs
        )
        out["inference_ms_per_row_val"] = _time_inference(
            model, tok, val_texts, tcfg.model_name, ml, bs,
            int(cfg["determinism"]["inference_timing_repeats"]),
        )  # fmt: skip
        if role == "primary":
            out["train_logits"], out["train_features"] = predict(
                model, tok, train_df["text"].tolist(), tcfg.model_name, ml, bs
            )
            save_final_model(model, tok, P.model_dir, tcfg)

        def infer() -> dict[str, Any]:
            df = eval_frame(cfg, P.smoke)
            lg, ft = predict(model, tok, df["text"].tolist(), tcfg.model_name, ml, bs)
            return {
                "ids": df["id"].tolist(),
                "gold": df["label"].map(l2i).to_numpy(),
                "logits": lg,
                "features": ft,
            }

        extra = {"role": role, "run": name, "smoke": P.smoke}
        if role == "primary":
            out["eval"], out["eval_headline"] = evaluate_test_once(
                out["fingerprint"], "evaluation", infer, P.test_log, _headline, extra=extra
            )
        elif role == "repeat":
            out["eval"], _ = evaluate_test_once(
                out["fingerprint"], "determinism_inference", infer, P.test_log, extra=extra
            )
        out["val_macro_f1_fp32"] = macro_f1(
            val_df["label"].map(l2i).to_numpy(), out["val_logits"].argmax(axis=1), len(labels)
        )
    finally:
        model = None  # drop the last reference so the GPU is free for the next run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["summary"] = _run_summary(res)
    return out


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    return {
        "identical": bool(np.array_equal(a, b)),
        "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
        "sha256_run1": array_sha256(a),
        "sha256_run2": array_sha256(b),
    }


def determinism_report(r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
    """Bit-for-bit comparison of two identically seeded eager runs (val + eval-split outputs)."""
    rep: dict[str, Any] = {
        "seed": r1["summary"]["config"]["model_seed"],
        "attn_implementation": r1["attn"],
        "fingerprint_run1": r1["fingerprint"],
        "fingerprint_run2": r2["fingerprint"],
        "state_dict_identical": r1["fingerprint"] == r2["fingerprint"],
    }
    for split, (a, b) in {
        "val": (r1["val_logits"], r2["val_logits"]),
        "test": (r1["eval"]["logits"], r2["eval"]["logits"]),
    }.items():
        rep[split] = {
            "logits": _compare_arrays(a, b),
            "softmax_float32": _compare_arrays(softmax(a), softmax(b)),
        }
    keys = ("train_loss", "eval_loss", "macro_f1", "accuracy")
    h1 = [{k: e[k] for k in keys} for e in r1["summary"]["epochs"]]
    h2 = [{k: e[k] for k in keys} for e in r2["summary"]["epochs"]]
    rep["epoch_history_identical"] = h1 == h2
    rep["bitwise_identical"] = bool(
        rep["state_dict_identical"]
        and rep["epoch_history_identical"]
        and all(
            rep[s][k]["identical"] for s in ("val", "test") for k in ("logits", "softmax_float32")
        )
    )
    return rep


def stage_train(cfg: dict[str, Any], P: Paths, smoke: bool) -> None:
    """Train twice (eager, same seed) + one sdpa timing run, all with exclusive GPU access."""
    data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    l2i = data_mod.label2id()
    tr = dict(cfg["train"])
    prefix = tr.pop("query_prefix")
    if QUERY_PREFIX.get(tr["model_name"], "") != prefix:
        raise ValueError("final.yaml query_prefix differs from models.QUERY_PREFIX")
    if smoke:
        tr["stop_epoch"] = 1
    wb = cfg["wandb"]
    base = {**tr, "wandb_project": wb["project"], "wandb_group": wb["group"]}
    train_df = get_frame(["train"], cfg["data_path"], cfg["splits_path"])
    val_df = get_frame(["val"], cfg["data_path"], cfg["splits_path"])
    train_df = train_df.sort_values("id").reset_index(drop=True)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    if (train_df["split"] != "train").any() or (val_df["split"] != "val").any():
        raise AssertionError("unexpected split in train/val frames")
    eager = TrainConfig.from_dict(base)
    sdpa = TrainConfig.from_dict({**base, "attn_implementation": "sdpa"})
    det = cfg["determinism"]
    t_start = time.perf_counter()
    runs: dict[str, dict[str, Any]] = {}
    with gpu_lock.gpu_exclusive(
        int(cfg["gpu_lock_expected_s"]) if not smoke else 600, "intent-router final"
    ) as lock_info:
        runs["run1"] = _run_one("final_run1_eager", "primary", eager, cfg, P, train_df, val_df)
        if det["repeat_run"]:
            runs["run2"] = _run_one(
                "final_run2_eager_repeat", "repeat", eager, cfg, P, train_df, val_df
            )
        if det["sdpa_timing_run"]:
            runs["sdpa"] = _run_one(
                "final_run3_sdpa_timing", "sdpa", sdpa, cfg, P, train_df, val_df
            )
    r1 = runs["run1"]

    # ---- predictions (ids only) and features/logits
    gold_val = val_df["label"].map(l2i).to_numpy()
    save_pred_csv(
        prediction_frame(val_df["id"].tolist(), gold_val, softmax(r1["val_logits"]), labels),
        P.results / "val_predictions.csv",
    )
    ev = r1["eval"]
    save_pred_csv(
        prediction_frame(ev["ids"], ev["gold"], softmax(ev["logits"]), labels),
        P.results / "test_predictions.csv",
    )
    P.outputs.mkdir(parents=True, exist_ok=True)
    np.savez(
        P.outputs / "features_logits.npz",
        train_ids=np.array(train_df["id"].tolist(), dtype=str),
        train_logits=r1["train_logits"],
        train_features=r1["train_features"],
        val_ids=np.array(val_df["id"].tolist(), dtype=str),
        val_logits=r1["val_logits"],
        val_features=r1["val_features"],
        test_ids=np.array(ev["ids"], dtype=str),
        test_logits=ev["logits"],
        test_features=ev["features"],
    )

    # ---- speed cost of eager vs sdpa (timing only; val metric, never test)
    speed: dict[str, Any] | None = None
    if "sdpa" in runs:

        def stats_of(r: dict[str, Any]) -> dict[str, Any]:
            s = r["summary"]
            return {
                "attn": r["attn"],
                "train_wall_clock_s": s["wall_clock_s"],
                "train_epoch_s_median": statistics.median(s["train_epoch_s"]),
                "inference_ms_per_row_val": r["inference_ms_per_row_val"],
                "peak_vram_mb": s["peak_vram_mb"],
                "val_macro_f1_fp32": r["val_macro_f1_fp32"],
                "val_macro_f1_last_epoch_amp": s["epochs"][-1]["macro_f1"],
                "gpu_exclusive": s["gpu_exclusive"],
            }

        # Compare against the warm repeat run when present: run 1 pays CUDA warm-up.
        eg, sd = stats_of(runs.get("run2", r1)), stats_of(runs["sdpa"])
        speed = {
            "eager": eg,
            "sdpa": sd,
            "eager_over_sdpa_train_epoch_time": eg["train_epoch_s_median"]
            / sd["train_epoch_s_median"],
            "eager_over_sdpa_inference_time": eg["inference_ms_per_row_val"]
            / sd["inference_ms_per_row_val"],
            "note": "val only; eager run 1 includes CUDA warm-up in its first epoch (median used)",
        }
        speed["eager_run1"] = stats_of(r1)

    det_rep = determinism_report(r1, runs["run2"]) if "run2" in runs else None
    if det_rep is not None:
        write_json(P.results / "determinism.json", det_rep)
    write_json(
        P.results / "train_summary.json",
        {
            "smoke": smoke,
            "git_sha": git_sha(),
            "stage_wall_clock_s": time.perf_counter() - t_start,
            "gpu_lock": lock_info,
            "model_fingerprint": r1["fingerprint"],
            "model_dir": str(P.model_dir),
            "runs": {k: {**v["summary"], "fingerprint": v["fingerprint"]} for k, v in runs.items()},
            "val_macro_f1_fp32_run1": r1["val_macro_f1_fp32"],
            "eval_headline_run1": r1["eval_headline"]
            | {"label": SMOKE_LABEL if smoke else TEST_LABEL},
            "speed_eager_vs_sdpa": speed,
            "test_eval_log": str(P.test_log),
        },
    )
    print(
        f"[train] fingerprint {r1['fingerprint'][:16]} val_macro_f1={r1['val_macro_f1_fp32']:.4f}"
    )
    if det_rep is not None:
        print(f"[train] bitwise identical: {det_rep['bitwise_identical']}")
        if not det_rep["bitwise_identical"]:
            raise AssertionError(
                f"two identical eager runs differ; see {P.results / 'determinism.json'}"
            )


# ------------------------------------------------------------------- baselines (eval)
def _baseline_fingerprint(kind: str, cfg: dict[str, Any], train_df: pd.DataFrame) -> str:
    """sha256 of the sklearn setup + the exact training data (ids and texts)."""
    import sklearn

    bc = cfg["baselines"]
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "kind": kind,
                "estimator": "LogisticRegression(class_weight=balanced,max_iter=5000)",
                "tfidf": "char_wb(2,5)" if kind == "B0" else None,
                "e5": bc["e5_model"] if kind == "B1" else None,
                "c_grid": bc["c_grid"],
                "inner_folds": bc["inner_folds"],
                "max_len": cfg["train"]["max_len"],
                "sklearn": sklearn.__version__,
            },
            sort_keys=True,
        ).encode()
    )
    for i, t, lab in zip(train_df["id"], train_df["text"], train_df["label"], strict=True):
        h.update(f"|{i}|{t}|{lab}".encode())
    return h.hexdigest()


def baseline_predictions(
    kind: str, cfg: dict[str, Any], P: Paths, train_df: pd.DataFrame, labels: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """B0/B1 refit on the train split; evaluate on the eval split once (through the guard).

    If the saved prediction file already exists it is reused (the guard would refuse anyway).
    """
    from intent_router.baselines import embed_e5, fit_predict_proba
    from intent_router.models import prepare_text

    csv = P.results / f"baseline_{kind}_test_predictions.csv"
    meta_path = P.results / "baselines_fit.json"
    metas = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if csv.exists() and kind in metas:
        return pd.read_csv(csv), metas[kind]
    bc = cfg["baselines"]
    l2i = data_mod.label2id()
    k = len(labels)
    fp = _baseline_fingerprint(kind, cfg, train_df)
    max_len = int(cfg["train"]["max_len"])

    def infer() -> dict[str, Any]:
        ev = eval_frame(cfg, P.smoke)
        ytr = train_df["label"].map(l2i).to_numpy()
        if kind == "B0":
            xtr, xev = train_df["text"].to_numpy(), ev["text"].to_numpy()
        else:
            texts = [prepare_text(t, bc["e5_model"]) for t in [*train_df["text"], *ev["text"]]]
            emb = embed_e5(texts, bc["e5_model"], max_len)
            xtr, xev = emb[: len(train_df)], emb[len(train_df) :]
        # loky workers cannot un-pickle once torch/CUDA is loaded in this process (the same
        # BrokenProcessPool baselines.py documents for B1); threads give identical results.
        with parallel_backend("threading", n_jobs=-1):
            probs, best_c = fit_predict_proba(
                kind, xtr, ytr, train_df["dup_group"].to_numpy(), xev,
                bc["c_grid"], int(bc["inner_folds"]), k,
            )  # fmt: skip
        return {
            "ids": ev["id"].tolist(),
            "gold": ev["label"].map(l2i).to_numpy(),
            "logits": probs,  # probabilities; _headline only takes argmax
            "best_C": best_c,
        }

    def call() -> dict[str, Any]:
        out, headline = evaluate_test_once(
            fp, "evaluation", infer, P.test_log, _headline,
            extra={"role": f"baseline_{kind}", "smoke": P.smoke},
        )  # fmt: skip
        return {**out, "headline": headline}

    if kind == "B1":
        with gpu_lock.gpu_exclusive(600, f"intent-router final baseline {kind}"):
            res = call()
    else:
        res = call()
    df = prediction_frame(res["ids"], res["gold"], res["logits"], labels)
    save_pred_csv(df, csv)
    meta = {"fingerprint": fp, "best_C": res["best_C"], "n_train": len(train_df)}
    metas[kind] = meta
    write_json(meta_path, metas)
    return df, meta


# ------------------------------------------------------------------------- figures
def _short(labels: list[str]) -> list[str]:
    return [lab.replace("shipment_information.", "ship.") for lab in labels]


def plot_confusion(mat: np.ndarray, labels: list[str], title: str, path: Path, fmt: str) -> None:
    """Annotated confusion heatmap (rows = gold, cols = predicted)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="Blues")
    names = _short(labels)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, fontsize=10)
    top = mat.max() if mat.size else 1
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > top / 2 else "black")  # fmt: skip
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(
    before: list[dict[str, float]], after: list[dict[str, float]], ece_b: float, ece_a: float,
    temperature: float, path: Path,
) -> None:  # fmt: skip
    """Two reliability diagrams (raw vs temperature-scaled) on the evaluated split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, bins, ece, title in (
        (axes[0], before, ece_b, "before (T = 1)"),
        (axes[1], after, ece_a, f"after (T = {temperature:.3f})"),
    ):
        width = 1.0 / len(bins)
        xs = [b["lo"] + width / 2 for b in bins if b["count"]]
        accs = [b["accuracy"] for b in bins if b["count"]]
        ax.bar(xs, accs, width=width * 0.95, edgecolor="black", alpha=0.7, label="accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.set_title(f"{title}: ECE = {ece:.4f}", fontsize=10)
        ax.set_xlabel("confidence (max softmax)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], cov_at_thr: float, path: Path
) -> None:
    """Risk-coverage curves (one per confidence score) with the shipped-threshold coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=name)
    ax.axvline(cov_at_thr, color="grey", linestyle=":", label="95%-val-retention threshold")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (error rate on accepted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- evaluate stage
def _wandb_final(
    cfg: dict[str, Any], summary: dict[str, Any], per_class: pd.DataFrame, y: np.ndarray,
    pred: np.ndarray, labels: list[str], errors: pd.DataFrame, rc: tuple[np.ndarray, np.ndarray],
    figs: dict[str, Path],
) -> str | None:  # fmt: skip
    """W&B final run: ids/labels/numbers only, never dataset text. Never raises."""
    try:
        import wandb

        run = wandb.init(
            project=cfg["wandb"]["project"], group=cfg["wandb"]["group"], name="final-track-a",
            job_type="evaluate", config=cfg, reinit=True,
        )  # fmt: skip
        run.summary.update(summary)
        rc_table = wandb.Table(
            data=[[float(c), float(r)] for c, r in zip(*rc, strict=True)],
            columns=["coverage", "risk"],
        )
        run.log(
            {
                "per_class": wandb.Table(dataframe=per_class),
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=[int(v) for v in y],
                    preds=[int(v) for v in pred],
                    class_names=labels,
                ),  # fmt: skip
                "misclassifications": wandb.Table(
                    columns=["id", "gold", "pred", "confidence"],
                    data=errors[["id", "gold_label", "pred_label", "confidence"]].values.tolist(),
                ),
                "risk_coverage": wandb.plot.line(
                    rc_table, "coverage", "risk", title="risk-coverage (temp-scaled MSP)"
                ),
                **{f"fig/{k}": wandb.Image(str(p)) for k, p in figs.items()},
            }
        )
        url = getattr(run, "url", None)
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001 - tracking must never kill the pipeline
        print(f"[wandb] final logging failed: {exc!r}")
        return None


def stage_evaluate(cfg: dict[str, Any], P: Paths, smoke: bool, use_wandb: bool) -> None:
    """Track A metrics, baselines, slices and error analysis from the saved predictions."""
    data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    k = len(labels)
    l2i = data_mod.label2id()
    bs_cfg = cfg["bootstrap"]
    n_res, seed = int(bs_cfg["n_resamples"]), int(bs_cfg["seed"])
    z = np.load(P.outputs / "features_logits.npz")
    val_df = get_frame(["val"], cfg["data_path"], cfg["splits_path"]).sort_values("id")
    val_df = val_df.reset_index(drop=True)
    y_val = val_df["label"].map(l2i).to_numpy()
    ev_csv = pd.read_csv(P.results / "test_predictions.csv")
    ev_ids = ev_csv["id"].tolist()
    y = ev_csv["gold"].to_numpy()
    if z["val_ids"].tolist() != val_df["id"].tolist() or z["test_ids"].tolist() != ev_ids:
        raise AssertionError("npz ids do not match the saved prediction files")
    val_logits, ev_logits = z["val_logits"], z["test_logits"]
    probs = softmax(ev_logits)
    pred = probs.argmax(axis=1)
    if not np.array_equal(pred, ev_csv["pred"].to_numpy()):
        raise AssertionError("saved predictions disagree with saved logits")

    # ---- headline metrics, per class, confusion, hierarchy
    ci = bootstrap_metrics_ci(y, pred, k, n_res, seed)
    conf_counts = confusion_matrix(y, pred, k)
    names = [{"index": i, "label": lab} for i, lab in enumerate(labels)]
    pc = per_class_report(y, pred, k)
    for row in pc:
        row["label"] = labels[row.pop("class_index")]
    hier = hierarchy_summary(y, pred, labels)

    # ---- calibration (temperature fitted on val only)
    temp = fit_temperature(val_logits, y_val)
    cal: dict[str, Any] = {"temperature": temp, "fitted_on": "val NLL (LBFGS on log T)"}
    bins: dict[str, list[dict[str, float]]] = {}
    for split, lg, yy in (("test", ev_logits, y), ("val", val_logits, y_val)):
        e0, b0 = ece_bins(softmax(lg), yy, int(cfg["calibration"]["ece_bins"]))
        e1, b1 = ece_bins(softmax(lg, temp), yy, int(cfg["calibration"]["ece_bins"]))
        cal[split] = {
            "ece_before": e0, "ece_after": e1,
            "nll_before": nll(lg, yy), "nll_after": nll(lg, yy, temp),
        }  # fmt: skip
        bins[split] = {"before": b0, "after": b1}  # type: ignore[assignment]
    cal["val"]["note"] = "after-scaling val numbers are in-sample (T was fitted on val)"
    cal["bins_eval_split"] = bins["test"]

    # ---- selective prediction (temperature-scaled MSP)
    probs_t = softmax(ev_logits, temp)
    conf_t, conf_raw = probs_t.max(axis=1), probs.max(axis=1)
    correct = pred == y
    val_conf_t = softmax(val_logits, temp).max(axis=1)
    sel_cfg = cfg["selective"]
    thr = sel_cfg["threshold"]
    thr_src = "config override"
    if thr is None:
        thr = threshold_at_retention(val_conf_t, float(sel_cfg["retention"]))
        thr_src = f"{sel_cfg['retention']:.0%} retention on val (temperature-scaled MSP)"
    cov_t, risk_t = risk_coverage(conf_t, correct)
    cov_r, risk_r = risk_coverage(conf_raw, correct)
    selective = {
        "threshold_source": thr_src,
        "val_retention_at_threshold": float(np.mean(val_conf_t >= thr)),
        "at_threshold": selective_at_threshold(conf_t, y, pred, float(thr), k),
        "aurc_temp_scaled_msp": aurc(conf_t, correct),
        "aurc_raw_msp": aurc(conf_raw, correct),
        "aurc_note": "mean selective risk over k=1..n accepted rows; lower is better",
        "risk_coverage_temp_scaled": {"coverage": cov_t, "risk": risk_t},
    }

    # ---- baselines on the same rows
    train_df = get_frame(["train"], cfg["data_path"], cfg["splits_path"])
    train_df = train_df.sort_values("id").reset_index(drop=True)
    comparisons: dict[str, Any] = {}
    for kind in ("B0", "B1"):
        bdf, meta = baseline_predictions(kind, cfg, P, train_df, labels)
        if bdf["id"].tolist() != ev_ids:
            raise AssertionError(f"{kind}: id order differs from model predictions")
        bp = bdf["pred"].to_numpy()
        bci = bootstrap_metrics_ci(y, bp, k, n_res, seed)
        comparisons[kind] = {
            "fit": meta,
            "macro_f1": bci["macro_f1"],
            "accuracy": bci["accuracy"],
            "delta_macro_f1_final_minus_baseline": paired_bootstrap_delta_f1(
                y, pred, bp, k, n_res, seed
            ),
            "mcnemar_exact_on_correctness": mcnemar_exact(pred == y, bp == y),
        }

    # ---- CV side by side
    sel = json.loads(Path(cfg["selection_path"]).read_text())
    cid = cfg["analysis"]["oof_config_id"]
    cv_row = sel["candidates"][cid]
    oof = load_oof_mean(
        Path(cfg["analysis"]["oof_dir"]) / f"{cid}.csv",
        k,
        int(cfg["analysis"]["oof_predictions_per_id"]),
    )
    oof_ci = bootstrap_metrics_ci(oof["gold"].to_numpy(), oof["pred"].to_numpy(), k, n_res, seed)
    cv = {
        "label": CV_LABEL,
        "config_id": cid,
        "source": cfg["selection_path"],
        "chosen_epoch": cv_row["chosen_epoch"],
        "n_fold_runs": cv_row["n_runs"],
        "macro_f1_mean": cv_row["macro_f1_mean"],
        "macro_f1_std": cv_row["macro_f1_std"],
        "accuracy_mean": cv_row["accuracy_mean"],
        "accuracy_std": cv_row["accuracy_std"],
        "oof_probability_averaged_426_rows": {
            "note": "probabilities averaged over the 9 OOF predictions per id, then scored once",
            **oof_ci,
        },
    }

    # ---- slices + error analysis
    eda_rows = pd.read_csv(cfg["analysis"]["eda_rows_path"])
    ev_pred = pd.DataFrame({"id": ev_ids, "gold": y, "pred": pred, "conf": conf_t})
    sources = {
        "oof": oof[["id", "gold", "pred"]],
        "test": ev_pred[["id", "gold", "pred"]],
    }
    slices = {
        "label_oof": CV_LABEL + " (OOF probability-averaged predictions, 426 train+val rows)",
        "label_test": SMOKE_LABEL if smoke else TEST_LABEL,
        "english_vs_non_english": "definition A: primary language en vs not en/undetermined",
        "short": "n_chars < 30",
        "indicative_only_rule": "n < 20",
        "label_combined": COMBINED_LABEL,
        "macro_f1_definition": SLICE_MACRO_F1_DEFINITION,
        "bootstrap": {
            "n_resamples": n_res,
            "seed": seed,
            "macro_f1": "gold classes of the slice, fixed across resamples",
        },
        "sources": slice_report(sources, eda_rows, labels, n_res, seed)
        if not smoke
        else slice_report({"oof": sources["oof"]}, eda_rows, labels, n_res, seed),
    }
    if smoke:
        slices["sources"]["test_stand_in"] = slice_report(
            {"val": sources["test"]}, eda_rows, labels, n_res, seed
        )["val"]
    top_k, n_ex = int(cfg["analysis"]["top_confusions"]), int(cfg["analysis"]["examples_per_pair"])
    err_analysis = {
        "oof": {
            "label": CV_LABEL,
            "hypothesis_errors_concentrate_in_shipment_information": shipment_family_hypothesis(
                oof, labels
            ),
            "top_confusions": top_confusions(oof, labels, top_k, n_ex),
        },
        "test": {
            "label": SMOKE_LABEL if smoke else TEST_LABEL,
            "hypothesis_errors_concentrate_in_shipment_information": shipment_family_hypothesis(
                ev_pred, labels
            ),
            "top_confusions": top_confusions(ev_pred, labels, top_k, n_ex),
            "n_errors": int((~correct).sum()),
        },
    }
    errs = ranked_errors(ev_pred)
    raw_conf = dict(zip(ev_ids, conf_raw, strict=True))
    err_out = pd.DataFrame(
        {
            "id": errs["id"],
            "gold_label": [labels[i] for i in errs["gold"]],
            "pred_label": [labels[i] for i in errs["pred"]],
            "confidence": errs["conf"],  # temperature-scaled MSP
            "confidence_msp_raw": [float(raw_conf[i]) for i in errs["id"]],
        }
    )
    err_out.to_csv(P.results / "test_errors.csv", index=False, float_format="%.6g")
    _write_error_examples(cfg, P, err_analysis, errs, labels, smoke)

    # ---- figures
    figs = {
        "confusion_counts": P.figures / "final_confusion_counts.png",
        "confusion_norm": P.figures / "final_confusion_norm.png",
        "reliability": P.figures / "final_reliability.png",
        "risk_coverage": P.figures / "final_risk_coverage.png",
    }
    suffix = " [SMOKE val stand-in]" if smoke else " (test, single evaluation)"
    plot_confusion(
        conf_counts, labels, "Confusion matrix, counts" + suffix, figs["confusion_counts"], "d"
    )
    plot_confusion(
        row_normalise(conf_counts), labels, "Confusion matrix, row-normalised" + suffix,
        figs["confusion_norm"], ".2f",
    )  # fmt: skip
    plot_reliability(
        bins["test"]["before"], bins["test"]["after"], cal["test"]["ece_before"],
        cal["test"]["ece_after"], temp, figs["reliability"],
    )  # fmt: skip
    plot_risk_coverage(
        {"MSP (raw)": (cov_r, risk_r), "MSP (temperature-scaled)": (cov_t, risk_t)},
        selective["at_threshold"]["coverage"], figs["risk_coverage"],
    )  # fmt: skip

    # ---- assemble track_a.json
    run_train = json.loads((P.results / "train_summary.json").read_text())
    track_a = {
        "smoke": smoke,
        "git_sha": git_sha(),
        "model_fingerprint": run_train["model_fingerprint"],
        "inference": "fp32, eval mode (training used fp16 autocast)",
        "test": {
            "label": SMOKE_LABEL if smoke else TEST_LABEL,
            "n": len(y),
            "bootstrap": {
                "n_resamples": n_res,
                "seed": seed,
                "method": "percentile, plain row resampling",
                "level": 0.95,
            },  # fmt: skip
            "macro_f1": ci["macro_f1"],
            "accuracy": ci["accuracy"],
            "classes": names,
            "per_class": pc,
            "confusion_counts": conf_counts,
            "confusion_row_normalised": row_normalise(conf_counts),
            "hierarchy": hier,
            "selective": selective,
            "baselines": comparisons,
        },
        "calibration": cal,
        "cv": cv,
        "val_sanity": {
            "macro_f1": macro_f1(y_val, val_logits.argmax(axis=1), k),
            "accuracy": accuracy(y_val, val_logits.argmax(axis=1)),
            "macro_f1_present": macro_f1_present(y_val, val_logits.argmax(axis=1), k),
            "n": len(y_val),
        },
        "figures": {k_: str(v) for k_, v in figs.items()},
        "test_eval_log": str(P.test_log),
    }
    wb_url = None
    if use_wandb:
        summ = {
            "test/macro_f1": ci["macro_f1"]["point"], "test/accuracy": ci["accuracy"]["point"],
            "test/label": track_a["test"]["label"], "cv/label": CV_LABEL,
            "cv/macro_f1_mean": cv["macro_f1_mean"], "calibration/temperature": temp,
            "calibration/ece_test_before": cal["test"]["ece_before"],
            "calibration/ece_test_after": cal["test"]["ece_after"],
            "selective/aurc": selective["aurc_temp_scaled_msp"],
            "model_fingerprint": track_a["model_fingerprint"],
        }  # fmt: skip
        wb_url = _wandb_final(
            cfg,
            summ,
            pd.DataFrame(pc),
            y,
            pred,
            labels,
            err_out,
            (cov_t, risk_t),
            figs,
        )
    track_a["wandb_url"] = wb_url
    write_json(P.results / "track_a.json", track_a)
    write_json(P.results / "slices.json", slices)
    write_json(P.results / "error_analysis.json", err_analysis)
    print(f"[evaluate] test macro-F1 {ci['macro_f1']} accuracy {ci['accuracy']}")


def _write_error_examples(
    cfg: dict[str, Any], P: Paths, err: dict[str, Any], test_errs: pd.DataFrame,
    labels: list[str], smoke: bool,
) -> None:  # fmt: skip
    """Example TEXTS go to the gitignored outputs dir only (never results/)."""
    text_of = dict(zip(data_mod.load_data(cfg["data_path"])["id"],
                       data_mod.load_data(cfg["data_path"])["text"], strict=True))  # fmt: skip
    lines = ["# Error examples (CONFIDENTIAL text; gitignored; do not copy into results/)\n"]
    for src in ("oof", "test"):
        lines.append(f"\n## {src} top confusion pairs\n")
        for pair in err[src]["top_confusions"]:
            lines.append(f"\n### {pair['gold']} -> {pair['pred']} (n={pair['count']})\n")
            lines += [f"- `{i}`: {text_of[i]}" for i in pair["example_ids"]]
    lines.append("\n## every evaluated-split error (most confident first)\n")
    for _, r in test_errs.iterrows():
        lines.append(
            f"- `{r['id']}` gold={labels[r['gold']]} pred={labels[r['pred']]} "
            f"conf={r['conf']:.3f}: {text_of[r['id']]}"
        )
    P.outputs.mkdir(parents=True, exist_ok=True)
    (P.outputs / "error_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Final model + Track A")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, VAL ONLY, outputs/final_smoke")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    use_wandb = bool(cfg["wandb"]["enabled"]) and not args.no_wandb and not args.smoke
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    if args.smoke and args.stage != "all":
        raise SystemExit("--smoke runs the whole pipeline on val: use --stage all")
    if args.stage == "analysis":  # saved files only; never inference (see analysis.main)
        from intent_router.analysis import main as analysis_main

        analysis_main(["--config", args.config])
        return
    P = make_paths(cfg, args.smoke)
    if args.stage in ("train", "all"):
        stage_train(cfg, P, args.smoke)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(cfg, P, args.smoke, use_wandb)


if __name__ == "__main__":
    main()
