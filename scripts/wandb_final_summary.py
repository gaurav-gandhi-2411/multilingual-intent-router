"""Log the curated W&B run `final-v1-summary` from SAVED results only.

No training and no test inference happens here: every number, curve and table is read from files
committed under results/ (written by the one logged v1 evaluation). The run carries ids, labels,
confidences and numbers only; dataset message text is never read (test_errors.csv's text-free
columns are whitelisted, data/dataset.csv is never opened). The project's visibility is only
READ through the API and reported, never changed.

Usage:
    python scripts/wandb_final_summary.py --dry-run     # WANDB_MODE=offline, ./wandb/offline-*
    python scripts/wandb_final_summary.py               # real online run (id final-v1-summary)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures are only rendered to wandb.Image
import matplotlib.pyplot as plt  # noqa: E402 - after backend selection on purpose
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

PROJECT = "intent-router"
GROUP = "final"
RUN_ID = "final-v1-summary"
GRAPHQL_URL = "https://api.wandb.ai/graphql"
TAGS = ["final", "v1", "summary", "track-a", "track-b", "saved-results-only"]
RESULTS = Path("results")
FINAL = RESULTS / "final"
TRACKB = RESULTS / "trackb"
SEEDS = (42, 43, 44)
# Whitelist for the misclassification table: NO message text may ever be logged.
ERROR_COLUMNS = ("id", "gold_label", "pred_label", "confidence")
# Track B methods plotted from the saved score CSVs (higher saved score == more "known").
PLOT_METHODS = ("maha_ft", "msp_temp")
SOURCE_FILES = (
    FINAL / "model_version.json",
    FINAL / "train_summary.json",
    FINAL / "track_a.json",
    FINAL / "test_predictions.csv",
    FINAL / "test_errors.csv",
    TRACKB / "headline.json",
    *(TRACKB / "scores" / f"headline_s{s}.csv" for s in SEEDS),
)


# ------------------------------------------------------------------ pure data preparation
def lr_at_epoch_ends(
    n_train: int, batch_size: int, epochs: int, warmup_ratio: float, lr: float, upto_epoch: int
) -> list[float]:
    """LR after the last step of each epoch for HF's linear warmup + linear decay schedule.

    The schedule horizon is `epochs` (20) even though training stops at `upto_epoch` (9); this is
    derived from the config, because the per-step LR was not saved. It mirrors
    get_linear_schedule_with_warmup(optim, int(warmup_ratio*total), total).
    """
    steps_per_epoch = math.ceil(n_train / batch_size)
    total = steps_per_epoch * epochs
    warm = int(warmup_ratio * total)
    out = []
    for epoch in range(1, upto_epoch + 1):
        step = epoch * steps_per_epoch
        factor = (
            step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
        )
        out.append(lr * factor)
    return out


def epoch_records(run: dict[str, Any], lrs: list[float]) -> list[dict[str, float]]:
    """One dict per epoch (step = epoch) with train/val loss, val macro-F1/accuracy and LR."""
    recs = []
    for ep, lr in zip(run["epochs"], lrs, strict=True):
        recs.append(
            {
                "epoch": int(ep["epoch"]),
                "train/loss": float(ep["train_loss"]),
                "val/loss": float(ep["eval_loss"]),
                "val/macro_f1": float(ep["macro_f1"]),
                "val/accuracy": float(ep["accuracy"]),
                "lr": float(lr),
            }
        )
    return recs


def per_class_rows(track_a: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Test per-class precision / recall / F1 / support rows (labels are class names, not text)."""
    cols = ["label", "precision", "recall", "f1", "support"]
    rows = [
        [c["label"], c["precision"], c["recall"], c["f1"], c["support"]]
        for c in track_a["test"]["per_class"]
    ]
    return cols, rows


def error_rows(errors: pd.DataFrame) -> list[list[Any]]:
    """Misclassification rows restricted to ids, gold, predicted and confidence (no text)."""
    missing = [c for c in ERROR_COLUMNS if c not in errors.columns]
    if missing:
        raise ValueError(f"test_errors.csv lacks columns {missing}")
    return errors[list(ERROR_COLUMNS)].values.tolist()


def class_names(track_a: dict[str, Any]) -> list[str]:
    """Class names ordered by class index."""
    return [c["label"] for c in sorted(track_a["test"]["classes"], key=lambda c: c["index"])]


def confusion_inputs(preds: pd.DataFrame) -> tuple[list[int], list[int]]:
    """(y_true, y_pred) integer class indices from the saved test_predictions.csv."""
    return [int(v) for v in preds["gold"]], [int(v) for v in preds["pred"]]


def track_a_summary(track_a: dict[str, Any], train_summary: dict[str, Any]) -> dict[str, float]:
    """Flat wandb.summary entries for Track A: point estimates with bootstrap CIs, val, CV."""
    t = track_a["test"]
    out: dict[str, float] = {"test/n": float(t["n"])}
    for m in ("macro_f1", "accuracy"):
        out[f"test/{m}"] = t[m]["point"]
        out[f"test/{m}_ci_lo"] = t[m]["lo"]
        out[f"test/{m}_ci_hi"] = t[m]["hi"]
    out["val/macro_f1_fp32_run1"] = train_summary["val_macro_f1_fp32_run1"]
    cv = track_a["cv"]
    out["cv/macro_f1_mean"] = cv["macro_f1_mean"]
    out["cv/macro_f1_std"] = cv["macro_f1_std"]
    out["calibration/temperature"] = track_a["calibration"]["temperature"]
    out["calibration/test_ece_before"] = track_a["calibration"]["test"]["ece_before"]
    out["calibration/test_ece_after"] = track_a["calibration"]["test"]["ece_after"]
    return out


def headline_summary(
    headline: dict[str, Any], methods: tuple[str, ...] = ("maha_ft", "msp")
) -> dict[str, float]:
    """Flat wandb.summary entries for the Track B headline holdout (mean/std over 3 seeds)."""
    out: dict[str, float] = {
        "trackb/n_eval_known": float(headline["n_eval_known"]),
        "trackb/n_eval_unknown": float(headline["n_eval_unknown"]),
        "trackb/n_seeds": float(len(headline["seeds"])),
    }
    for method in methods:
        agg = headline["methods"][method]
        for key in ("auroc", "aupr", "fpr_at_95tpr", "strict_rejection_recall", "retention_known"):
            out[f"trackb/{method}/{key}_mean"] = agg[key]["mean"]
            out[f"trackb/{method}/{key}_std"] = agg[key]["std"]
    return out


def unknown_score(scores: pd.DataFrame, method: str) -> np.ndarray:
    """Eval-row score where HIGHER means more likely unknown (saved scores are 'higher = known')."""
    ev = scores[scores["set"] == "eval"]
    return -ev[method].to_numpy(dtype=float)


def eval_labels(scores: pd.DataFrame) -> np.ndarray:
    """1 for unknown-class eval rows (the positive class), 0 for known."""
    ev = scores[scores["set"] == "eval"]
    return ev["is_unknown"].to_numpy(dtype=bool).astype(int)


def roc_auroc(scores: pd.DataFrame, method: str) -> float:
    """AUROC (unknown = positive) recomputed from a saved score CSV."""
    return float(roc_auc_score(eval_labels(scores), unknown_score(scores, method)))


def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file (provenance of every source file in the run config)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    """Current HEAD sha, or 'unknown'."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_config(
    model_version: dict[str, Any], train_summary: dict[str, Any], n_train: int
) -> dict[str, Any]:
    """Run config: train config, model fingerprint, code git SHAs, source-file hashes."""
    return {
        "train_config": model_version["train_config"],
        "model_version": model_version["version"],
        "deployed_epoch": model_version["deployed_epoch"],
        "model_fingerprint": model_version["model_fingerprint"],
        "shipped_ood_score": model_version["shipped_ood_score"],
        "seed": 42,
        "n_train": n_train,
        "results_git_sha": train_summary["git_sha"],  # code that produced the saved results
        "summary_script_git_sha": git_head(),
        "source_sha256": {str(p): sha256_file(p) for p in SOURCE_FILES},
        "note": "logged from saved results only; no retraining and no test inference",
    }


# ------------------------------------------------------------------------------ figures
def plot_trackb_hist(scores_by_seed: dict[int, pd.DataFrame], method: str) -> plt.Figure:
    """Known vs unknown eval-score histograms, one panel per model seed."""
    fig, axes = plt.subplots(
        1, len(scores_by_seed), figsize=(4 * len(scores_by_seed), 3), sharey=True
    )
    for ax, (seed, df) in zip(np.atleast_1d(axes), scores_by_seed.items(), strict=True):
        ev = df[df["set"] == "eval"]
        bins = np.histogram_bin_edges(ev[method], bins=20)
        ax.hist(ev.loc[~ev["is_unknown"].astype(bool), method], bins=bins, alpha=0.6, label="known")
        ax.hist(
            ev.loc[ev["is_unknown"].astype(bool), method], bins=bins, alpha=0.6, label="unknown"
        )
        ax.set_title(f"{method}, seed {seed}")
        ax.set_xlabel("score (higher = more known)")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    return fig


def plot_trackb_roc(
    scores_by_seed: dict[int, pd.DataFrame], methods: tuple[str, ...]
) -> plt.Figure:
    """ROC (unknown = positive) per method and seed from the saved score CSVs."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for method in methods:
        for seed, df in scores_by_seed.items():
            fpr, tpr, _ = roc_curve(eval_labels(df), unknown_score(df, method))
            ax.plot(fpr, tpr, label=f"{method} s{seed} (AUROC {roc_auroc(df, method):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false positive rate (known flagged unknown)")
    ax.set_ylabel("true positive rate (unknown flagged)")
    ax.set_title("Track B headline holdout: ROC")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def check_scores_reproduce_headline(
    scores_by_seed: dict[int, pd.DataFrame], headline: dict[str, Any], tol: float = 1e-3
) -> None:
    """Self-check: AUROC recomputed from saved score CSVs equals the saved headline AUROC."""
    for seed, df in scores_by_seed.items():
        saved = headline["runs"][f"headline_s{seed}"]["per_method"]["maha_ft"]["auroc"]
        got = roc_auroc(df, "maha_ft")
        if abs(got - saved) > tol:
            raise AssertionError(f"seed {seed}: recomputed AUROC {got:.4f} != saved {saved:.4f}")


# ----------------------------------------------------------------------------------- main
def _visibility_report(api: Any, entity: str) -> dict[str, Any]:
    """Read (never write) visibility with two GraphQL reads of the same project.

    - anonymous (`trust_env=False`: requests would otherwise pick up the netrc credentials and
      this would silently be an authenticated call): `project` must come back null for a
      private project, i.e. the public internet cannot see it or its run;
    - authenticated (the wandb API): the project's `access` field and the run's state.
    """
    import requests

    out: dict[str, Any] = {"entity": entity}
    query = (
        "query($n:String!,$e:String!,$r:String!){project(name:$n,entityName:$e)"
        "{name access run(name:$r){name}}}"
    )
    body = {"query": query, "variables": {"n": PROJECT, "e": entity, "r": RUN_ID}}
    try:
        anon = requests.Session()
        anon.trust_env = False  # no netrc, no env proxies/creds: a genuinely anonymous caller
        out["anonymous_query"] = anon.post(GRAPHQL_URL, json=body, timeout=30).json()
        out["anonymous_can_see_project"] = (
            out["anonymous_query"].get("data", {}).get("project") is not None
        )
    except Exception as exc:  # noqa: BLE001 - report, do not fail the run
        out["anonymous_query_error"] = repr(exc)
    try:
        key = api.api_key
        auth = requests.post(GRAPHQL_URL, json=body, auth=("api", key), timeout=30).json()
        out["authenticated_query"] = auth
        run = api.run(f"{entity}/{PROJECT}/{RUN_ID}")
        out["authenticated_run_url"] = run.url
        out["authenticated_run_state"] = run.state
    except Exception as exc:  # noqa: BLE001
        out["authenticated_error"] = repr(exc)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--dry-run", action="store_true", help="WANDB_MODE=offline; nothing is uploaded"
    )
    ap.add_argument("--entity", default=None, help="default: discovered via the wandb API")
    args = ap.parse_args(argv)

    model_version = json.loads((FINAL / "model_version.json").read_text())
    train_summary = json.loads((FINAL / "train_summary.json").read_text())
    track_a = json.loads((FINAL / "track_a.json").read_text())
    headline = json.loads((TRACKB / "headline.json").read_text())
    preds = pd.read_csv(FINAL / "test_predictions.csv")
    errors = pd.read_csv(FINAL / "test_errors.csv")
    scores = {s: pd.read_csv(TRACKB / "scores" / f"headline_s{s}.csv") for s in SEEDS}
    check_scores_reproduce_headline(scores, headline)

    splits = pd.read_csv("splits/splits.csv")
    n_train = int((splits["split"] == "train").sum())
    cfg = model_version["train_config"]
    run1 = train_summary["runs"]["run1"]
    lrs = lr_at_epoch_ends(
        n_train,
        cfg["batch_size"],
        cfg["epochs"],
        cfg["warmup_ratio"],
        cfg["lr"],
        len(run1["epochs"]),
    )
    records = epoch_records(run1, lrs)

    if args.dry_run:
        os.environ["WANDB_MODE"] = "offline"
    import wandb

    entity = args.entity
    api = None
    if not args.dry_run:
        api = wandb.Api()
        entity = entity or api.default_entity
        try:
            api.run(f"{entity}/{PROJECT}/{RUN_ID}")
            print(
                f"run {RUN_ID} already exists at {entity}/{PROJECT}: not re-logging "
                f"(delete it manually to regenerate)"
            )
            print(json.dumps(_visibility_report(api, entity), indent=1, default=str))
            return
        except wandb.errors.CommError:
            pass  # not found: create it

    run = wandb.init(
        project=PROJECT,
        entity=entity,
        group=GROUP,
        name=RUN_ID,
        id=RUN_ID,
        resume="allow",
        job_type="summary",
        tags=TAGS,
        config=build_config(model_version, train_summary, n_train),
    )
    for rec in records:
        step = rec["epoch"]
        run.log({k: v for k, v in rec.items() if k != "epoch"}, step=step)

    last = records[-1]["epoch"]
    labels = class_names(track_a)
    y_true, y_pred = confusion_inputs(preds)
    cols, rows = per_class_rows(track_a)
    figs: dict[str, Any] = {}
    for method in PLOT_METHODS:
        figs[f"trackb/hist_{method}"] = wandb.Image(plot_trackb_hist(scores, method))
    figs["trackb/roc"] = wandb.Image(plot_trackb_roc(scores, PLOT_METHODS))
    plt.close("all")
    saved_pngs = {
        "final/confusion_counts": "final_confusion_counts.png",
        "final/confusion_norm": "final_confusion_norm.png",
        "final/reliability": "final_reliability.png",
        "final/risk_coverage": "final_risk_coverage.png",
        "trackb/saved_hist_msp": "trackb_headline_hist_msp.png",
        "trackb/saved_hist_maha_ft": "trackb_headline_hist_maha_ft.png",
        "trackb/saved_hist_maha_frozen": "trackb_headline_hist_maha_frozen.png",
        "trackb/saved_roc": "trackb_headline_roc.png",
        "trackb/saved_risk_coverage": "trackb_headline_risk_coverage.png",
    }
    for key, name in saved_pngs.items():
        figs[key] = wandb.Image(str(RESULTS / "figures" / name))
    run.log(
        {
            "test/per_class": wandb.Table(columns=cols, data=rows),
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=labels
            ),
            "test/misclassifications": wandb.Table(
                columns=list(ERROR_COLUMNS), data=error_rows(errors)
            ),
            **figs,
        },
        step=last,
    )
    run.summary.update({**track_a_summary(track_a, train_summary), **headline_summary(headline)})
    url = getattr(run, "url", None)
    mode = "offline" if args.dry_run else "online"
    print(f"run logged ({mode}): {url or run.dir}")
    run.finish()

    if api is not None and entity:
        print(json.dumps(_visibility_report(api, entity), indent=1, default=str))


if __name__ == "__main__":
    main()
