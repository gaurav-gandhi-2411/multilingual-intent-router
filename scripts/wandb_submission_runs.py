"""Log the curated runs of the submission W&B project `multilingual-intent-router`.

Every number is read from SAVED files under results/ (no training, no test inference; the live
training run is scripts/wandb_final_rerun.py). Runs (deterministic ids, resume="allow"; a re-run
appends one history row and overwrites summaries/tables, it never creates a duplicate run):

    final-v1-eval-saved   group final           saved single test evaluation (NOT recomputed)
    bakeoff-summary       group bakeoff         model bake-off aggregates + selection
    trackb-headline       group trackB-headline 3 seeds, 10 OOD scores, curves from score CSVs
    tradeoff-v1-v3        group tradeoff        v1 vs v3 tables from results/tradeoff_v1_v3.*
    llm-baseline          group llm-baseline    zero/few-shot LLM vs fine-tuned model

No dataset message text is read or logged: only ids, labels, confidences and numbers (the
misclassification table is whitelisted to those columns). Project visibility is never changed.

Usage:
    python scripts/wandb_submission_runs.py --dry-run [--only NAME ...]   # WANDB_MODE=offline
    python scripts/wandb_submission_runs.py [--only NAME ...]             # online
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENTITY = "gauravgandhi429-gaurav-gandhi"
PROJECT = "multilingual-intent-router"
RESULTS = Path("results")
FINAL = RESULTS / "final"
TRACKB = RESULTS / "trackb"
SEEDS = (42, 43, 44)
SAVED_LABEL = "saved single evaluation, not recomputed"
RUNS: dict[str, dict[str, Any]] = {
    "eval-saved": {"id": "final-v1-eval-saved", "group": "final"},
    "bakeoff": {"id": "bakeoff-summary", "group": "bakeoff"},
    "trackb": {"id": "trackb-headline", "group": "trackB-headline"},
    "tradeoff": {"id": "tradeoff-v1-v3", "group": "tradeoff"},
    "llm": {"id": "llm-baseline", "group": "llm-baseline"},
}


def _load_summary_module() -> Any:
    """The old project's curated-run script, reused for its pure data-prep functions."""
    path = ROOT / "scripts" / "wandb_final_summary.py"
    spec = importlib.util.spec_from_file_location("wandb_final_summary", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ pure data preparation
def parse_md_tables(md: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """(section heading, columns, rows) for every pipe table in a markdown document."""
    out: list[tuple[str, list[str], list[list[str]]]] = []
    heading = ""
    cur: list[list[str]] = []

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 2:
            cols, rows = cur[0], [r for r in cur[2:] if len(r) == len(cur[0])]
            out.append((heading, cols, rows))
        cur = []

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            heading = s.lstrip("#").strip()
        elif s.startswith("|"):
            cur.append([c.strip() for c in s.strip("|").split("|")])
        else:
            flush()
    flush()
    return out


def md_bullets(md: str, heading_prefix: str) -> list[str]:
    """Bullet lines ('- ...') under the first heading that starts with `heading_prefix`."""
    lines, inside, out = md.splitlines(), False, []
    for line in lines:
        if line.startswith("#"):
            inside = line.lstrip("#").strip().startswith(heading_prefix)
        elif inside and line.startswith("- "):
            out.append(line[2:].strip())
    return out


def slug(text: str) -> str:
    """Lowercase, [a-z0-9_] key fragment for a W&B table name."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def bakeoff_config_rows(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """One row per bake-off config from results/bakeoff/summary.json (CV aggregates)."""
    cols = [
        "config", "n_runs", "n_exclusive", "chosen_epoch", "macro_f1_mean", "macro_f1_std",
        "accuracy_mean", "accuracy_std", "mean_wall_clock_s", "max_peak_vram_mb",
    ]  # fmt: skip
    rows = [[name, *(c.get(k) for k in cols[1:])] for name, c in summary["configs"].items()]
    return cols, rows


def bakeoff_baseline_rows(baselines: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """B0 (TF-IDF LR) / B1 (frozen e5 + LR) CV aggregates."""
    cols = ["baseline", "n_fold_runs", "macro_f1_mean", "macro_f1_std", "accuracy_mean"]
    rows = [
        [k, v["n_fold_runs"], v["macro_f1_mean"], v["macro_f1_std"], v["accuracy_mean"]]
        for k, v in baselines.items()
    ]
    return cols, rows


def bakeoff_selection_rows(sel: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Candidate rows (best LR per model) with a `selected` flag."""
    cols = [
        "candidate", "selected", "chosen_epoch", "n_runs", "macro_f1_mean", "macro_f1_std",
        "accuracy_mean", "params", "mean_wall_clock_s",
    ]  # fmt: skip
    rows = [
        [name, name == sel["winner"], *(c.get(k) for k in cols[2:])]
        for name, c in sel["candidates"].items()
    ]
    return cols, rows


def calibration_rows(track_a: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Temperature scaling and ECE/NLL before/after on val (fit) and test (saved)."""
    cal = track_a["calibration"]
    cols = ["split", "temperature", "ece_before", "ece_after", "nll_before", "nll_after"]
    rows = [
        [s, cal["temperature"], cal[s]["ece_before"], cal[s]["ece_after"],
         cal[s].get("nll_before"), cal[s].get("nll_after")]
        for s in ("val", "test")
    ]  # fmt: skip
    return cols, rows


def trackb_method_rows(headline: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Per-method mean/std over the headline seeds: AUROC, strict rejection@95, retention."""
    keys = (
        "auroc", "aupr", "fpr_at_95tpr", "strict_rejection_recall", "retention_known",
        "known_accepted_macro_f1_present",
    )  # fmt: skip
    cols = ["method", "n_seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")]
    rows = []
    for method, agg in headline["methods"].items():
        vals = [v for k in keys for v in (agg[k]["mean"], agg[k]["std"])]
        rows.append([method, agg["auroc"]["n"], *vals])
    return cols, rows


def trackb_seed_rows(headline: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Operating points per seed and method (threshold set on the cal split at 95% retention)."""
    cols = [
        "seed", "method", "auroc", "fpr_at_95tpr", "threshold", "retention_known",
        "strict_rejection_recall", "n_unknown_not_rejected",
    ]  # fmt: skip
    rows = []
    for run in headline["runs"].values():
        for method, m in run["per_method"].items():
            rows.append([run["seed"], method, *(m.get(k) for k in cols[2:])])
    return cols, rows


def llm_rows(summary: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """LLM zero/few-shot rows (Track A, Track B, latency, cost) plus the fine-tuned model."""
    cols = [
        "system", "track_a_macro_f1", "track_a_macro_f1_lo", "track_a_macro_f1_hi",
        "track_a_accuracy", "parse_failure_rate", "delta_macro_f1_vs_final",
        "trackb_strict_rejection_recall", "trackb_retention_known",
        "latency_p50_s", "latency_p95_s", "usd_per_1k_messages_estimate",
    ]  # fmt: skip
    ta, tb, lat, cost = summary["track_a"], summary["track_b"], summary["latency"], summary["cost"]
    rows: list[list[Any]] = []
    for name, a in ta["llm"].items():
        b = tb["llm"].get(name, {})
        la = lat["llm"]["configs"].get(name, {})
        c = cost["rows"].get(f"llm|{name}", {})
        rows.append([
            f"llm|{name}", a["macro_f1"]["point"], a["macro_f1"]["lo"], a["macro_f1"]["hi"],
            a["accuracy"]["point"], a.get("parse_failure_rate"),
            a.get("delta_macro_f1_llm_minus_final", {}).get("delta"),
            b.get("strict_rejection_recall"), b.get("retention_known"),
            la.get("p50_s"), la.get("p95_s"), c.get("usd_per_1k_messages"),
        ])  # fmt: skip
    f, ship = ta["final_model"], tb["shipped_maha_ft"]
    gpu = lat["finetuned"].get("gpu_batch1", {})
    rows.append([
        "fine-tuned e5-base (shipped v1)", f["macro_f1"]["point"], f["macro_f1"]["lo"],
        f["macro_f1"]["hi"], f["accuracy"]["point"], None, 0.0,
        ship["strict_rejection_recall"]["mean"], ship["retention_known"]["mean"],
        gpu.get("p50_s"), gpu.get("p95_s"), cost["rows"].get("ft|gpu_batch1", {}).get(
            "usd_per_1k_messages"
        ),
    ])  # fmt: skip
    return cols, rows


# ------------------------------------------------------------------------------ plumbing
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _init(key: str, mod: Any, sources: list[Path], tags: list[str], extra: dict[str, Any]) -> Any:
    """wandb.init with a deterministic id; config carries source hashes for provenance."""
    import wandb

    spec = RUNS[key]
    config = {
        "label": SAVED_LABEL if key == "eval-saved" else "logged from saved results only",
        "note": "no training and no test inference in this run",
        "source_sha256": {p.as_posix(): mod.sha256_file(p) for p in sources},
        "summary_script_git_sha": mod.git_head(),
        **extra,
    }
    return wandb.init(
        project=PROJECT, entity=ENTITY, group=spec["group"], name=spec["id"], id=spec["id"],
        resume="allow", job_type="summary", tags=tags, config=config,
    )  # fmt: skip


def _table(cols: list[str], rows: list[list[Any]]) -> Any:
    import wandb

    return wandb.Table(columns=cols, data=rows)


def _png(name: str) -> Any:
    import wandb

    return wandb.Image(str(RESULTS / "figures" / name))


# ------------------------------------------------------------------------------ the runs
def log_eval_saved(mod: Any) -> Any:
    """Saved single test evaluation of the shipped v1 model (labelled, not recomputed)."""
    import wandb

    mv = _read_json(FINAL / "model_version.json")
    ta = _read_json(FINAL / "track_a.json")
    ts = _read_json(FINAL / "train_summary.json")
    preds = pd.read_csv(FINAL / "test_predictions.csv")
    errors = pd.read_csv(FINAL / "test_errors.csv")
    sources = [
        FINAL / "model_version.json", FINAL / "track_a.json", FINAL / "train_summary.json",
        FINAL / "test_predictions.csv", FINAL / "test_errors.csv",
    ]  # fmt: skip
    run = _init(
        "eval-saved",
        mod,
        sources,
        ["final", "v1", "track-a", "saved-not-recomputed"],
        {
            "model_fingerprint": mv["model_fingerprint"],
            "results_git_sha": ts["git_sha"],
            "live_training_run": "final-v1-train",
            "eval_provenance": (
                "one guarded evaluation on the test split, written by intent_router.final; "
                "read here from results/final, never re-run"
            ),
        },
    )
    cols, rows = mod.per_class_rows(ta)
    y_true, y_pred = mod.confusion_inputs(preds)
    cal_cols, cal_rows = calibration_rows(ta)
    run.log(
        {
            "test/per_class": _table(cols, rows),
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=mod.class_names(ta)
            ),
            "test/calibration": _table(cal_cols, cal_rows),
            "test/misclassifications": _table(list(mod.ERROR_COLUMNS), mod.error_rows(errors)),
            "test/confusion_counts_png": _png("final_confusion_counts.png"),
            "test/confusion_norm_png": _png("final_confusion_norm.png"),
            "test/reliability_png": _png("final_reliability.png"),
            "test/risk_coverage_png": _png("final_risk_coverage.png"),
        }
    )
    summ = mod.track_a_summary(ta, ts)
    run.summary.update({**summ, "label": SAVED_LABEL})
    return run


def log_bakeoff(mod: Any) -> Any:
    """Bake-off aggregates (CV, 3 models x LRs + ablations), baselines and the selection."""
    bake = RESULTS / "bakeoff"
    summary, baselines, sel = (
        _read_json(bake / n) for n in ("summary.json", "baselines.json", "selection.json")
    )
    run = _init(
        "bakeoff",
        mod,
        [bake / "summary.json", bake / "baselines.json", bake / "selection.json"],
        ["bakeoff", "cv", "model-selection"],
        {"winner": sel["winner"], "selection_rule": "mean CV macro-F1 at chosen epoch; see tables"},
    )
    c_cols, c_rows = bakeoff_config_rows(summary)
    b_cols, b_rows = bakeoff_baseline_rows(baselines)
    s_cols, s_rows = bakeoff_selection_rows(sel)
    cfg_table = _table(c_cols, c_rows)
    import wandb

    run.log(
        {
            "bakeoff/configs": cfg_table,
            "bakeoff/baselines": _table(b_cols, b_rows),
            "bakeoff/selection": _table(s_cols, s_rows),
            "bakeoff/macro_f1_by_config": wandb.plot.bar(
                cfg_table, "config", "macro_f1_mean", title="CV macro-F1 (mean) by config"
            ),
        }
    )
    t2 = sel["top2_ttest"]
    run.summary.update(
        {
            "selection/winner": sel["winner"],
            "selection/top2_p": t2["p"],
            "selection/top2_mean_diff": t2["mean_diff"],
            "baseline/B0_macro_f1_mean": baselines["B0"]["macro_f1_mean"],
            "baseline/B1_macro_f1_mean": baselines["B1"]["macro_f1_mean"],
        }
    )
    return run


def log_trackb(mod: Any) -> Any:
    """Track B headline holdout: 3 seeds x 10 scores, curves recomputed from saved score CSVs."""
    headline = _read_json(TRACKB / "headline.json")
    scores = {s: pd.read_csv(TRACKB / "scores" / f"headline_s{s}.csv") for s in SEEDS}
    mod.check_scores_reproduce_headline(scores, headline)
    sources = [TRACKB / "headline.json", *(TRACKB / "scores" / f"headline_s{s}.csv" for s in SEEDS)]
    run = _init(
        "trackb",
        mod,
        sources,
        ["trackb", "open-set", "headline-holdout"],
        {"holdout_classes": headline["holdout"], "seeds": list(SEEDS)},
    )
    import matplotlib.pyplot as plt
    import wandb

    m_cols, m_rows = trackb_method_rows(headline)
    s_cols, s_rows = trackb_seed_rows(headline)
    logged: dict[str, Any] = {
        "trackb/methods": _table(m_cols, m_rows),
        "trackb/operating_points": _table(s_cols, s_rows),
        "trackb/roc": wandb.Image(mod.plot_trackb_roc(scores, mod.PLOT_METHODS)),
        "trackb/saved_roc": _png("trackb_headline_roc.png"),
        "trackb/saved_risk_coverage": _png("trackb_headline_risk_coverage.png"),
    }
    for method in mod.PLOT_METHODS:
        logged[f"trackb/hist_{method}"] = wandb.Image(mod.plot_trackb_hist(scores, method))
    for name in ("msp", "maha_ft", "maha_frozen"):
        logged[f"trackb/saved_hist_{name}"] = _png(f"trackb_headline_hist_{name}.png")
    plt.close("all")
    run.log(logged)
    run.summary.update(mod.headline_summary(headline, tuple(headline["methods"])))
    return run


def log_tradeoff(mod: Any) -> Any:
    """v1 vs v3 tables, verdict and deployment guidance from results/tradeoff_v1_v3.*."""
    md_path, js_path = RESULTS / "tradeoff_v1_v3.md", RESULTS / "tradeoff_v1_v3.json"
    md = md_path.read_text(encoding="utf-8")
    js = _read_json(js_path)
    run = _init(
        "tradeoff",
        mod,
        [md_path, js_path],
        ["tradeoff", "v1", "v3"],
        {
            "v1_fingerprint": js["versions"]["v1"]["model_fingerprint"]["value"],
            "v3_fingerprint": js["versions"]["v3"]["model_fingerprint"]["value"],
            "shipped": "v1",
        },
    )
    logged: dict[str, Any] = {}
    for heading, cols, rows in parse_md_tables(md):
        logged[f"tradeoff/{slug(heading)}"] = _table(cols, rows)
    for part, key in (("Verdict", "verdict"), ("Deployment guidance", "guidance")):
        logged[f"tradeoff/{key}"] = _table(["statement"], [[b] for b in md_bullets(md, part)])
    run.log(logged)
    run.summary.update({"shipped": "v1", "n_tables": len(logged)})
    return run


def log_llm(mod: Any) -> Any:
    """Zero/few-shot LLM baseline vs the fine-tuned model (accuracy, open-set, latency, cost)."""
    path = RESULTS / "llm_baseline" / "summary.json"
    summary = _read_json(path)
    run = _init(
        "llm",
        mod,
        [path],
        ["llm-baseline", "track-a", "track-b"],
        {"results_git_sha": summary["git_sha"], "cost_label": summary["cost"]["label"]},
    )
    cols, rows = llm_rows(summary)
    run.log({"llm/comparison": _table(cols, rows)})
    run.summary.update({f"llm/{r[0]}/macro_f1": r[1] for r in rows})
    return run


LOGGERS = {
    "eval-saved": log_eval_saved,
    "bakeoff": log_bakeoff,
    "trackb": log_trackb,
    "tradeoff": log_tradeoff,
    "llm": log_llm,
}


def main(argv: list[str] | None = None) -> None:
    """Log the selected runs (default: all)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="WANDB_MODE=offline; nothing uploaded")
    ap.add_argument("--only", nargs="+", choices=sorted(LOGGERS), help="subset of runs")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    os.environ["WANDB_DIR"] = "outputs"
    if args.dry_run:
        os.environ["WANDB_MODE"] = "offline"
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_summary_module()
    for key in args.only or list(LOGGERS):
        run = LOGGERS[key](mod)
        print(f"{key}: {run.id} -> {getattr(run, 'url', None) or run.dir}")
        run.finish()


if __name__ == "__main__":
    main()
