"""Zero-/few-shot local-LLM baseline vs the fine-tuned router, with latency and cost.

Rules fixed before the runs (LLM baseline). Confidentiality: LLM calls go ONLY
through llm_audit's loopback-guarded client (local Ollama); results/ holds ids, labels, model
outputs and numbers, while prompts/responses with dataset text go to outputs/ (gitignored).

Stages: track_a (74 test rows, 12 labels + unknown), track_b (the open-set evaluation headline
holdout set,
10 known labels + unknown), latency_ft (fine-tuned model batch-1/batch-32 latency, no metrics),
report (summary.json, cost estimates, figure), rescore_track_a (recompute track_a metrics from the
saved LLM predictions against the current final model; no LLM/GPU, not part of `all`).
Test-row LLM batches are logged as
`baseline_inference` and the fine-tuned latency pass as `latency_inference` in the shared test log;
neither touches the final model's single `evaluation` entry.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.data import get_frame
from intent_router.evaluate import (
    _f1_from_confusion,
    bootstrap_metrics_ci,
    confusion_matrix,
    git_sha,
    macro_f1_gold_classes,
    mcnemar_exact,
)
from intent_router.llm_audit import (
    LABEL_DESCRIPTIONS,
    OllamaJudge,
    _http_json,
    _ollama_whitelisted,
    validate_loopback_url,
)

STAGES = ("track_a", "track_b", "latency_ft", "report", "rescore_track_a", "all")
UNKNOWN = "unknown"
UNKNOWN_DESCRIPTION = "fits none of the listed intents"
PARSE_FAIL = "<parse_failure>"
CALL_TYPE_LLM = "baseline_inference"
CALL_TYPE_LATENCY = "latency_inference"
SMOKE_LABEL = "SMOKE: val rows, NOT a test score"
PRED_COLUMNS = ["id", "model", "mode", "gold", "pred", "parse_ok", "latency_s"]
RETRY_REMINDER = (
    "Your answer was not in the required format. Reply with exactly one label name from the "
    f'list, or "{UNKNOWN}", and nothing else.'
)


# ---------------------------------------------------------------------- parsing / prompt
def parse_label(raw: str, labels: Sequence[str]) -> str | None:
    """Strictly parse: exactly a label or 'unknown' (case-insensitive, trimmed), else None."""
    by_lower = {label.lower(): label for label in [*labels, UNKNOWN]}
    return by_lower.get(raw.strip().lower())


def sample_fewshot(
    train: pd.DataFrame, seed: int, n_shots: int, exclude_classes: Sequence[str] = ()
) -> list[dict[str, str]]:
    """n_shots TRAIN-split examples from distinct classes (seeded, deterministic).

    Classes in exclude_classes (the Track B held-out classes) are never drawn, i.e. the draw is
    redone over the remaining classes. Returns [{id, text, label}] in a seeded presentation order.
    """
    if (train["split"] != "train").any():
        raise ValueError("few-shot examples must come from the train split only")
    rng = random.Random(seed)  # noqa: S311 - seeded sampling, not security
    classes = sorted(set(train["label"]) - set(exclude_classes))
    if len(classes) < n_shots:
        raise ValueError(f"only {len(classes)} classes available for {n_shots} shots")
    picked = rng.sample(classes, n_shots)
    shots = []
    for c in picked:
        i = rng.choice(sorted(train.loc[train["label"] == c, "id"]))
        text = str(train.loc[train["id"] == i, "text"].iloc[0])
        shots.append({"id": str(i), "text": text, "label": c})
    return shots


def build_prompt(
    text: str, labels: Sequence[str], shots: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """System message (4b label descriptions + unknown), few-shot turns, then the query."""
    lines = "\n".join(f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in labels)
    system = (
        "You route chat messages from a supply-chain / logistics assistant to exactly one "
        "intent label. The available labels are:\n"
        f"{lines}\n- {UNKNOWN}: {UNKNOWN_DESCRIPTION}\n\n"
        "Answer with exactly one label name and nothing else."
    )
    msgs = [{"role": "system", "content": system}]
    for s in shots:
        msgs.append({"role": "user", "content": f"Message:\n{s['text']}\n\nAnswer:"})
        msgs.append({"role": "assistant", "content": s["label"]})
    msgs.append({"role": "user", "content": f"Message:\n{text}\n\nAnswer:"})
    return msgs


def classify(
    judge: OllamaJudge, text: str, labels: Sequence[str], shots: Sequence[dict[str, str]]
) -> dict[str, Any]:
    """One message: ask, retry once with a reminder if unparseable. Wall time covers both calls."""
    msgs = build_prompt(text, labels, shots)
    t0 = time.perf_counter()
    raw1 = judge.chat(msgs)
    pred = parse_label(raw1, labels)
    raws = [raw1]
    if pred is None:
        raw2 = judge.chat(
            [
                *msgs,
                {"role": "assistant", "content": raw1},
                {"role": "user", "content": RETRY_REMINDER},
            ]
        )
        raws.append(raw2)
        pred = parse_label(raw2, labels)
    return {
        "pred": pred if pred is not None else PARSE_FAIL,
        "parse_ok": pred is not None,
        "latency_s": time.perf_counter() - t0,
        "raw": raws,
    }


# ---------------------------------------------------------------------- metrics
def _to_index(preds: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Label -> class index; 'unknown' and parse failures -> len(labels) (an extra wrong class)."""
    l2i = {lab: i for i, lab in enumerate(labels)}
    return np.array([l2i.get(p, len(labels)) for p in preds], dtype=np.int64)


def paired_delta_macro_f1(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, k: int, n_res: int, seed: int
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(A) - macro-F1(B) over the k real classes (label set fixed).

    Predictions may use index k ('unknown'/failure, an extra class). The same resampled rows
    score both, percentile 95% CI. Unlike evaluate.paired_bootstrap_delta_f1 the macro mean is
    over exactly range(k), so the extra class does not dilute the average.
    """
    n, kk = len(y), k + 1
    idx = np.random.default_rng(seed).integers(0, n, size=(n_res, n))
    fa, fb = y * kk + pred_a, y * kk + pred_b
    labels = np.arange(k)
    d = np.empty(n_res)
    for b in range(n_res):
        ca = np.bincount(fa[idx[b]], minlength=kk * kk).reshape(kk, kk)
        cb = np.bincount(fb[idx[b]], minlength=kk * kk).reshape(kk, kk)
        d[b] = _f1_from_confusion(ca, False, labels) - _f1_from_confusion(cb, False, labels)
    lo, hi = np.quantile(d, [0.025, 0.975])
    point = _f1_from_confusion(confusion_matrix(y, pred_a, kk), False, labels) - (
        _f1_from_confusion(confusion_matrix(y, pred_b, kk), False, labels)
    )
    return {
        "delta": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "frac_resamples_le_zero": float(np.mean(d <= 0)),
    }


def metrics_track_a(
    gold: np.ndarray,
    preds: Sequence[str],
    parse_ok: Sequence[bool],
    labels: Sequence[str],
    n_res: int,
    seed: int,
    final_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Track A metrics; unknown / parse failure are wrong. macro-F1 over labels=range(12)."""
    k = len(labels)
    p = _to_index(preds, labels)
    ci = bootstrap_metrics_ci(gold, p, k + 1, n_res, seed, fixed_labels=np.arange(k))
    out: dict[str, Any] = {
        "n": len(gold),
        "macro_f1": ci["macro_f1"],
        "accuracy": ci["accuracy"],
        "parse_failure_rate": float(1.0 - np.mean(parse_ok)),
        "unknown_rate": float(np.mean([x == UNKNOWN for x in preds])),
        "bootstrap": {"n_resamples": n_res, "seed": seed, "method": "percentile, row resampling"},
    }
    if final_pred is not None:
        out["delta_macro_f1_llm_minus_final"] = paired_delta_macro_f1(
            gold, p, final_pred, k, n_res, seed
        )
        out["mcnemar_exact_on_correctness_a_is_llm"] = mcnemar_exact(p == gold, final_pred == gold)
    return out


def metrics_track_b(
    gold: Sequence[str], is_unknown: Sequence[bool], preds: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Strict rejection recall, retention and known-side metrics among accepted rows.

    Rejection = output 'unknown'; a parse failure is NOT a rejection. Accepted known rows keep
    their parse failures (counted wrong). macro_f1_present = mean over the gold classes of the
    accepted known rows, as in the shipped method's `known_accepted_macro_f1_present`.
    """
    unk = np.asarray(is_unknown, dtype=bool)
    pr = np.asarray(preds, dtype=object)
    rejected = pr == UNKNOWN
    n_known, n_unknown = int((~unk).sum()), int(unk.sum())
    acc_mask = ~unk & ~rejected
    out: dict[str, Any] = {
        "n_known": n_known,
        "n_unknown": n_unknown,
        "strict_rejection_recall": float(rejected[unk].mean()) if n_unknown else None,
        "retention_known": float(acc_mask.sum() / n_known) if n_known else None,
        "n_known_accepted": int(acc_mask.sum()),
        "parse_failures_known": int(((~unk) & (pr == PARSE_FAIL)).sum()),
        "parse_failures_unknown": int((unk & (pr == PARSE_FAIL)).sum()),
        "known_accepted_macro_f1_present": None,
        "known_accepted_accuracy": None,
    }
    if acc_mask.any():
        y = _to_index(list(np.asarray(gold, dtype=object)[acc_mask]), labels)
        p = _to_index(list(pr[acc_mask]), labels)
        out["known_accepted_macro_f1_present"] = macro_f1_gold_classes(y, p, len(labels) + 1)
        out["known_accepted_accuracy"] = float(np.mean(y == p))
    return out


def latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / mean seconds and sequential throughput (messages per second)."""
    a = np.asarray(latencies, dtype=float)
    return {
        "n": len(a),
        "p50_s": float(np.percentile(a, 50)),
        "p95_s": float(np.percentile(a, 95)),
        "mean_s": float(a.mean()),
        "throughput_msg_per_s": float(len(a) / a.sum()),
    }


def cost_per_1k(latency_s: float, usd_per_hr: float) -> float:
    """USD per 1000 messages = latency_s x 1000 / 3600 x $/hr (an estimate)."""
    return latency_s * 1000.0 / 3600.0 * usd_per_hr


# ---------------------------------------------------------------------- plumbing
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    """Write indented JSON (numpy scalars converted)."""

    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def make_dirs(cfg: dict[str, Any], smoke: bool) -> dict[str, Path]:
    """Real paths from the config, or a self-contained sandbox under smoke_dir."""
    if smoke:
        root = Path(cfg["smoke_dir"])
        return {
            "results": root / "results",
            "outputs": root / "outputs",
            "log": root / "test_eval_log.jsonl",
            "figure": root / "llm_baseline.png",
        }
    return {
        "results": Path(cfg["results_dir"]),
        "outputs": Path(cfg["outputs_dir"]),
        "log": Path(cfg["test_eval_log"]),
        "figure": Path(cfg["figure_path"]),
    }


def append_log(path: Path, entry: dict[str, Any]) -> None:
    """Append one jsonl line to the shared test log (never rewrites existing lines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"timestamp": _now(), "git_sha": git_sha(), **entry}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def model_digests(cfg: dict[str, Any]) -> dict[str, str]:
    """Ollama digest per configured model tag (loopback /api/tags); fails if a tag is missing."""
    base = validate_loopback_url(cfg["ollama"]["base_url"])
    tags = {m["name"]: m["digest"] for m in _http_json(f"{base}/api/tags", None, 30)["models"]}
    missing = [m["tag"] for m in cfg["models"] if m["tag"] not in tags]
    if missing:
        raise RuntimeError(f"models not pulled: {missing} (ollama pull <tag>)")
    return {m["tag"]: tags[m["tag"]] for m in cfg["models"]}


def _lock_info(lock: dict[str, Any], seen: list[dict[str, Any]], before: Any, after: Any) -> Any:
    return {
        "waited_s": lock["waited_s"],
        "ollama_processes_whitelisted": [
            {"pid": p.get("pid"), "process_name": p.get("process_name")} for p in seen
        ],
        "snapshot_before": before,
        "snapshot_after": after,
    }


def run_configs(
    cfg: dict[str, Any],
    rows: pd.DataFrame,
    labels: list[str],
    shots_by_mode: dict[str, Any],
    stage: str,
    d: dict[str, Path],
    smoke: bool,
    n_test_rows: int,
    extra_log: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every (model, mode) over rows (sequential, id order). Returns predictions + timing.

    rows: id, text, gold (label), plus any extra columns (kept in the predictions). Raw replies
    with dataset text go to outputs/ only. Each (model, mode) batch appends ONE
    `baseline_inference` line to the test log first, if it contains test rows.
    """
    digests = model_digests(cfg)
    d["outputs"].mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    timing: dict[str, Any] = {}
    for m in cfg["models"]:
        judge = OllamaJudge(cfg["ollama"]["base_url"], m["tag"], m["think"], cfg["ollama"])
        try:
            for mode in cfg["modes"]:
                shots = shots_by_mode[mode]
                key = f"{m['name']}|{mode}"
                if n_test_rows:
                    append_log(
                        d["log"],
                        {
                            "call_type": CALL_TYPE_LLM,
                            "model_tag": m["tag"],
                            "model_digest": digests[m["tag"]],
                            "mode": mode,
                            "n_rows": n_test_rows,
                            "stage": stage,
                            "smoke": smoke,
                            **extra_log,
                        },
                    )
                t0 = time.perf_counter()
                judge.chat([{"role": "user", "content": "Reply with: ok"}])  # load + warm
                load_s = time.perf_counter() - t0
                recs: list[dict[str, Any]] = []
                with (d["outputs"] / f"raw_{stage}_{m['name']}_{mode}.jsonl").open(
                    "w", encoding="utf-8"
                ) as fh:
                    for r in rows.itertuples():
                        res = classify(judge, r.text, labels, shots)
                        recs.append(
                            {
                                "id": r.id,
                                "model": m["name"],
                                "mode": mode,
                                "gold": r.gold,
                                "pred": res["pred"],
                                "parse_ok": res["parse_ok"],
                                "latency_s": round(res["latency_s"], 4),
                            }
                        )
                        fh.write(
                            json.dumps(
                                {
                                    "id": r.id,
                                    "text": r.text,
                                    "raw": res["raw"],
                                    "shot_ids": [s["id"] for s in shots],
                                }
                            )
                            + "\n"
                        )
                df = pd.DataFrame(recs)
                for c in rows.columns:
                    if c not in ("text", "gold", "id"):
                        df[c] = rows[c].to_numpy()
                parts.append(df)
                timing[key] = {
                    "load_s": round(load_s, 2),
                    "model_tag": m["tag"],
                    "model_digest": digests[m["tag"]],
                    **latency_stats(df["latency_s"].to_numpy()),
                }
                print(
                    f"[{stage}] {key}: parse_fail={1 - df['parse_ok'].mean():.3f} "
                    f"p50={timing[key]['p50_s']:.2f}s",
                    flush=True,
                )
        finally:
            judge.unload()
    return pd.concat(parts, ignore_index=True), timing


def _fewshot_by_mode(cfg: dict[str, Any], exclude: Sequence[str]) -> dict[str, Any]:
    train = get_frame(["train"], cfg["data_path"], cfg["splits_path"])
    five = sample_fewshot(train, cfg["seed"], cfg["n_shots"], exclude)
    return {"zero": [], "five": five}


def _gpu_llm(cfg: dict[str, Any], what: str) -> Any:
    """Context managers: whitelist ollama processes, then hold exclusive GPU access."""
    from contextlib import ExitStack

    stack = ExitStack()
    seen = stack.enter_context(_ollama_whitelisted())
    lock = stack.enter_context(gpu_lock.gpu_exclusive(int(cfg["expected_duration_s"]), what))
    return stack, seen, lock


# ---------------------------------------------------------------------- track A
def stage_track_a(cfg: dict[str, Any], smoke: bool) -> None:
    """All 74 test rows (ids from the final model's predictions), 12 labels + unknown."""
    d = make_dirs(cfg, smoke)
    df = data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    text_of = dict(zip(df["id"], df["text"], strict=True))
    label_of = dict(zip(df["id"], df["label"], strict=True))
    if smoke:  # VAL rows only; never touches test rows or the real log
        val = get_frame(["val"], cfg["data_path"], cfg["splits_path"])
        ids = sorted(val.sample(n=cfg["smoke_rows"], random_state=cfg["seed"])["id"])
        final = None
    else:
        final = pd.read_csv(cfg["test_predictions_path"]).sort_values("id").reset_index(drop=True)
        ids = final["id"].tolist()
    rows = pd.DataFrame(
        {"id": ids, "text": [text_of[i] for i in ids], "gold": [label_of[i] for i in ids]}
    )
    if final is not None and rows["gold"].tolist() != final["gold_label"].tolist():
        raise AssertionError("gold labels differ from results/final/test_predictions.csv")
    stack, seen, lock = _gpu_llm(cfg, "python -m intent_router.llm_baseline track_a")
    with stack:
        before = gpu_lock.gpu_snapshot()
        preds, timing = run_configs(
            cfg,
            rows,
            labels,
            _fewshot_by_mode(cfg, ()),
            "track_a",
            d,
            smoke,
            0 if smoke else len(rows),
            {},
        )
        after = gpu_lock.gpu_snapshot()
    l2i = {lab: i for i, lab in enumerate(labels)}
    n_res = 200 if smoke else int(cfg["bootstrap"]["n_resamples"])
    seed = int(cfg["bootstrap"]["seed"])
    metrics: dict[str, Any] = {}
    for (model, mode), g in preds.groupby(["model", "mode"], sort=False):
        g = g.sort_values("id")
        y = g["gold"].map(l2i).to_numpy()
        fp = final["pred"].to_numpy() if final is not None else None
        metrics[f"{model}|{mode}"] = metrics_track_a(
            y, g["pred"].tolist(), g["parse_ok"].tolist(), labels, n_res, seed, fp
        )
    d["results"].mkdir(parents=True, exist_ok=True)
    preds.to_csv(d["results"] / "predictions_track_a.csv", index=False)
    write_json(
        d["results"] / "track_a.json",
        {
            "label": SMOKE_LABEL if smoke else "unbiased (baseline_inference on the test split)",
            "smoke": smoke,
            "git_sha": git_sha(),
            "labels_offered": [*labels, UNKNOWN],
            "fewshot_ids": {m: [s["id"] for s in v] for m, v in _fewshot_by_mode(cfg, ()).items()},
            "metrics": metrics,
            "gpu_lock": _lock_info(lock, seen, before, after),
        },
    )
    write_json(
        d["results"] / "llm_latency.json",
        {
            "note": "sequential wall seconds per message (retry included); model load excluded "
            "(load_s = first tiny chat after unload, reported separately)",
            "smoke": smoke,
            "configs": timing,
        },
    )


def rescore_metrics(
    preds: pd.DataFrame, final: pd.DataFrame, labels: Sequence[str], n_res: int, seed: int
) -> dict[str, Any]:
    """Track A metrics per (model, mode) from saved LLM predictions vs a final model's predictions.

    Same computation as stage_track_a (id order, metrics_track_a with the final model's `pred`
    as the paired reference); raises if any config's gold labels differ from final['gold_label'].
    """
    final = final.sort_values("id").reset_index(drop=True)
    l2i = {lab: i for i, lab in enumerate(labels)}
    metrics: dict[str, Any] = {}
    for (model, mode), g in preds.groupby(["model", "mode"], sort=False):
        g = g.sort_values("id")
        if g["id"].tolist() != final["id"].tolist() or (
            g["gold"].tolist() != final["gold_label"].tolist()
        ):
            raise AssertionError(f"ids/gold of {model}|{mode} differ from the final predictions")
        metrics[f"{model}|{mode}"] = metrics_track_a(
            g["gold"].map(l2i).to_numpy(),
            g["pred"].tolist(),
            g["parse_ok"].tolist(),
            labels,
            n_res,
            seed,
            final["pred"].to_numpy(),
        )
    return metrics


def stage_rescore_track_a(cfg: dict[str, Any]) -> None:
    """Recompute track_a.json `metrics` from saved LLM predictions against the current final model.

    No Ollama, no GPU, no test-log entry: it only re-reads results/ files. Other keys of
    track_a.json are kept; the final model fingerprint and git sha of the rescoring are added.
    """
    data_mod.load_data(cfg["data_path"])  # populates data_mod.LABELS
    res = Path(cfg["results_dir"])
    preds = pd.read_csv(res / "predictions_track_a.csv")
    final = pd.read_csv(cfg["test_predictions_path"])
    metrics = rescore_metrics(
        preds,
        final,
        list(data_mod.LABELS),
        int(cfg["bootstrap"]["n_resamples"]),
        int(cfg["bootstrap"]["seed"]),
    )
    version = json.loads(
        (Path(cfg["test_predictions_path"]).parent / "model_version.json").read_text()
    )
    ta = json.loads((res / "track_a.json").read_text())
    ta["metrics"] = metrics
    ta["rescored_against_final_fingerprint"] = version["model_fingerprint"]
    ta["rescored_git_sha"] = git_sha()
    write_json(res / "track_a.json", ta)
    print(f"[rescore_track_a] rewrote {res / 'track_a.json'}")


# ---------------------------------------------------------------------- track B
def headline_rows(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    """The open-set evaluation headline holdout rows (set == 'eval'), ids and gold from
    headline_s42.csv.

    Smoke: a seeded 2 known + 1 unknown sample of VAL rows instead. Columns: id, text, gold,
    is_unknown, split.
    """
    df = data_mod.load_data(cfg["data_path"])
    text_of = dict(zip(df["id"], df["text"], strict=True))
    hs = pd.read_csv(cfg["headline_scores_path"])
    if smoke:
        val = hs[hs["split"] == "val"]
        k = val[~val["is_unknown"]].sample(n=cfg["smoke_rows"] - 1, random_state=cfg["seed"])
        u = val[val["is_unknown"]].sample(n=1, random_state=cfg["seed"])
        sel = pd.concat([k, u])
    else:
        sel = hs[hs["set"] == "eval"]
    sel = sel.sort_values("id")
    out = pd.DataFrame(
        {
            "id": sel["id"].tolist(),
            "text": [text_of[i] for i in sel["id"]],
            "gold": sel["gold"].tolist(),
            "is_unknown": sel["is_unknown"].tolist(),
            "split": sel["split"].tolist(),
        }
    )
    return out.reset_index(drop=True)


def stage_track_b(cfg: dict[str, Any], smoke: bool) -> None:
    """Headline holdout set, 10 known labels + unknown; rejection = output 'unknown'."""
    d = make_dirs(cfg, smoke)
    data_mod.load_data(cfg["data_path"])
    holdout = list(cfg["holdout"])
    known = [lab for lab in data_mod.LABELS if lab not in holdout]
    rows = headline_rows(cfg, smoke)
    if not smoke:  # the identical-rows guarantee, checked against the open-set evaluation numbers
        hj = json.loads(Path(cfg["headline_json_path"]).read_text())
        if (int((~rows["is_unknown"]).sum()), int(rows["is_unknown"].sum())) != (
            hj["n_eval_known"],
            hj["n_eval_unknown"],
        ):
            raise AssertionError("headline rows differ from headline.json counts")
        if sorted(rows.loc[rows["is_unknown"], "gold"].unique()) != sorted(holdout):
            raise AssertionError("unknown rows are not exactly the held-out classes")
        if sorted(hj["holdout"]) != sorted(holdout):
            raise AssertionError("config holdout differs from headline.json")
    n_test = int((rows["split"] == "test").sum())
    shots = _fewshot_by_mode(cfg, holdout)
    stack, seen, lock = _gpu_llm(cfg, "python -m intent_router.llm_baseline track_b")
    with stack:
        before = gpu_lock.gpu_snapshot()
        preds, timing = run_configs(
            cfg,
            rows,
            known,
            shots,
            "track_b",
            d,
            smoke,
            0 if smoke else n_test,
            {
                "n_rows_non_test": len(rows) - n_test,
                "note": "unknown rows from train/val are "
                "not test rows; n_rows counts test rows only",
            },
        )
        after = gpu_lock.gpu_snapshot()
    metrics = {}
    for (model, mode), g in preds.groupby(["model", "mode"], sort=False):
        metrics[f"{model}|{mode}"] = metrics_track_b(
            g["gold"].tolist(), g["is_unknown"].tolist(), g["pred"].tolist(), data_mod.LABELS
        )
    d["results"].mkdir(parents=True, exist_ok=True)
    preds.to_csv(d["results"] / "predictions_track_b.csv", index=False)
    write_json(
        d["results"] / "track_b.json",
        {
            "label": SMOKE_LABEL
            if smoke
            else "headline holdout set (open-set evaluation), LLM baseline",
            "smoke": smoke,
            "git_sha": git_sha(),
            "holdout": holdout,
            "labels_offered": [*known, UNKNOWN],
            "n_known": int((~rows["is_unknown"]).sum()),
            "n_unknown": int(rows["is_unknown"].sum()),
            "n_test_rows_logged": n_test,
            "n_non_test_rows": len(rows) - n_test,
            "fewshot_ids": {m: [s["id"] for s in v] for m, v in shots.items()},
            "metrics": metrics,
            "timing": timing,
            "gpu_lock": _lock_info(lock, seen, before, after),
        },
    )


# ---------------------------------------------------------------------- fine-tuned latency
def _time_batch1(
    model: Any, tok: Any, texts: list[str], device: str, max_len: int, warmup: int
) -> list[float]:
    """Per-message seconds (tokenise + forward + argmax), batch 1, after `warmup` untimed calls."""
    import torch

    def one(t: str) -> None:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.inference_mode():
            int(model(**enc).logits.argmax(dim=-1).item())  # .item() syncs the GPU

    for i in range(warmup):
        one(texts[i % len(texts)])
    out = []
    for t in texts:
        t0 = time.perf_counter()
        one(t)
        out.append(time.perf_counter() - t0)
    return out


def _time_batched(
    model: Any, tok: Any, texts: list[str], device: str, max_len: int, bs: int, repeats: int
) -> float:
    """Median over repeats of messages/second at batch size bs (after one warm-up pass)."""
    import torch

    def run() -> None:
        for i in range(0, len(texts), bs):
            enc = tok(
                texts[i : i + bs],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                model(**enc).logits.argmax(dim=-1).cpu()

    run()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        rates.append(len(texts) / (time.perf_counter() - t0))
    return float(np.median(rates))


def stage_latency_ft(cfg: dict[str, Any], smoke: bool) -> None:
    """Fine-tuned model latency on test texts (no metrics, no labels used); one log line."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from intent_router.models import state_dict_sha256

    d = make_dirs(cfg, smoke)
    df = data_mod.load_data(cfg["data_path"])
    text_of = dict(zip(df["id"], df["text"], strict=True))
    if smoke:
        val = get_frame(["val"], cfg["data_path"], cfg["splits_path"])
        ids = sorted(val.sample(n=cfg["smoke_ft_texts"], random_state=cfg["seed"])["id"])
    else:
        ids = sorted(pd.read_csv(cfg["test_predictions_path"])["id"])
    mdir = Path(cfg["final_model_dir"])
    mcfg = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    prefix, max_len = mcfg.get("query_prefix", ""), int(cfg["latency"]["max_len"])
    texts = [prefix + text_of[i] for i in ids]
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(
        mdir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    fingerprint = state_dict_sha256(model)
    lc = cfg["latency"]
    default_threads = torch.get_num_threads()
    res: dict[str, Any] = {
        "smoke": smoke,
        "n_texts": len(texts),
        "model_fingerprint": fingerprint,
        "precision": "fp32, eager attention, eval mode",
        "max_len": max_len,
        "query_prefix": prefix,
        "warmup": lc["ft_warmup"],
        "torch": torch.__version__,
        "metrics_computed": False,
    }
    with gpu_lock.gpu_exclusive(600, "python -m intent_router.llm_baseline latency_ft"):
        append_log(
            d["log"],
            {
                "call_type": CALL_TYPE_LATENCY,
                "model_fingerprint": fingerprint,
                "n_rows": len(texts),
                "stage": "latency_ft",
                "smoke": smoke,
            },
        )
        if torch.cuda.is_available():
            res["gpu_name"] = torch.cuda.get_device_name(0)
            model.to("cuda")
            res["gpu_batch1"] = latency_stats(
                _time_batch1(model, tok, texts, "cuda", max_len, lc["ft_warmup"])
            )
            res["gpu_batch32_throughput_msg_per_s"] = _time_batched(
                model, tok, texts, "cuda", max_len, lc["ft_batch_size"], lc["ft_batch_repeats"]
            )
            model.to("cpu")
        res["cpu_threads_default"] = default_threads
        for name, n in (("cpu_batch1_1thread", 1), ("cpu_batch1_default_threads", default_threads)):
            torch.set_num_threads(n)
            res[name] = {
                "threads": n,
                **latency_stats(_time_batch1(model, tok, texts, "cpu", max_len, lc["ft_warmup"])),
            }
        torch.set_num_threads(default_threads)
    d["results"].mkdir(parents=True, exist_ok=True)
    write_json(d["results"] / "ft_latency.json", res)
    print(f"[latency_ft] gpu b1 p50={res.get('gpu_batch1', {}).get('p50_s')}", flush=True)


# ---------------------------------------------------------------------- report
def build_costs(cfg: dict[str, Any], llm: dict[str, Any], ft: dict[str, Any]) -> dict[str, Any]:
    """Cost per 1k messages for every LLM config and the fine-tuned model (all ESTIMATES)."""
    gpu = float(cfg["cost"]["gpu_usd_per_hr"])
    cpu_per_vcpu = float(cfg["cost"]["cpu_usd_per_hr"])  # price of ONE vCPU
    gpu_basis = f"1 T4-class GPU at ${gpu:.2f}/hr"
    rows: dict[str, Any] = {}
    for key, v in llm["configs"].items():
        rows[f"llm|{key}"] = {
            "latency_p50_s": v["p50_s"],
            "usd_per_hr": gpu,
            "usd_per_hr_basis": gpu_basis,
            "hardware": "T4-class GPU (stand-in)",
            "usd_per_1k_messages": cost_per_1k(v["p50_s"], gpu),
            "optimistic": True,
            "usd_per_1k_messages_p95": cost_per_1k(v["p95_s"], gpu),
        }
    if "gpu_batch1" in ft:
        rows["ft|gpu_batch1"] = {
            "latency_p50_s": ft["gpu_batch1"]["p50_s"],
            "usd_per_hr": gpu,
            "usd_per_hr_basis": gpu_basis,
            "hardware": "T4-class GPU (3070-measured stand-in)",
            "optimistic": True,
            "usd_per_1k_messages": cost_per_1k(ft["gpu_batch1"]["p50_s"], gpu),
        }
        rows["ft|gpu_batch32"] = {
            "latency_p50_s": 1.0 / ft["gpu_batch32_throughput_msg_per_s"],
            "usd_per_hr": gpu,
            "usd_per_hr_basis": gpu_basis,
            "hardware": "T4-class GPU (3070-measured stand-in)",
            "optimistic": True,
            "usd_per_1k_messages": cost_per_1k(1.0 / ft["gpu_batch32_throughput_msg_per_s"], gpu),
        }
    for k in ("cpu_batch1_1thread", "cpu_batch1_default_threads"):
        if k in ft:
            n_vcpu = int(ft[k]["threads"])  # a run on N threads occupies N vCPUs
            cpu = n_vcpu * cpu_per_vcpu
            rows[f"ft|{k}"] = {
                "latency_p50_s": ft[k]["p50_s"],
                "usd_per_hr": cpu,
                "usd_per_hr_basis": f"{n_vcpu} vCPU x ${cpu_per_vcpu:.2f}/hr",
                "n_vcpu": n_vcpu,
                "hardware": f"{n_vcpu} vCPU(s) = threads used",
                "usd_per_1k_messages": cost_per_1k(ft[k]["p50_s"], cpu),
            }
    return {
        "label": "ESTIMATES, not measured cloud cost",
        "formula": "usd_per_1k_messages = latency_s x 1000 / 3600 x usd_per_hr",
        "assumptions": {
            "gpu_usd_per_hr": gpu,
            "cpu_usd_per_vcpu_per_hr": cpu_per_vcpu,
            "gpu_note": "GPU rows are priced as a T4-class stand-in from RTX 3070 timings "
            "(optimistic, a lower bound)",
            "cpu_note": "CPU rows count vCPUs = threads used, each priced at "
            "cpu_usd_per_vcpu_per_hr",
            "llm_note": str(cfg["cost"]["llm_note"]).strip(),
            "latency_basis": "p50 per message, batch 1 (ft batch32: 1/throughput)",
        },
        "rows": rows,
    }


def make_figure(summary: dict[str, Any], path: Path) -> None:
    """Left: Track A macro-F1 (95% CI). Right: Track B rejection recall vs retention."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = summary["track_a"]
    names = list(a["llm"]) + ["final model"]
    f1 = [a["llm"][k]["macro_f1"] for k in a["llm"]] + [a["final_model"]["macro_f1"]]
    pts = [x["point"] for x in f1]
    err = np.array([[x["point"] - x["lo"], x["hi"] - x["point"]] for x in f1]).T
    b = summary["track_b"]
    bn = list(b["llm"]) + ["shipped (maha_ft, 95%)"]
    rec = [b["llm"][k]["strict_rejection_recall"] for k in b["llm"]]
    ret = [b["llm"][k]["retention_known"] for k in b["llm"]]
    rec.append(b["shipped_maha_ft"]["strict_rejection_recall"]["mean"])
    ret.append(b["shipped_maha_ft"]["retention_known"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(range(len(names)), pts, yerr=err, capsize=3, color="#4c72b0")
    axes[0].set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("macro-F1 over 12 classes (test, 95% CI)")
    axes[0].set_title("Track A: LLM baselines vs fine-tuned", fontsize=10)
    x = np.arange(len(bn))
    axes[1].bar(x - 0.2, rec, 0.4, label="strict rejection recall", color="#c44e52")
    axes[1].bar(x + 0.2, ret, 0.4, label="retention (known)", color="#55a868")
    axes[1].set_xticks(x, bn, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Track B: headline holdout", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _wandb_summary(cfg: dict[str, Any], summary: dict[str, Any], figure: Path) -> str | None:
    """W&B group stretch-llm-baseline: summary numbers + figure only. Never raises."""
    try:
        import wandb

        run = wandb.init(
            project=cfg["wandb"]["project"],
            group=cfg["wandb"]["group"],
            name="llm-baseline-summary",
            job_type="llm_baseline",
            reinit=True,
        )
        flat: dict[str, Any] = {}
        for k, v in summary["track_a"]["llm"].items():
            flat[f"track_a/{k}/macro_f1"] = v["macro_f1"]["point"]
            flat[f"track_a/{k}/accuracy"] = v["accuracy"]["point"]
        for k, v in summary["track_b"]["llm"].items():
            flat[f"track_b/{k}/strict_rejection_recall"] = v["strict_rejection_recall"]
            flat[f"track_b/{k}/retention_known"] = v["retention_known"]
        run.summary.update(flat)
        run.log({"fig/llm_baseline": wandb.Image(str(figure))})
        url = getattr(run, "url", None)
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001 - tracking must never kill the pipeline
        print(f"[wandb] logging failed: {exc!r}")
        return None


def stage_report(cfg: dict[str, Any], smoke: bool) -> None:
    """Merge the stage outputs into summary.json (+ costs) and the figure."""
    d = make_dirs(cfg, smoke)
    res = d["results"]
    ta = json.loads((res / "track_a.json").read_text())
    tb = json.loads((res / "track_b.json").read_text())
    llm_lat = json.loads((res / "llm_latency.json").read_text())
    ft = json.loads((res / "ft_latency.json").read_text())
    shipped = json.loads(Path(cfg["headline_json_path"]).read_text())["methods"]["maha_ft"]
    keys = ("strict_rejection_recall", "retention_known", "known_accepted_macro_f1_present")
    if smoke:  # no final-model reference on val rows
        final_m = {"macro_f1": {"point": 0.0, "lo": 0.0, "hi": 0.0}}
    else:
        final = pd.read_csv(cfg["test_predictions_path"])
        final_m = bootstrap_metrics_ci(
            final["gold"].to_numpy(),
            final["pred"].to_numpy(),
            cfg["n_classes"],
            int(cfg["bootstrap"]["n_resamples"]),
            int(cfg["bootstrap"]["seed"]),
        )
    costs = build_costs(cfg, llm_lat, ft)
    summary: dict[str, Any] = {
        "smoke": smoke,
        "git_sha": git_sha(),
        "label": SMOKE_LABEL if smoke else "Zero-/few-shot LLM baseline (pre-registered)",
        "models": {k: v["model_digest"] for k, v in llm_lat["configs"].items()},
        "track_a": {
            "n": next(iter(ta["metrics"].values()))["n"],
            "llm": ta["metrics"],
            "final_model": final_m,
            "labels_offered": ta["labels_offered"],
            "fewshot_ids": ta["fewshot_ids"],
        },
        "track_b": {
            "llm": tb["metrics"],
            "n_known": tb["n_known"],
            "n_unknown": tb["n_unknown"],
            "n_test_rows_logged": tb["n_test_rows_logged"],
            "n_non_test_rows": tb["n_non_test_rows"],
            "shipped_maha_ft": {
                k: {"mean": shipped[k]["mean"], "std": shipped[k]["std"]} for k in keys
            },
            "shipped_note": "mean/std over model seeds 42-44 (results/trackb/headline.json)",
            "fewshot_ids": tb["fewshot_ids"],
        },
        "latency": {"llm": llm_lat, "finetuned": ft},
        "cost": costs,
    }
    make_figure(summary, d["figure"])
    summary["figure"] = str(d["figure"])
    use_wb = bool(cfg["wandb"]["enabled"]) and not smoke
    summary["wandb_url"] = _wandb_summary(cfg, summary, d["figure"]) if use_wb else None
    write_json(res / "summary.json", summary)
    print(f"[report] wrote {res / 'summary.json'}")


# ---------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    """CLI: `--stage {track_a,track_b,latency_ft,report,rescore_track_a,all} [--smoke]`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/llm_baseline.yaml")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--smoke", action="store_true", help="3 VAL rows per config; smoke_dir only")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_loopback_url(cfg["ollama"]["base_url"])
    if not cfg["wandb"]["enabled"] or args.smoke:
        os.environ["WANDB_MODE"] = "disabled"
    if args.stage in ("track_a", "all"):
        stage_track_a(cfg, args.smoke)
    if args.stage in ("track_b", "all"):
        stage_track_b(cfg, args.smoke)
    if args.stage in ("latency_ft", "all"):
        stage_latency_ft(cfg, args.smoke)
    if args.stage in ("report", "all"):
        stage_report(cfg, args.smoke)
    if args.stage == "rescore_track_a":  # not part of `all`: re-reads saved results only
        stage_rescore_track_a(cfg)


if __name__ == "__main__":
    main()
