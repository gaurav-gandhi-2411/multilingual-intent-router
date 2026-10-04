from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from intent_router import data as data_mod
from intent_router import evaluate as evaluate_mod
from intent_router.evaluate import (
    bootstrap_metrics_ci,
    family_mask,
    selective_at_threshold,
    softmax,
)

INDICATIVE_N = 20  # slices below this are flagged "indicative only"
SHORT_CHARS = 30  # "short text" = fewer characters than this
MIN_LANG_N = 5  # per-language slices need at least this many rows
UNDETERMINED = "undetermined"
SLICE_MACRO_F1_DEFINITION = (
    "macro-F1 of a slice = mean of per-class F1 over the classes present in the slice's GOLD "
    "labels (label set fixed once per slice and reused in every bootstrap resample; equals "
    "sklearn f1_score(labels=gold_classes, average='macro', zero_division=0)). Predictions "
    "into classes outside the slice's gold set add no class. Reported only when the slice has "
    ">= 2 gold classes, else accuracy is the primary metric. Caveat: a fixed gold class that "
    "is absent from a bootstrap resample scores 0, so percentile CIs on small slices with "
    "rare classes are biased downward and can exclude the point estimate (e.g. lang_fr)."
)
COMBINED_LABEL = (
    "mixed-source: CV OOF probability-averaged predictions (train+val) + final-model "
    "predictions (test)"
)


def load_oof_mean(path: Path, n_classes: int, expected_per_id: int) -> pd.DataFrame:
    """Average a config's OOF probabilities per id (3 fold-seeds x 3 model seeds => 9 each).

    Returns id, gold, pred, conf (max averaged prob), prob_0..prob_{K-1}, sorted by id.
    Raises if any id does not have exactly expected_per_id predictions or gold disagrees.
    """
    raw = pd.read_csv(path)
    counts = raw.groupby("id").size()
    if not (counts == expected_per_id).all():
        bad = counts[counts != expected_per_id]
        raise ValueError(f"{path}: {len(bad)} ids without {expected_per_id} predictions")
    if (raw.groupby("id")["gold"].nunique() != 1).any():
        raise ValueError(f"{path}: gold label differs between predictions of the same id")
    cols = [f"prob_{i}" for i in range(n_classes)]
    mean = raw.groupby("id")[cols].mean()
    out = mean.copy()
    out.insert(0, "gold", raw.groupby("id")["gold"].first().astype(int))
    out.insert(1, "pred", mean.to_numpy().argmax(axis=1))
    out.insert(2, "conf", mean.to_numpy().max(axis=1))
    return out.reset_index().sort_values("id").reset_index(drop=True)


def slice_masks(df: pd.DataFrame, labels: list[str]) -> dict[str, np.ndarray]:
    """Boolean masks per slice. df needs gold, lang, code_mixed, is_noisy, n_chars.

    English vs non-English is definition A (primary language); 'undetermined' rows are in
    neither, reported as their own slice.
    """
    lang = df["lang"].to_numpy()
    non_en = (lang != "en") & (lang != UNDETERMINED)
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(df), dtype=bool),
        "english": lang == "en",
        "non_english_defA": non_en,
        "undetermined_lang": lang == UNDETERMINED,
        "code_mixed": df["code_mixed"].to_numpy(dtype=bool),
        "short_lt30_chars": df["n_chars"].to_numpy() < SHORT_CHARS,
        "noisy": df["is_noisy"].to_numpy(dtype=bool),
        "clean": ~df["is_noisy"].to_numpy(dtype=bool),
        "shipment_family": family_mask(labels)[df["gold"].to_numpy()],
        "non_shipment_family": ~family_mask(labels)[df["gold"].to_numpy()],
    }
    for lg, n in df["lang"].value_counts().items():
        if n >= MIN_LANG_N and lg != UNDETERMINED:
            masks[f"lang_{lg}"] = lang == lg
    return masks


def slice_metrics(sub: pd.DataFrame, n_classes: int, n_resamples: int, seed: int) -> dict[str, Any]:
    """n, macro-F1 (only if >= 2 gold classes; over the slice's fixed gold classes), accuracy."""
    n = len(sub)
    out: dict[str, Any] = {"n": n, "indicative_only": n < INDICATIVE_N}
    if n == 0:
        return out
    y, p = sub["gold"].to_numpy(), sub["pred"].to_numpy()
    n_gold = int(len(np.unique(y)))
    gold_classes = np.unique(y)  # fixed once; identical in the point estimate and every resample
    ci = bootstrap_metrics_ci(y, p, n_classes, n_resamples, seed, fixed_labels=gold_classes)
    out["n_gold_classes"] = n_gold
    out["accuracy"] = ci["accuracy"]
    out["macro_f1"] = ci["macro_f1"] if n_gold >= 2 else None
    out["primary_metric"] = "macro_f1" if n_gold >= 2 else "accuracy"
    return out


def slice_report(
    sources: dict[str, pd.DataFrame],
    eda_rows: pd.DataFrame,
    labels: list[str],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Slice metrics per source (each df: id, gold, pred) plus a combined view.

    'combined' concatenates the sources (one prediction per id) and is a MIXED-SOURCE view
    (COMBINED_LABEL: OOF CV predictions for train+val, final-model predictions for test); it is
    the only view covering all 62 non-English rows.
    """
    n_classes = len(labels)
    named = dict(sources)
    named["combined"] = pd.concat(list(sources.values()), ignore_index=True)
    report: dict[str, Any] = {}
    for src, frame in named.items():
        merged = frame.merge(eda_rows, on="id", how="left", validate="one_to_one")
        if merged["lang"].isna().any():
            raise ValueError(f"source {src}: ids missing from eda_rows.csv")
        report[src] = {
            name: slice_metrics(merged[m], n_classes, n_resamples, seed)
            for name, m in slice_masks(merged, labels).items()
        }
    return report


def shipment_family_hypothesis(df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    """Do errors concentrate in shipment_information.*? (df: id, gold, pred).

    Exact binomial test: of all errors, how many have a family gold class, against the
    family's share of rows (one-sided 'greater' pre-specified; two-sided also reported).
    Also per-gold-class error counts and where family errors go (within vs cross family).
    """
    gold, pred = df["gold"].to_numpy(), df["pred"].to_numpy()
    fam = family_mask(labels)
    err = pred != gold
    fam_gold = fam[gold]
    n_err, n_fam_err = int(err.sum()), int((err & fam_gold).sum())
    row_share = float(fam_gold.mean())
    out: dict[str, Any] = {
        "n_rows": len(df),
        "n_errors": n_err,
        "family_share_of_rows": row_share,
        "family_errors": n_fam_err,
        "family_share_of_errors": float(n_fam_err / n_err) if n_err else None,
    }
    if n_err:
        for alt in ("greater", "two-sided"):
            res = sps.binomtest(n_fam_err, n_err, row_share, alternative=alt)
            out[f"binomial_p_{alt.replace('-', '_')}"] = float(res.pvalue)
    within = int((err & fam_gold & fam[pred]).sum())
    out["family_errors_within_family"] = within
    out["family_errors_cross_family"] = n_fam_err - within
    out["share_family_errors_within_family"] = float(within / n_fam_err) if n_fam_err else None
    out["errors_from_outside_into_family"] = int((err & ~fam_gold & fam[pred]).sum())
    out["by_gold_class"] = [
        {
            "label": lab,
            "rows": int((gold == i).sum()),
            "errors": int((err & (gold == i)).sum()),
            "error_rate": float(err[gold == i].mean()) if (gold == i).any() else None,
            "share_of_errors": float((err & (gold == i)).sum() / n_err) if n_err else None,
        }
        for i, lab in enumerate(labels)
    ]
    return out


def ranked_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Misclassified rows ordered by confidence (desc), ties by id; needs id, gold, pred, conf."""
    wrong = df[df["gold"] != df["pred"]]
    return wrong.sort_values(["conf", "id"], ascending=[False, True]).reset_index(drop=True)


def top_confusions(
    df: pd.DataFrame, labels: list[str], k: int = 5, n_examples: int = 3
) -> list[dict[str, Any]]:
    """Top-k (gold -> pred) confusion pairs with counts and the most confident example ids."""
    wrong = ranked_errors(df)
    pairs = Counter(zip(wrong["gold"].tolist(), wrong["pred"].tolist(), strict=True))
    ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    out = []
    for (g, p), cnt in ranked:
        ids = wrong[(wrong["gold"] == g) & (wrong["pred"] == p)]["id"].tolist()[:n_examples]
        out.append({"gold": labels[g], "pred": labels[p], "count": int(cnt), "example_ids": ids})
    return out


def per_fold_run_hypothesis(
    raw: pd.DataFrame, labels: list[str], n_boot: int = 10_000, seed: int = 42, k: int = 5
) -> dict[str, Any]:
    """Shipment-family error concentration over ALL individual fold-run OOF predictions.

    raw: one row per (id, fold_seed, model_seed) with gold/pred. Each id appears 9 times, so the
    exact binomial test treats repeated rows as independent (anti-conservative: p-values are
    optimistic); the cluster bootstrap resamples whole ids (all of an id's predictions move
    together) and is the honest interval. Returns counts, tests, within/cross-family shares and
    the top-k (gold -> pred) confusion pairs by raw count.
    """
    base = shipment_family_hypothesis(raw, labels)
    gold, pred = raw["gold"].to_numpy(), raw["pred"].to_numpy()
    fam = family_mask(labels)
    err = pred != gold
    fam_err = err & fam[gold]
    ids, inv = np.unique(raw["id"].to_numpy(), return_inverse=True)
    n_ids = len(ids)
    per_id = {
        "n": np.bincount(inv, minlength=n_ids).astype(float),
        "fam_rows": np.bincount(inv, weights=fam[gold].astype(float), minlength=n_ids),
        "err": np.bincount(inv, weights=err.astype(float), minlength=n_ids),
        "fam_err": np.bincount(inv, weights=fam_err.astype(float), minlength=n_ids),
    }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_ids, size=(n_boot, n_ids))
    share_err = per_id["fam_err"][idx].sum(axis=1) / per_id["err"][idx].sum(axis=1)
    share_rows = per_id["fam_rows"][idx].sum(axis=1) / per_id["n"][idx].sum(axis=1)
    diff = share_err - share_rows
    pairs = Counter(zip(gold[err].tolist(), pred[err].tolist(), strict=True))
    top = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    base["n_unique_ids"] = n_ids
    base["n_predictions_per_id"] = int(per_id["n"].min())
    base["independence_caveat"] = (
        "each id contributes 9 non-independent predictions; binomial p-values treat them as "
        "independent and are optimistic. Use the cluster bootstrap (resampling ids)."
    )
    base["cluster_bootstrap"] = {
        "n_resamples": n_boot,
        "seed": seed,
        "resampled_unit": "row id (all its fold-run predictions)",
        "family_share_of_errors_ci95": [
            float(v) for v in np.nanquantile(share_err, [0.025, 0.975])
        ],
        "family_share_of_rows_ci95": [float(v) for v in np.quantile(share_rows, [0.025, 0.975])],
        "share_errors_minus_share_rows_ci95": [
            float(v) for v in np.nanquantile(diff, [0.025, 0.975])
        ],
        "frac_resamples_diff_le_zero": float(np.nanmean(diff <= 0)),
    }
    base["top_confusions_by_count"] = [
        {"gold": labels[g], "pred": labels[p], "count": int(c)} for (g, p), c in top
    ]
    return base


def _dump(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def _log_lines(path: Path) -> int:
    return len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()])


def _forbid_inference() -> None:
    """Fail if the inference entry points are loaded, and booby-trap the test-eval door."""
    # final.py itself may be loaded (the --stage analysis entry) but only imports baselines lazily
    # inside the stages that run inference; its inference callables are never invoked here.
    loaded = [m for m in ("intent_router.baselines", "intent_router.ood") if m in sys.modules]
    assert not loaded, f"analysis must not import inference code: {loaded}"

    def refuse(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("analysis is saved-files-only: inference/test evaluation is forbidden")

    evaluate_mod.evaluate_test_once = refuse  # type: ignore[assignment]
    if "intent_router.models" in sys.modules:
        sys.modules["intent_router.models"].predict = refuse  # type: ignore[attr-defined]


def _write_error_examples_md(
    path: Path, oof_conf: list[dict[str, Any]], test_errs: pd.DataFrame, labels: list[str],
    text_of: dict[str, str],
) -> None:  # fmt: skip
    """Example TEXTS go to the gitignored outputs dir only (never results/)."""
    lines = ["# Error examples (CONFIDENTIAL text; gitignored; do not copy into results/)\n"]
    lines.append("\n## OOF (probability-averaged, 426 rows) top-5 confusion pairs\n")
    for pair in oof_conf:
        lines.append(f"\n### {pair['gold']} -> {pair['pred']} (n={pair['count']})\n")
        lines += [f"- `{i}`: {text_of[i]}" for i in pair["example_ids"]]
    lines.append(f"\n## every test error, n={len(test_errs)} (most confident first)\n")
    lines.append("(conf = temperature-scaled max softmax probability)\n")
    for _, r in test_errs.iterrows():
        lines.append(
            f"- `{r['id']}` gold={labels[r['gold']]} pred={labels[r['pred']]} "
            f"conf={r['conf']:.3f}: {text_of[r['id']]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Regenerate slices.json, error_analysis.json and track_a.json's selective block.

    Reads ONLY saved CSV/JSON files (test/val predictions, OOF, eda rows); never runs inference,
    retrains, or calls evaluate_test_once. test_eval_log.jsonl line count is recorded before and
    after and asserted unchanged.
    """
    ap = argparse.ArgumentParser(description="Re-run slice/error analysis from saved files")
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args(argv)
    _forbid_inference()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    res_dir = Path(cfg["results_dir"])
    log_path = Path(cfg["test_eval_log"])
    lines_before = _log_lines(log_path)
    full = data_mod.load_data(cfg["data_path"])
    labels = list(data_mod.LABELS)
    k = len(labels)
    n_res, seed = int(cfg["bootstrap"]["n_resamples"]), int(cfg["bootstrap"]["seed"])
    an = cfg["analysis"]

    test = pd.read_csv(res_dir / "test_predictions.csv")
    val = pd.read_csv(res_dir / "val_predictions.csv")
    oof_path = Path(an["oof_dir"]) / f"{an['oof_config_id']}.csv"
    oof = load_oof_mean(oof_path, k, int(an["oof_predictions_per_id"]))
    eda_rows = pd.read_csv(an["eda_rows_path"])
    prob_cols = [f"prob_{i}" for i in range(k)]

    # ---- selective block of track_a.json: existing threshold, temperature-scaled MSP
    track_a = json.loads((res_dir / "track_a.json").read_text(encoding="utf-8"))
    temp = float(track_a["calibration"]["temperature"])
    old_sel = track_a["test"]["selective"]
    thr = float(old_sel["at_threshold"]["threshold"])

    def conf_t(df: pd.DataFrame) -> np.ndarray:
        logp = np.log(np.clip(df[prob_cols].to_numpy(dtype=np.float64), 1e-300, None))
        return softmax(logp, temp).max(axis=1)  # log p = logits - const, so T-scaling is exact

    test_conf = conf_t(test)
    val_conf = conf_t(val)
    new_at = selective_at_threshold(
        test_conf, test["gold"].to_numpy(), test["pred"].to_numpy(), thr, k
    )
    for key in ("n_accepted", "coverage", "accuracy"):  # must be the same accepted rows as before
        assert abs(new_at[key] - old_sel["at_threshold"][key]) < 1e-9, (key, new_at, old_sel)
    assert abs(float(np.mean(val_conf >= thr)) - old_sel["val_retention_at_threshold"]) < 1e-9
    new_at["macro_f1_present_previous_definition"] = old_sel["at_threshold"].get(
        "macro_f1_present_previous_definition", old_sel["at_threshold"]["macro_f1_present"]
    )
    new_at["macro_f1_present_definition"] = (
        "mean F1 over the classes present in the GOLD labels of the accepted rows (fixed label "
        "set); previous definition (gold OR predicted classes) kept above for reference"
    )
    old_sel["at_threshold"] = new_at

    # ---- slices
    ev_pred = pd.DataFrame(
        {"id": test["id"], "gold": test["gold"], "pred": test["pred"], "conf": test_conf}
    )
    sources = {"oof": oof[["id", "gold", "pred"]], "test": ev_pred[["id", "gold", "pred"]]}
    slices: dict[str, Any] = {
        "label_oof": "post-selection (optimistic) CV estimate (OOF probability-averaged "
        "predictions, 426 train+val rows)",
        "label_test": "unbiased (single test evaluation)",
        "label_combined": COMBINED_LABEL,
        "english_vs_non_english": "definition A: primary language en vs not en/undetermined",
        "short": "n_chars < 30",
        "indicative_only_rule": "n < 20",
        "macro_f1_definition": SLICE_MACRO_F1_DEFINITION,
        "bootstrap": {
            "n_resamples": n_res,
            "seed": seed,
            "macro_f1": "gold classes of the slice, fixed across resamples",
        },
        "sources": slice_report(sources, eda_rows, labels, n_res, seed),
    }

    # ---- error analysis
    raw_oof = pd.read_csv(oof_path)
    top_k, n_ex = int(an["top_confusions"]), int(an["examples_per_pair"])
    oof_top = top_confusions(oof, labels, top_k, n_ex)
    key = "hypothesis_errors_concentrate_in_shipment_information"
    err_analysis: dict[str, Any] = {
        "oof": {
            "label": "post-selection (optimistic) CV estimate",
            key: shipment_family_hypothesis(oof, labels),
            "top_confusions": oof_top,
        },
        "oof_all_fold_runs": {
            "label": "post-selection (optimistic) CV estimate; every individual fold-run "
            f"prediction of {an['oof_config_id']} (not probability-averaged)",
            "n_fold_runs": int(raw_oof.groupby(["fold_seed", "model_seed", "fold"]).ngroups),
            key: per_fold_run_hypothesis(raw_oof, labels, n_res, seed, top_k),
        },
        "test": {
            "label": "unbiased (single test evaluation)",
            key: shipment_family_hypothesis(ev_pred, labels),
            "top_confusions": top_confusions(ev_pred, labels, top_k, n_ex),
            "n_errors": int((ev_pred["gold"] != ev_pred["pred"]).sum()),
        },
    }
    errs = ranked_errors(ev_pred)
    text_of = dict(zip(full["id"], full["text"], strict=True))
    _write_error_examples_md(
        Path(cfg["outputs_dir"]) / "error_examples.md", oof_top, errs, labels, text_of
    )

    lines_after = _log_lines(log_path)
    assert lines_after == lines_before, "test_eval_log.jsonl changed during analysis"
    prov = {
        "inputs": [
            str(res_dir / "test_predictions.csv"),
            str(res_dir / "val_predictions.csv"),
            str(oof_path),
            str(an["eda_rows_path"]),
            str(res_dir / "track_a.json") + " (threshold, temperature)",
        ],
        "inference_run": False,
        "test_eval_log_lines_before": lines_before,
        "test_eval_log_lines_after": lines_after,
    }
    slices["analysis_regeneration"] = prov
    err_analysis["analysis_regeneration"] = prov
    old_sel["analysis_regeneration"] = prov
    _dump(res_dir / "slices.json", slices)
    _dump(res_dir / "error_analysis.json", err_analysis)
    _dump(res_dir / "track_a.json", track_a)
    print(f"[analysis] test_eval_log lines {lines_before} -> {lines_after}")


if __name__ == "__main__":
    main()
