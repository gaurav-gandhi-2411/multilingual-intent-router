from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from intent_router.aggregate import _group, chosen_epoch, load_runs, summarize_config
from intent_router.stats import accuracy, macro_f1, nadeau_bengio_ttest

# Fixed before the runs: the ablation winner config.
PREREG_ABLATION_WINNER = "e5_lr3e-05"
ALPHA = 0.05
VERIFY_TOL = 1e-9  # recomputed-from-npz vs per-epoch metric stored in the run json

Cell = tuple[int, int]  # (fold_seed_idx, fold)


# ------------------------------------------------------------------ verification
def verify_run(run: dict[str, Any], outputs_dir: Path) -> float | None:
    """Recompute per-epoch macro-F1/accuracy from the npz probs and overwrite the run's values.

    Returns the max abs difference vs the json values, or None if the npz is missing.
    Raises AssertionError if the difference exceeds VERIFY_TOL.
    """
    path = outputs_dir / f"{run['run_id']}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    probs, gold = z["probs"], z["gold"]
    assert probs.shape[0] == len(run["epochs"]), f"{run['run_id']}: epoch count mismatch"
    worst = 0.0
    for e, p in zip(run["epochs"], probs, strict=True):
        pred = p.argmax(axis=1)
        f1, acc = macro_f1(gold, pred), accuracy(gold, pred)
        worst = max(worst, abs(f1 - e["macro_f1"]), abs(acc - e["accuracy"]))
        e["macro_f1"], e["accuracy"] = f1, acc
    assert worst <= VERIFY_TOL, f"{run['run_id']}: npz vs json metric diff {worst:.3e}"
    return worst


def verify_runs(runs: list[dict[str, Any]], outputs_dir: Path) -> dict[str, Any]:
    """Verify many runs; summarise max diff and how many had no npz."""
    diffs = [verify_run(r, outputs_dir) for r in runs]
    done = [d for d in diffs if d is not None]
    return {
        "n_runs": len(runs),
        "n_verified": len(done),
        "n_missing_npz": len(diffs) - len(done),
        "max_abs_diff": max(done) if done else None,
    }


# ------------------------------------------------------------------ helpers
def mean_ratio(runs: list[dict[str, Any]]) -> float:
    """n_test / n_train as mean held-out size over mean train size across fold-runs."""
    return float(np.mean([r["n_eval"] for r in runs]) / np.mean([r["n_train"] for r in runs]))


def count_params(model_name: str) -> int:
    """Parameters of the 12-way classifier, instantiated on the meta device (no weights, no GPU)."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    try:
        conf = AutoConfig.from_pretrained(model_name, num_labels=12, local_files_only=True)
    except OSError:
        conf = AutoConfig.from_pretrained(model_name, num_labels=12)
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(conf)
    return int(sum(p.numel() for p in model.parameters()))


def exclusive_wall_clock(runs: list[dict[str, Any]]) -> tuple[float | None, int]:
    """(mean wall-clock over runs with exclusive GPU access, n_exclusive)."""
    ex = [r["wall_clock_s"] for r in runs if r.get("gpu_exclusive") is True]
    return (float(np.mean(ex)) if ex else None, len(ex))


def cell_scores(runs: list[dict[str, Any]], epoch: int, key: str = "macro_f1") -> dict[Cell, float]:
    """Mean over model seeds of the metric at `epoch` (1-based) per (fold_seed, fold) cell."""
    acc: dict[Cell, list[float]] = {}
    for r in runs:
        acc.setdefault((r["fold_seed_idx"], r["fold"]), []).append(r["epochs"][epoch - 1][key])
    return {c: float(np.mean(v)) for c, v in acc.items()}


def paired_ttest(a: dict[Cell, float], b: dict[Cell, float], ratio: float) -> dict[str, Any]:
    """Nadeau-Bengio test of a - b over the cells both dicts share."""
    common = sorted(set(a) & set(b))
    t, p, df = nadeau_bengio_ttest(
        np.array([a[c] for c in common]), np.array([b[c] for c in common]), 1, ratio
    )
    return {
        "J": len(common),
        "mean_diff": float(np.mean([a[c] - b[c] for c in common])),
        "t": t,
        "p": p,
        "df": df,
        "n_test_over_n_train": ratio,
    }


# ------------------------------------------------------------------ A) selection
def select(
    cands: dict[str, list[dict[str, Any]]],
    baselines: dict[str, Any] | None,
    params: dict[str, int],
) -> dict[str, Any]:
    """Apply the selection rules to the fully-confirmed candidates (1-3) plus baseline context."""
    summ = {c: summarize_config(rs) for c, rs in cands.items()}
    order = sorted(summ, key=lambda c: -summ[c]["macro_f1_mean"])
    winner = order[0]
    w = summ[winner]
    thr = w["macro_f1_mean"] - w["macro_f1_std"]
    table = {}
    for c in order:
        s = summ[c]
        table[c] = {
            "chosen_epoch": s["chosen_epoch"],
            "n_runs": s["n_runs"],
            "macro_f1_mean": s["macro_f1_mean"],
            "macro_f1_std": s["macro_f1_std"],
            "accuracy_mean": s["accuracy_mean"],
            "accuracy_std": s["accuracy_std"],
            "params": params.get(c),
            "mean_wall_clock_s": s["mean_wall_clock_s"],
            "n_exclusive": s["n_exclusive"],
        }
    rule2 = []
    for c in order[1:]:
        s = summ[c]
        within = s["macro_f1_mean"] >= thr
        smaller = params.get(c) is not None and params[c] < params[winner]
        wc_c, wc_w = s["mean_wall_clock_s"], w["mean_wall_clock_s"]
        faster = wc_c is not None and wc_w is not None and wc_c < wc_w
        rule2.append(
            {
                "config": c,
                "within_1_std": bool(within),
                "gap_to_winner": w["macro_f1_mean"] - s["macro_f1_mean"],
                "smaller": bool(smaller),
                "faster": bool(faster),
                "preferred_by_rule2": bool(within and (smaller or faster)),
            }
        )
    out: dict[str, Any] = {
        "winner": winner,
        "rule1_winner_std_threshold": thr,
        "candidates": table,
        "rule2": rule2,
        "rule2_triggered": [r["config"] for r in rule2 if r["preferred_by_rule2"]],
    }
    if len(order) >= 2:
        ru = order[1]
        ratio = mean_ratio(cands[winner] + cands[ru])
        out["top2_ttest"] = {
            "a": winner,
            "b": ru,
            **paired_ttest(
                cell_scores(cands[winner], w["chosen_epoch"]),
                cell_scores(cands[ru], summ[ru]["chosen_epoch"]),
                ratio,
            ),
        }
    if baselines:
        ratio = mean_ratio(cands[winner])
        wc = cell_scores(cands[winner], w["chosen_epoch"])
        ctx = {}
        for name in ("B1", "B0"):
            if name in baselines:
                bc = {
                    (p["fold_seed_idx"], p["fold"]): p["macro_f1"]
                    for p in baselines[name]["per_fold"]
                }
                ctx[name] = {"a": winner, "b": name, **paired_ttest(wc, bc, ratio)}
        out["baseline_context_ttests"] = {
            "note": "context only; baselines are floors, not selection candidates",
            **ctx,
        }
    return out


# ------------------------------------------------------------------ B) ablations
def noise_band(winner_runs: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    """Std across model seeds of the per-seed mean macro-F1 at the winner's global epoch."""
    by_seed: dict[int, list[float]] = {}
    for r in winner_runs:
        by_seed.setdefault(r["model_seed"], []).append(r["epochs"][epoch - 1]["macro_f1"])
    means = {s: float(np.mean(v)) for s, v in sorted(by_seed.items())}
    vals = np.array(list(means.values()))
    return {
        "epoch": epoch,
        "per_seed_mean_macro_f1": means,
        "n_fold_runs_per_seed": {s: len(v) for s, v in sorted(by_seed.items())},
        "std_ddof1": float(vals.std(ddof=1)),
        "std_ddof0": float(vals.std(ddof=0)),
        "value": float(vals.std(ddof=1)),
    }


def decide(delta: float, band: float, p: float) -> dict[str, Any]:
    """Pre-registered rule: adopt iff delta > band AND p < 0.05; else reject."""
    beats_band, sig = delta > band, p < ALPHA
    if beats_band and sig:
        why = "adopt"
    elif not beats_band and sig:
        why = "reject: delta within noise band"
    elif beats_band:
        why = "reject: p >= 0.05"
    else:
        why = "reject: delta within noise band and p >= 0.05"
    return {
        "adopt": bool(beats_band and sig),
        "beats_noise_band": bool(beats_band),
        "p_below_alpha": bool(sig),
        "verdict": why,
    }


def ablation_result(
    name: str,
    abl_runs: list[dict[str, Any]],
    base_runs: list[dict[str, Any]],
    band: float,
    global_epoch: int,
    expected_cells: set[Cell],
) -> dict[str, Any]:
    """One ablation vs the winner's matching runs. Incomplete => status only, no decision."""
    cells = {(r["fold_seed_idx"], r["fold"]) for r in abl_runs}
    n = len(abl_runs)
    if n < len(expected_cells) or cells != expected_cells:
        return {
            "status": "incomplete",
            "n_finished": n,
            "n_expected": len(expected_cells),
            "decision": None,
        }
    e_a = chosen_epoch(abl_runs)
    e_b = chosen_epoch(base_runs)  # symmetric: baseline picks its own best epoch on these runs
    a_f1, b_f1 = cell_scores(abl_runs, e_a), cell_scores(base_runs, e_b)
    ratio = mean_ratio(abl_runs + base_runs)
    test = paired_ttest(a_f1, b_f1, ratio)
    delta = test["mean_diff"]
    a_acc, b_acc = cell_scores(abl_runs, e_a, "accuracy"), cell_scores(base_runs, e_b, "accuracy")
    sens = float(
        np.mean(list(a_f1.values())) - np.mean(list(cell_scores(base_runs, global_epoch).values()))
    )
    wc, n_ex = exclusive_wall_clock(abl_runs)
    wc_b, n_ex_b = exclusive_wall_clock(base_runs)
    return {
        "status": "complete",
        "n_finished": n,
        "ablation_epoch": e_a,
        "baseline_epoch_own_argmax": e_b,
        "baseline_global_epoch": global_epoch,
        "ablation_macro_f1": float(np.mean(list(a_f1.values()))),
        "baseline_macro_f1": float(np.mean(list(b_f1.values()))),
        "delta_macro_f1": delta,
        "delta_macro_f1_vs_baseline_at_global_epoch": sens,
        "delta_accuracy": float(
            np.mean([a_acc[c] - b_acc[c] for c in sorted(set(a_acc) & set(b_acc))])
        ),
        "noise_band": band,
        "ttest": test,
        "decision": decide(delta, band, test["p"]),
        "mean_wall_clock_s": wc,
        "n_exclusive": n_ex,
        "baseline_mean_wall_clock_s": wc_b,
        "baseline_n_exclusive": n_ex_b,
    }


# ------------------------------------------------------------------ driver
def _fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Selection and ablations")
    ap.add_argument("--config", default="configs/bakeoff.yaml")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--outputs-dir", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = Path(args.results_dir or cfg["results_dir"])
    outputs = Path(args.outputs_dir or cfg["outputs_dir"])
    runs = load_runs(res / "runs")
    full = len(cfg["fold_seed_idx"]) * len(cfg["model_seeds"]) * cfg["n_folds"]

    plain = [r for r in runs if not r.get("ablation")]
    groups = {c: rs for c, rs in _group(plain).items() if len(rs) == full}
    abl_all = [r for r in runs if r.get("ablation")]
    used = [r for rs in groups.values() for r in rs] + abl_all
    ver = verify_runs(used, outputs)  # also replaces json metrics by npz-recomputed values
    print(f"verification: {ver}")

    name_of = {c: rs[0]["config"]["model_name"] for c, rs in groups.items()}
    params = {c: count_params(m) for c, m in name_of.items()}
    bpath = res / "baselines.json"
    baselines = json.loads(bpath.read_text()) if bpath.exists() else None
    sel = select(groups, baselines, params)
    sel["verification"] = ver
    sel["n_runs_required_per_candidate"] = full
    sel["incomplete_configs"] = {c: len(rs) for c, rs in _group(plain).items() if len(rs) != full}
    (res / "selection.json").write_text(json.dumps(sel, indent=2))

    print(f"\nSELECTION (candidates with {full} runs; winner = {sel['winner']})")
    for c, t in sel["candidates"].items():
        print(
            f"{c:22s} ep={t['chosen_epoch']:2d} "
            f"F1={t['macro_f1_mean']:.4f}+-{t['macro_f1_std']:.4f} "
            f"acc={t['accuracy_mean']:.4f} params={t['params'] / 1e6:.1f}M "
            f"wall={_fmt(t['mean_wall_clock_s'], '.1f')}s (n_excl={t['n_exclusive']})"
        )
    print(f"rule 1 threshold (winner mean - 1 std) = {sel['rule1_winner_std_threshold']:.4f}")
    for r in sel["rule2"]:
        print(
            f"rule 2: {r['config']:20s} within_1std={r['within_1_std']} "
            f"gap={r['gap_to_winner']:.4f}"
            f" smaller={r['smaller']} faster={r['faster']} -> preferred={r['preferred_by_rule2']}"
        )
    for key in ("top2_ttest",):
        if key in sel:
            t = sel[key]
            print(
                f"top-2 NB: {t['a']} vs {t['b']}: diff={t['mean_diff']:.4f} t={t['t']:.3f} "
                f"p={t['p']:.4f} df={t['df']} J={t['J']}"
            )
    for name, t in sel.get("baseline_context_ttests", {}).items():
        if isinstance(t, dict):
            print(
                f"context NB: {t['a']} vs {name}: diff={t['mean_diff']:.4f} t={t['t']:.3f} "
                f"p={t['p']:.2e} df={t['df']} J={t['J']}"
            )

    # ---- ablations
    win = PREREG_ABLATION_WINNER
    out: dict[str, Any] = {"preregistered_winner": win, "selection_winner": sel["winner"]}
    if sel["winner"] != win:
        print(f"WARNING: selection winner {sel['winner']} != pre-registered {win}")
    if win not in groups:
        raise SystemExit(f"pre-registered ablation winner {win} has no complete {full}-run set")
    wruns = groups[win]
    g_epoch = chosen_epoch(wruns)
    ab_seeds, ab_fs = set(cfg["ablate"]["model_seeds"]), set(cfg["ablate"]["fold_seed_idx"])
    base = [r for r in wruns if r["model_seed"] in ab_seeds and r["fold_seed_idx"] in ab_fs]
    cells = {(fs, f) for fs in ab_fs for f in range(cfg["n_folds"])}
    band = noise_band(wruns, g_epoch)
    out["winner_global_epoch"] = g_epoch
    out["noise_band"] = band
    out["n_baseline_runs"] = len(base)
    out["ablations"] = {}
    wname, wlr = wruns[0]["config"]["model_name"], wruns[0]["config"]["lr"]
    print(
        f"\nABLATIONS vs {win} (global epoch {g_epoch}); noise band = {band['value']:.5f} "
        f"(ddof=1; ddof=0 {band['std_ddof0']:.5f}; per-seed {band['per_seed_mean_macro_f1']})"
    )
    for ab in cfg["ablations"]:
        ar = [
            r
            for r in abl_all
            if r["ablation"] == ab
            and r["config"]["model_name"] == wname
            and r["config"]["lr"] == wlr
        ]
        o = ablation_result(ab, ar, base, band["value"], g_epoch, cells)
        out["ablations"][ab] = o
        if o["status"] != "complete":
            print(f"{ab:16s} INCOMPLETE {o['n_finished']}/{o['n_expected']} runs: no decision")
            continue
        t = o["ttest"]
        print(
            f"{ab:16s} ep={o['ablation_epoch']:2d}(base {o['baseline_epoch_own_argmax']:2d}) "
            f"dF1={o['delta_macro_f1']:+.4f} "
            f"(vs global-ep base {o['delta_macro_f1_vs_baseline_at_global_epoch']:+.4f}) "
            f"dAcc={o['delta_accuracy']:+.4f} t={t['t']:.3f} p={t['p']:.4f} "
            f"wall={_fmt(o['mean_wall_clock_s'], '.1f')}s n_excl={o['n_exclusive']} "
            f"-> {o['decision']['verdict']}"
        )
    (res / "ablations.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
