from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from intent_router.aggregate import _group, load_runs, summarize_config
from intent_router.selection import count_params, select, verify_runs

# Fixed before the runs (selection-robustness check): e5 lr 3e-5, fold-seed s0 / model seed 0 (20
# epochs).
COMPARATOR_CONFIG = "e5_lr3e-05"
COMPARATOR_EXPECTED = 0.9441
EDGE_LR = 1e-4  # top of the selection-robustness check LR grid
EDGE_EPOCH = 30  # last epoch of the selection-robustness check schedule


def s0_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plain (no-ablation) fold-seed s0 / model-seed 0 runs: the single-seed screening set."""
    return [
        r
        for r in runs
        if not r.get("ablation") and r["fold_seed_idx"] == 0 and r["model_seed"] == 0
    ]


def comparator(bakeoff_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """e5 lr3e-5 s0/ms0 score at its own argmax epoch; asserts it equals 0.9441 (4 dp)."""
    runs = _group(s0_runs(bakeoff_runs)).get(COMPARATOR_CONFIG, [])
    assert runs, f"no {COMPARATOR_CONFIG} s0/ms0 bake-off runs found"
    s = summarize_config(runs)
    assert round(s["macro_f1_mean"], 4) == COMPARATOR_EXPECTED, (
        f"comparator {s['macro_f1_mean']!r} != pre-registered {COMPARATOR_EXPECTED}"
    )
    return {"config": COMPARATOR_CONFIG, "exact_score": s["macro_f1_mean"], **s}


def decide(
    summaries: dict[str, dict[str, Any]], lrs: dict[str, float], comp_score: float
) -> dict[str, Any]:
    """Pre-registered rule: confirm every config whose score > comparator (strict).

    Also flags configs that are at an edge again (chosen epoch == 30, or lr == 1e-4).
    """
    return {
        "comparator_score": comp_score,
        "confirm": sorted(c for c, s in summaries.items() if s["macro_f1_mean"] > comp_score),
        "edge_epoch_30": sorted(c for c, s in summaries.items() if s["chosen_epoch"] == EDGE_EPOCH),
        "edge_lr_1e-4": sorted(c for c, lr in lrs.items() if lr == EDGE_LR),
    }


def confirm_selection(
    cand_runs: dict[str, list[dict[str, Any]]],
    e5_runs: list[dict[str, Any]],
    params: dict[str, int],
    expected: int,
) -> dict[str, Any]:
    """Re-apply the selection rules (1-3 via selection.select) over confirmed candidates + e5."""
    assert len(e5_runs) == expected, f"e5 has {len(e5_runs)} runs, expected {expected}"
    groups = {COMPARATOR_CONFIG: e5_runs}
    groups.update({c: rs for c, rs in cand_runs.items() if len(rs) == expected})
    return select(groups, None, params)


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="selection-robustness check selection-robustness check summary"
    )
    ap.add_argument("--config", default="configs/bakeoff.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    sc = cfg["selection_check"]
    res, outputs = Path(sc["results_dir"]), Path(sc["outputs_dir"])
    bake_res, bake_out = Path(cfg["results_dir"]), Path(cfg["outputs_dir"])

    bake_runs = load_runs(bake_res / "runs")
    sel_runs = load_runs(res / "runs")
    if not sel_runs:
        raise SystemExit(f"no runs found in {res / 'runs'}")
    e5_all = [r for r in bake_runs if r["config"]["model_name"].endswith("multilingual-e5-base")]
    ver_b = verify_runs(e5_all, bake_out)  # replaces json metrics by npz-recomputed values
    ver_s = verify_runs(sel_runs, outputs)
    print(f"verification bake-off(e5): {ver_b}\nverification selcheck: {ver_s}")

    comp = comparator(bake_runs)
    print(f"comparator {comp['config']} s0: ep={comp['chosen_epoch']} F1={comp['exact_score']!r}")

    summaries: dict[str, dict[str, Any]] = {}
    lrs: dict[str, float] = {}
    for cid, rs in sorted(_group(s0_runs(sel_runs)).items()):
        summaries[cid] = summarize_config(rs)
        lrs[cid] = float(rs[0]["config"]["lr"])
    decision = decide(summaries, lrs, comp["exact_score"])

    print(f"\n{'config':22s} n  ep  F1(s0)           acc              wall_s  vramMB  n_excl")
    for cid, s in summaries.items():
        print(
            f"{cid:22s} {s['n_runs']} {s['chosen_epoch']:3d} "
            f"{s['macro_f1_mean']:.4f}+-{_fmt(s['macro_f1_std'])} "
            f"{s['accuracy_mean']:.4f}+-{_fmt(s['accuracy_std'])} "
            f"{_fmt(s['mean_wall_clock_s'], '.0f'):>6s} {_fmt(s['max_peak_vram_mb'], '.0f'):>7s} "
            f"{s['n_exclusive']}/{s['n_total']}"
            + ("  > comparator" if cid in decision["confirm"] else "")
        )
    print(f"decision: confirm={decision['confirm']} (strict > {comp['exact_score']:.6f})")
    print(
        f"flags: edge_epoch_30={decision['edge_epoch_30']} edge_lr_1e-4={decision['edge_lr_1e-4']}"
    )

    # Configs with a full selcheck_confirm set (45 runs) vs e5's 45 bake-off runs.
    full = len(sc["confirm"]["fold_seed_idx"]) * len(sc["confirm"]["model_seeds"]) * cfg["n_folds"]
    sel_plain = [r for r in sel_runs if not r.get("ablation")]
    confirmed = {c: rs for c, rs in _group(sel_plain).items() if len(rs) == full}
    confirm_out: dict[str, Any] | None = None
    if confirmed:
        e5 = _group([r for r in e5_all if not r.get("ablation")])[COMPARATOR_CONFIG]
        names = {c: rs[0]["config"]["model_name"] for c, rs in confirmed.items()}
        names[COMPARATOR_CONFIG] = e5[0]["config"]["model_name"]
        params = {c: count_params(m) for c, m in names.items()}
        confirm_out = confirm_selection(confirmed, e5, params, full)
        print(f"\nCONFIRM (selection rules re-applied; winner = {confirm_out['winner']})")
        for c, t in confirm_out["candidates"].items():
            print(
                f"{c:22s} ep={t['chosen_epoch']:2d} "
                f"F1={t['macro_f1_mean']:.4f}+-{t['macro_f1_std']:.4f} "
                f"acc={t['accuracy_mean']:.4f} wall={_fmt(t['mean_wall_clock_s'], '.1f')}s"
            )
        print(f"rule 2 triggered: {confirm_out['rule2_triggered']}")
        if "top2_ttest" in confirm_out:
            t = confirm_out["top2_ttest"]
            print(f"NB: {t['a']} vs {t['b']}: diff={t['mean_diff']:.4f} p={t['p']:.4f}")

    summary = {
        "note": "post-hoc selection-robustness check check; never part of the bake-off aggregates",
        "comparator": comp,
        "configs": summaries,
        "decision": decision,
        "confirm": confirm_out,
        "verification": {"bakeoff_e5": ver_b, "selection_check": ver_s},
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
