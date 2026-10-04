from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router.stats import nadeau_bengio_ttest


def config_id(cfg: dict[str, Any], ablation: str | None) -> str:
    """Config identity = model x LR x ablation (seeds/folds are repetitions of it)."""
    cid = f"{cfg['model_short']}_lr{cfg['lr']:g}"
    return f"{cid}_{ablation}" if ablation else cid


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Load every run json under runs_dir (sorted for determinism)."""
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]


def _group(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        out.setdefault(config_id(r["config"], r.get("ablation")), []).append(r)
    return out


def _curve(runs: list[dict[str, Any]], key: str) -> np.ndarray:
    n_ep = min(len(r["epochs"]) for r in runs)
    return np.array([[e[key] for e in r["epochs"][:n_ep]] for r in runs])  # (runs, epochs)


def chosen_epoch(runs: list[dict[str, Any]]) -> int:
    """1-based epoch maximising the mean (over fold-runs) macro-F1 curve."""
    return int(np.argmax(_curve(runs, "macro_f1").mean(axis=0))) + 1


def summarize_config(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std at the chosen epoch plus cost metrics for one config's fold-runs."""
    ep = chosen_epoch(runs)
    f1 = _curve(runs, "macro_f1")[:, ep - 1]
    acc = _curve(runs, "accuracy")[:, ep - 1]
    cell: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for r, f, a in zip(runs, f1, acc, strict=True):
        cell.setdefault((r["fold_seed_idx"], r["fold"]), []).append((float(f), float(a)))
    per_fold = [
        {
            "fold_seed_idx": fs,
            "fold": fo,
            "macro_f1": float(np.mean([v[0] for v in vals])),
            "accuracy": float(np.mean([v[1] for v in vals])),
            "n_model_seeds": len(vals),
        }
        for (fs, fo), vals in sorted(cell.items())
    ]
    # Cost metrics only from runs that had the GPU to themselves (missing field => not exclusive);
    # F1/accuracy above use every run.
    excl = [r for r in runs if r.get("gpu_exclusive") is True]
    return {
        "n_runs": len(runs),
        "n_exclusive": len(excl),
        "n_total": len(runs),
        "any_non_exclusive": len(excl) < len(runs),
        "chosen_epoch": ep,
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else None,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else None,
        "mean_wall_clock_s": float(np.mean([r["wall_clock_s"] for r in excl])) if excl else None,
        "max_peak_vram_mb": float(max(r["peak_vram_mb"] for r in excl)) if excl else None,
        "any_nan": bool(any(r.get("nan_detected") for r in runs)),
        "mean_curve_macro_f1": [float(x) for x in _curve(runs, "macro_f1").mean(axis=0)],
        "per_fold": per_fold,
    }


def best_lr_per_model(runs_dir: Path) -> dict[str, float]:
    """Sweep winner per model: highest mean macro-F1 at own chosen epoch over the s0/ms0 runs
    with no ablation (the sweep protocol; confirm runs never influence it)."""
    sweep = [
        r
        for r in load_runs(runs_dir)
        if not r.get("ablation") and r["fold_seed_idx"] == 0 and r["model_seed"] == 0
    ]
    best: dict[str, tuple[float, float]] = {}
    for runs in _group(sweep).values():
        s = summarize_config(runs)
        cfg = runs[0]["config"]
        key = cfg["model_name"]
        if key not in best or s["macro_f1_mean"] > best[key][0]:
            best[key] = (s["macro_f1_mean"], float(cfg["lr"]))
    return {m: lr for m, (_, lr) in best.items()}


def write_oof(runs: list[dict[str, Any]], epoch: int, outputs_dir: Path, path: Path) -> None:
    """OOF predictions at the chosen epoch for every fold-run of a config (ids only, no text)."""
    frames = []
    for r in runs:
        z = np.load(outputs_dir / f"{r['run_id']}.npz", allow_pickle=False)
        probs = z["probs"][epoch - 1]
        df = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(probs.shape[1])])
        df.insert(0, "pred", probs.argmax(axis=1))
        df.insert(0, "gold", z["gold"])
        df.insert(0, "fold", r["fold"])
        df.insert(0, "model_seed", r["model_seed"])
        df.insert(0, "fold_seed", r["fold_seed_idx"])
        df.insert(0, "id", z["ids"])
        frames.append(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def top2_ttest(summaries: dict[str, dict[str, Any]], n_train: int, n_test: int) -> dict[str, Any]:
    """Nadeau-Bengio test between the two best no-ablation configs sharing the same fold cells."""
    cands = sorted(
        (c for c in summaries if "_" not in c.split("_lr", 1)[1]),
        key=lambda c: -summaries[c]["macro_f1_mean"],
    )
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            ka = {(p["fold_seed_idx"], p["fold"]): p["macro_f1"] for p in summaries[a]["per_fold"]}
            kb = {(p["fold_seed_idx"], p["fold"]): p["macro_f1"] for p in summaries[b]["per_fold"]}
            common = sorted(set(ka) & set(kb))
            if len(common) >= 15:
                t, p, df = nadeau_bengio_ttest(
                    np.array([ka[k] for k in common]),
                    np.array([kb[k] for k in common]),
                    n_train,
                    n_test,
                )
                return {"a": a, "b": b, "J": len(common), "t": t, "p": p, "df": df}
    return {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate bake-off fold-runs")
    ap.add_argument("--config", default="configs/bakeoff.yaml")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--oof-dir", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    res = Path(args.results_dir or cfg["results_dir"])
    outputs = Path(args.outputs_dir or cfg["outputs_dir"])
    oof = Path(args.oof_dir or cfg["oof_dir"])
    runs = load_runs(res / "runs")
    if not runs:
        raise SystemExit(f"no runs found in {res / 'runs'}")
    summaries: dict[str, dict[str, Any]] = {}
    for cid, rs in sorted(_group(runs).items()):
        s = summarize_config(rs)
        summaries[cid] = s
        write_oof(rs, s["chosen_epoch"], outputs, oof / f"{cid}.csv")
    r0 = runs[0]
    summary = {
        "configs": summaries,
        "best_lr_per_model": best_lr_per_model(res / "runs"),
        "top2_ttest": top2_ttest(summaries, r0["n_train"], r0["n_eval"]),
        "note": "epoch chosen per config as argmax of the mean-over-fold-runs macro-F1 curve",
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    for cid, s in summaries.items():
        print(
            f"{cid:40s} n={s['n_runs']:3d} ep={s['chosen_epoch']:2d} "
            f"F1={s['macro_f1_mean']:.4f} acc={s['accuracy_mean']:.4f} "
            f"excl={s['n_exclusive']}/{s['n_total']}"
            + (
                "  ** NON-EXCLUSIVE GPU RUNS (excluded from timing/VRAM) **"
                if s["any_non_exclusive"]
                else ""
            )
        )


if __name__ == "__main__":
    main()
