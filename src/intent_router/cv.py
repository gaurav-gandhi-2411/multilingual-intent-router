from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import gpu_lock
from intent_router.data import get_frame, load_splits
from intent_router.models import model_short
from intent_router.train import TrainConfig, train_fold

SELCHECK_STAGES = ("selcheck", "selcheck_confirm")  # selection-robustness check post-hoc check
STAGES = ("sweep", "confirm", "ablate", "timing", *SELCHECK_STAGES)
TIMING_LR = 3e-5  # timing probe: fold 0 / fold-seed s0 / model seed 0 / lr 3e-5 / 1 epoch
PROBE_LABEL = "estimate, extrapolated from 1-epoch probe"


@dataclass(frozen=True)
class RunSpec:
    """One fold-run in the grid."""

    model_name: str
    lr: float
    fold_seed_idx: int
    model_seed: int
    fold: int
    ablation: str | None = None
    epochs: int | None = None  # set only for selection-check runs; keeps bake-off ids unchanged

    @property
    def run_id(self) -> str:
        ep = f"_ep{self.epochs}" if self.epochs else ""
        base = (
            f"{model_short(self.model_name)}_lr{self.lr:g}{ep}_fs{self.fold_seed_idx}"
            f"_ms{self.model_seed}_f{self.fold}"
        )
        return f"{base}_{self.ablation}" if self.ablation else base


# ------------------------------------------------------------------- CV data
def load_cv_frame(
    data_path: str = "data/dataset.csv", splits_path: str = "splits/splits.csv"
) -> pd.DataFrame:
    """train+val rows only (the 85% CV pool). Asserts no test id can be present."""
    frame = get_frame(["train", "val"], data_path, splits_path)
    test_ids = set(load_splits(splits_path).query("split == 'test'")["id"])
    leaked = test_ids & set(frame["id"])
    assert not leaked, f"test ids leaked into CV frame: {sorted(leaked)[:5]}"
    assert not (frame["split"] == "test").any(), "test rows in CV frame"
    return frame


def fold_split(
    frame: pd.DataFrame, fold_seed_idx: int, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, eval_df) for one (fold seed, fold); frame must be a CV frame."""
    assert "test" not in set(frame["split"]), "CV frame must not contain test rows"
    col = f"cv_fold_s{fold_seed_idx}"
    assert (frame[col] >= 0).all(), f"{col} has -1 (test) entries in CV frame"
    ev = frame[frame[col] == fold].reset_index(drop=True)
    tr = frame[frame[col] != fold].reset_index(drop=True)
    assert not set(ev["id"]) & set(tr["id"])
    return tr, ev


def resolve_max_len(cfg: dict[str, Any], eda_path: str = "results/eda.json") -> int:
    """cfg['max_len'] if an int; if 'auto', the recommended max_len recorded by the EDA."""
    ml = cfg.get("max_len", "auto")
    if isinstance(ml, int):
        return ml
    eda = json.loads(Path(eda_path).read_text())

    def find(obj: Any) -> int | None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "max_len" in k and isinstance(v, int | float) and not isinstance(v, bool):
                    return int(v)
            for v in obj.values():
                hit = find(v)
                if hit is not None:
                    return hit
        return None

    hit = find(eda)
    if hit is None:
        raise KeyError(f"no *max_len* key found in {eda_path}; set max_len in the config")
    return hit


# ------------------------------------------------------------------ run grids
def build_grid(
    stage: str,
    cfg: dict[str, Any],
    models: list[str],
    best_lr: dict[str, float],
    ablations: list[str] | None = None,
    lrs: list[float] | None = None,
    folds: list[int] | None = None,
) -> list[RunSpec]:
    """Deterministically ordered run grid for a stage (bake-off sweep and winner-only ablations)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    folds = folds if folds is not None else list(range(cfg["n_folds"]))
    grid: list[RunSpec] = []
    if stage in SELCHECK_STAGES:
        sc = cfg["selection_check"]
        if stage == "selcheck":
            sel_lrs = lrs or sc["lrs"]
            fss, mss = sc["fold_seed_idx"], sc["model_seeds"]
        else:
            if not lrs:
                raise ValueError("selcheck_confirm needs explicit --lrs")
            sel_lrs = lrs
            fss, mss = sc["confirm"]["fold_seed_idx"], sc["confirm"]["model_seeds"]
        for m in models:
            for lr in sel_lrs:
                for fs in fss:
                    for ms in mss:
                        grid += [RunSpec(m, lr, fs, ms, f, None, sc["epochs"]) for f in folds]
    elif stage == "sweep":
        for m in models:
            for lr in lrs or cfg["lr_grid"]:
                grid += [RunSpec(m, lr, 0, 0, f) for f in folds]
    elif stage == "confirm":
        for m in models:
            lr = best_lr[m]
            for fs in cfg["fold_seed_idx"]:
                for ms in cfg["model_seeds"]:
                    grid += [RunSpec(m, lr, fs, ms, f) for f in folds]
    else:
        names = ablations or list(cfg["ablations"])
        for m in models:
            lr = best_lr[m]
            for ab in names:
                for fs in cfg["ablate"]["fold_seed_idx"]:
                    for ms in cfg["ablate"]["model_seeds"]:
                        grid += [RunSpec(m, lr, fs, ms, f, ab) for f in folds]
    return grid


def _paths(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    res = Path(args.results_dir or cfg["results_dir"])
    out = Path(args.outputs_dir or cfg["outputs_dir"])
    return res / "runs", out


def _train_config(
    spec: RunSpec, cfg: dict[str, Any], group: str, epochs: int, flags: dict[str, Any]
) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model_name": spec.model_name,
            "lr": spec.lr,
            "epochs": epochs,
            "batch_size": cfg["batch_size"],
            "warmup_ratio": cfg["warmup_ratio"],
            "weight_decay": cfg["weight_decay"],
            "max_len": resolve_max_len(cfg),
            "precision": cfg["models"][spec.model_name]["precision"],
            "model_seed": spec.model_seed,
            "wandb_project": cfg["wandb"]["project"],
            "wandb_group": group,
            **flags,
        }
    )


def expected_run_s(cfg: dict[str, Any], model_name: str) -> int:
    """Per-fold-run duration estimate (configs/bakeoff.yaml models.<m>.expected_run_s)."""
    return int(cfg["models"][model_name]["expected_run_s"])


def locked_train(
    spec: RunSpec,
    cfg: dict[str, Any],
    frame: pd.DataFrame,
    tcfg: TrainConfig,
    expected_s: int,
    run_id: str,
) -> tuple[Any, dict[str, Any], int, int]:
    """take exclusive GPU access -> train_fold (snapshots + per-epoch checks inside) -> release.

    Any failure (lock, GPU query, W&B, training) propagates so the run fails rather than
    silently passing. Returns (FoldResult, lock info, n_train, n_eval).
    """
    train_df, eval_df = fold_split(frame, spec.fold_seed_idx, spec.fold)
    meta = {
        "run_id": run_id,
        "fold_seed_idx": spec.fold_seed_idx,
        "fold_seed": cfg_fold_seed(cfg, spec.fold_seed_idx),
        "fold": spec.fold,
        "ablation": spec.ablation or "none",
    }
    with gpu_lock.gpu_exclusive(expected_s, f"intent-router {run_id}") as lock_info:
        res = train_fold(tcfg, train_df, eval_df, meta)
    return res, lock_info, len(train_df), len(eval_df)


def gpu_fields(res: Any, lock_info: dict[str, Any]) -> dict[str, Any]:
    """The GPU-exclusivity record stored in every run json."""
    return {
        "gpu_snapshot_start": res.gpu_snapshot_start,
        "gpu_snapshot_end": res.gpu_snapshot_end,
        "gpu_exclusive": res.gpu_exclusive,
        "gpu_foreign_seen": res.gpu_foreign_seen,
        "gpu_lock_waited_s": lock_info["waited_s"],
    }


def execute(spec: RunSpec, cfg: dict[str, Any], frame: pd.DataFrame, args: Any) -> None:
    """Run one fold-run and write its npz + json (no raw text)."""
    runs_dir, out_dir = _paths(cfg, args)
    flags = cfg["ablations"][spec.ablation] if spec.ablation else {}
    group = args.group or (
        cfg["wandb"]["group_ablations"] if spec.ablation else cfg["wandb"]["group_bakeoff"]
    )
    tcfg = _train_config(spec, cfg, group, args.epochs or spec.epochs or cfg["epochs"], flags)
    res, lock_info, n_train, n_eval = locked_train(
        spec, cfg, frame, tcfg, expected_run_s(cfg, spec.model_name), spec.run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{spec.run_id}.npz",
        probs=res.probs,
        ids=np.array(res.eval_ids),
        gold=res.gold,
    )
    record = {
        "run_id": spec.run_id,
        "ablation": spec.ablation,
        "fold_seed_idx": spec.fold_seed_idx,
        "model_seed": spec.model_seed,
        "fold": spec.fold,
        "n_train": n_train,
        "n_eval": n_eval,
        "epochs": res.epochs,
        "wall_clock_s": res.wall_clock_s,
        "peak_vram_mb": res.peak_vram_mb,
        "wandb_url": res.wandb_url,
        "wandb_logged": res.wandb_logged,
        "nan_detected": res.nan_detected,
        "precision": res.precision,
        "config": res.config,
        **gpu_fields(res, lock_info),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Written last: the json is the "done" marker used for resumability.
    (runs_dir / f"{spec.run_id}.json").write_text(json.dumps(record, indent=2))


def extrapolate(timings: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Extrapolate sweep/confirm wall-clock from 1-epoch probes.

    per-run = load + E * (train_epoch + eval); runs per model: sweep = |lr_grid| * n_folds,
    confirm = |fold_seeds| * |model_seeds| * n_folds minus the n_folds sweep runs reused
    (the best LR's s0/ms0 runs). Per-epoch cost includes first-epoch CUDA warmup, so this
    leans high.
    """
    epochs = int(cfg["epochs"])
    n_folds = int(cfg["n_folds"])
    n_sweep = len(cfg["lr_grid"]) * n_folds
    n_confirm = len(cfg["fold_seed_idx"]) * len(cfg["model_seeds"]) * n_folds - n_folds
    per_model: dict[str, Any] = {}
    stage_s = {"sweep": 0.0, "confirm": 0.0}
    for name, t in timings.items():
        run_s = t["model_load_s"] + epochs * (t["train_epoch_s"] + t["eval_s"])
        per_model[name] = {
            "per_run_s": run_s,
            "sweep_runs": n_sweep,
            "confirm_runs": n_confirm,
            "sweep_h": run_s * n_sweep / 3600,
            "confirm_h": run_s * n_confirm / 3600,
        }
        stage_s["sweep"] += run_s * n_sweep
        stage_s["confirm"] += run_s * n_confirm
    return {
        "label": PROBE_LABEL,
        "assumptions": f"{epochs} epochs/run; per-run = load + epochs*(train_epoch + eval)",
        "per_model": per_model,
        "per_stage_h": {k: v / 3600 for k, v in stage_s.items()},
        "total_h": sum(stage_s.values()) / 3600,
    }


def run_timing(cfg: dict[str, Any], models: list[str], frame: pd.DataFrame, res_dir: Path) -> None:
    """1-epoch probe (fold 0, s0, ms0, lr 3e-5) per model, with exclusive GPU access.

    Writes <results_dir>/timing.json only; never touches runs/, so probes cannot enter the
    bake-off aggregate. Any failure propagates (no partial silent pass).
    """
    timings: dict[str, dict[str, Any]] = {}
    for name in models:
        spec = RunSpec(name, TIMING_LR, 0, 0, 0)
        run_id = f"timing_{model_short(name)}"
        print(f"[timing] {run_id}", flush=True)
        tcfg = _train_config(spec, cfg, "timing", 1, {})
        t0 = time.perf_counter()
        res, lock_info, _, _ = locked_train(
            spec, cfg, frame, tcfg, min(expected_run_s(cfg, name), 600), run_id
        )
        total = time.perf_counter() - t0
        timings[name] = {
            "model_load_s": res.load_s,
            "train_epoch_s": res.train_epoch_s[0],
            "eval_s": res.eval_epoch_s[0],
            "total_wall_s": total,  # includes lock wait + foreign-process wait
            "total_wall_excl_lock_wait_s": total - lock_info["waited_s"],
            "peak_vram_mb": res.peak_vram_mb,
            "precision": res.precision,
            "wandb_url": res.wandb_url,
            "wandb_logged": res.wandb_logged,
            **gpu_fields(res, lock_info),
        }
    out = {"timings": timings, "extrapolation": extrapolate(timings, cfg)}
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "timing.json").write_text(json.dumps(out, indent=2))
    ex = out["extrapolation"]
    print(
        f"[timing] {ex['label']}: sweep {ex['per_stage_h']['sweep']:.2f} h, "
        f"confirm {ex['per_stage_h']['confirm']:.2f} h, total {ex['total_h']:.2f} h"
    )


def cfg_fold_seed(cfg: dict[str, Any], idx: int) -> int:
    """Underlying StratifiedGroupKFold seed for cv_fold_s{idx} (from base config)."""
    base = yaml.safe_load(Path(cfg.get("base_config", "configs/base.yaml")).read_text())
    return int(base["split"]["cv_seeds"][idx])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bake-off CV runner")
    ap.add_argument("--config", default="configs/bakeoff.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: all)")
    ap.add_argument("--best-lr", nargs="*", default=[], help="override, e.g. xlm-roberta=3e-5")
    ap.add_argument("--ablations", nargs="*", default=None)
    ap.add_argument("--lrs", nargs="*", type=float, default=None, help="sweep LR subset")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke tests)")
    ap.add_argument("--group", default=None, help="override W&B group (smoke tests)")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    models = args.models or list(cfg["models"])
    unknown = set(models) - set(cfg["models"])
    if unknown:
        raise SystemExit(f"unknown models {sorted(unknown)}; config has {list(cfg['models'])}")
    if args.stage in SELCHECK_STAGES:
        sc = cfg["selection_check"]
        models = args.models or list(sc["models"])
        if args.stage == "selcheck_confirm" and not (args.models and args.lrs):
            raise SystemExit("selcheck_confirm requires --models and --lrs")
        if set(models) - set(sc["models"]):
            raise SystemExit(f"selcheck models must be within {sc['models']}")
        # Separate dirs so these runs never enter the bake-off aggregates.
        args.results_dir = args.results_dir or sc["results_dir"]
        args.outputs_dir = args.outputs_dir or sc["outputs_dir"]
        args.group = args.group or sc["wandb_group"]
    runs_dir, _ = _paths(cfg, args)

    if args.stage == "timing":
        print(f"stage=timing models={models} (1 epoch, fold 0, s0, ms0, lr {TIMING_LR:g})")
        if args.dry_run:
            return 0
        run_timing(cfg, models, load_cv_frame(), runs_dir.parent)
        return 0

    best_lr: dict[str, float] = {}
    if args.stage in ("confirm", "ablate"):
        from intent_router.aggregate import best_lr_per_model

        best_lr = best_lr_per_model(runs_dir)
        for item in args.best_lr:
            key, val = item.split("=")
            match = [m for m in cfg["models"] if m == key or model_short(m) == key]
            if len(match) != 1:
                raise SystemExit(f"--best-lr key {key!r} does not match one model")
            best_lr[match[0]] = float(val)
        missing = [m for m in models if m not in best_lr]
        if missing:
            raise SystemExit(f"no best LR for {missing}: run the sweep first or pass --best-lr")

    grid = build_grid(args.stage, cfg, models, best_lr, args.ablations, args.lrs, args.folds)
    todo = [s for s in grid if not (runs_dir / f"{s.run_id}.json").exists()]
    if args.max_runs:
        todo = todo[: args.max_runs]
    print(f"stage={args.stage} grid={len(grid)} done={len(grid) - len(todo)} todo={len(todo)}")
    if args.dry_run:
        for s in todo:
            print("  ", s.run_id)
        return 0

    frame = load_cv_frame()
    failures: list[str] = []
    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec.run_id}", flush=True)
        try:
            execute(spec, cfg, frame, args)
        except Exception:  # noqa: BLE001 - keep the long sweep going; report at the end
            traceback.print_exc()
            failures.append(spec.run_id)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if failures:
        print(f"FAILED runs ({len(failures)}): {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
