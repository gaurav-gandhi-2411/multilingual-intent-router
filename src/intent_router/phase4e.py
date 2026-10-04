"""Multi-axis selection (pre-registered before any run).

Candidates v1 / a3 / a1 / a1a3 x model seeds {0,1,2}, scored on five axes against v1 (CV macro-F1,
Track B DEV, neutral-swap flip rate, translation agreement, calibration). Entry point:

    python -m intent_router.phase4e --stage <stage> [--cand <key>] [--smoke] [--no-wandb]

Stages: curves -> epoch -> folds -> trackb (GPU, per candidate), then analyze -> select (read saved
outputs only), plus reuse_check (read-only dry run of the 4c reuse gates). If the selection is
not v1: confirm_final -> final_retrain (phase4e_final.py, once, after `select`). Every stage is
resumable (a result file's existence is the "done" marker) and writes only ids and numbers to
results/phase4e/ (never dataset or synthetic text). Training, folds and Track B reuse phase4c.py /
trackb_improve.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import phase4c as p4
from intent_router import robustness as rb
from intent_router import trackb_improve as ti
from intent_router.cv import load_cv_frame
from intent_router.evaluate import ece_bins, fit_temperature, git_sha, softmax
from intent_router.models import model_short
from intent_router.stats import accuracy, macro_f1
from intent_router.trackb import gpu_section
from intent_router.train import train_fold

CANDIDATES = ("v1", "a3", "a1", "a1a3")
AXES = ("a", "b", "c", "d", "e")
STAGES = (
    "curves",
    "epoch",
    "folds",
    "trackb",
    "analyze",
    "select",
    "reuse_check",
    "all",
    "confirm_final",
    "final_retrain",
)
# Post-selection stages (phase4e_final.py); never part of `all` (they make logged test-split calls).
FINAL_STAGES = ("confirm_final", "final_retrain")
OOF_NAMES = ("clean", "swap", "noise", "mt")
# Float slack so a delta of exactly -0.010 (a difference of two means, never bit-exact) stays
# eligible while -0.0101 does not.
EPS = 1e-9
# phase4c config fields that do not change the trained weights (W&B bookkeeping only).
IGNORED_CONFIG_FIELDS = ("wandb_group", "wandb_project")


def recipe_for(key: str) -> p4.Recipe:
    """Recipes of the four candidates; a1a3 gets its own key so it never collides with 4c's comb."""
    table = {
        "v1": p4.Recipe("v1"),
        "a3": p4.Recipe("a3", a3=True),
        "a1": p4.Recipe("a1", a1=True),
        "a1a3": p4.Recipe("a1a3", a1=True, a3=True),
    }
    if key not in table:
        raise KeyError(
            f"{key!r} is not a Multi-axis selection candidate; expected one of {CANDIDATES}"
        )
    return table[key]


def write_json(path: Path, obj: Any) -> None:
    """rb.write_json with non-finite floats turned into null (strict JSON, fail-closed readers)."""

    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not math.isfinite(float(x)):
            return None
        return x

    rb.write_json(path, clean(obj))


# ================================================================================== environment
def make_env(args: argparse.Namespace) -> p4.Env:
    """phase4c.make_env on configs/phase4e.yaml, with the smoke sandbox redirected to 4e."""
    args.smoke_root = "outputs/phase4e_smoke"
    args.smoke_adopt = None
    return p4.make_env(args)


def model_seeds(env: p4.Env) -> list[int]:
    return [int(s) for s in env.pcfg["model_seeds"]]


def seed_results(env: p4.Env, cand: str, seed: int) -> Path:
    return env.P.results / cand / f"s{seed}"


def seed_outputs(env: p4.Env, cand: str, seed: int) -> Path:
    return env.P.outputs / cand / f"s{seed}"


def seed_env(env: p4.Env, cand: str, seed: int) -> p4.Env:
    """Env for one (candidate, model seed): guard.model_seed overridden (eval_fold and
    ensure_fold_model read it), Track B score tables / run jsons / npz under the seed directory."""
    pcfg = {**env.pcfg, "guard": {**env.pcfg["guard"], "model_seed": int(seed)}}
    P = p4.P4Paths(seed_results(env, cand, seed), seed_outputs(env, cand, seed), env.smoke)
    paths = ti.Paths(
        P.results / "trackb", P.outputs / "trackb_arrays",
        env.P.results / "test_inference_log.jsonl", env.smoke,
    )  # fmt: skip
    return p4.Env(pcfg, P, replace(env.ctx, P=paths), env.smoke, None, env.inputs)


def curve_path(env: p4.Env, cand: str, seed: int, fold: int) -> Path:
    return seed_results(env, cand, seed) / "curves" / f"f{fold}.json"


def epoch_path(env: p4.Env, cand: str) -> Path:
    return env.P.results / cand / "epoch.json"


def e_star_of(env: p4.Env, cand: str) -> int:
    """The chosen epoch of a candidate (from its epoch.json)."""
    ep = p4.read_json(epoch_path(env, cand))
    if ep is None:
        raise SystemExit(f"no epoch.json for {cand}: run --stage epoch first")
    return int(ep["e_star"])


# ====================================================================================== curves
def expected_config(tcfg: Any) -> dict[str, Any]:
    """The `config` block a curve file records for this TrainConfig (train.py adds model_short)."""
    return {**dataclasses.asdict(tcfg), "model_short": model_short(tcfg.model_name)}


def config_diff(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """{field: (stored, expected)} for every training-relevant field that differs."""
    keys = (set(stored) | set(expected)) - set(IGNORED_CONFIG_FIELDS)
    return {
        k: [stored.get(k, "<missing>"), expected.get(k, "<missing>")]
        for k in sorted(keys)
        if stored.get(k, "<missing>") != expected.get(k, "<missing>")
    }


def reuse_enabled(env: p4.Env, cand: str) -> bool:
    return (not env.smoke) and cand in env.pcfg["reuse_4c"]["keys"]


def check_curve_reuse(env: p4.Env, rec: p4.Recipe) -> dict[str, Any]:
    """Compare the 4c seed-0 guard curve files with what 4e would train, field by field.

    match=True only if, for every fold: the stored `config` equals the TrainConfig 4e builds
    (ignoring W&B group/project), the stored extras (factors, A3 row counts) equal the extras 4e
    builds, the file holds a full 20-epoch curve, no NaN was seen, and 4c's fold seed / factor
    settings / A3 input path equal 4e's.
    """
    rc = env.pcfg["reuse_4c"]
    key4c = rc["keys"].get(rec.key)
    out: dict[str, Any] = {"candidate": rec.key, "from": key4c, "match": False, "per_fold": []}
    if key4c is None:
        out["why"] = "no 4c counterpart"
        return out
    pc4 = yaml.safe_load(Path(rc["config"]).read_text(encoding="utf-8"))
    global_diff = {}
    for name, a, b in (
        ("guard.fold_seed_idx", pc4["guard"]["fold_seed_idx"], env.pcfg["guard"]["fold_seed_idx"]),
        ("guard.n_folds", pc4["guard"]["n_folds"], env.pcfg["guard"]["n_folds"]),
        ("factors", pc4["factors"], env.pcfg["factors"]),
        ("inputs.a3_aug", pc4["inputs"]["a3_aug"], env.pcfg["inputs"]["a3_aug"]),
        ("inputs.eval_mt", pc4["inputs"]["eval_mt"], env.pcfg["inputs"]["eval_mt"]),
    ):
        if a != b:
            global_diff[name] = [a, b]
    out["global_diff"] = global_diff
    seed = int(rc["seed"])
    es = seed_env(env, rec.key, seed)
    tcfg = p4.train_cfg_for(es, rec, seed, None)
    want = expected_config(tcfg)
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    ok = not global_diff
    for f in env.folds:
        src = Path(rc["results_dir"]) / "guard" / f"{key4c}_f{f}.json"
        rec_f: dict[str, Any] = {"fold": f, "file": str(src)}
        if not src.exists():
            rec_f["why"] = "missing"
            ok = False
            out["per_fold"].append(rec_f)
            continue
        stored = json.loads(src.read_text(encoding="utf-8"))
        tr, ev = p4.fold_data(es, frame, f)
        ex = p4.build_extras(rec, es.inputs if rec.factors else None, es.pcfg, tr, p4.cv_forbid(ev))
        diff = config_diff(stored["config"], want)
        rec_f.update(
            config_diff=diff,
            extras_equal=stored["extras"] == ex.info,
            n_epochs=len(stored["macro_f1"]),
            fold_ok=int(stored["fold"]) == f,
            nan_detected=bool(stored["nan_detected"]),
        )
        if not (
            not diff and rec_f["extras_equal"] and rec_f["n_epochs"] == 20 and rec_f["fold_ok"]
            and not rec_f["nan_detected"]
        ):  # fmt: skip
            ok = False
        out["per_fold"].append(rec_f)
    out["match"] = bool(ok)
    return out


def print_reuse_report(rep: Mapping[str, Any]) -> None:
    """Human-readable match / mismatch lines of check_curve_reuse."""
    print(f"[reuse] {rep['candidate']} <- 4c {rep['from']}: curves match={rep['match']}")
    if rep.get("why"):
        print(f"[reuse]   {rep['why']}")
    for k, v in rep.get("global_diff", {}).items():
        print(f"[reuse]   MISMATCH {k}: 4c={v[0]} 4e={v[1]}")
    for r in rep["per_fold"]:
        if r.get("why"):
            print(f"[reuse]   fold {r['fold']}: {r['why']} {r['file']}")
            continue
        print(
            f"[reuse]   fold {r['fold']}: config_diff={r['config_diff'] or 'none'} "
            f"extras_equal={r['extras_equal']} epochs={r['n_epochs']} nan={r['nan_detected']}"
        )


def run_curves(env: p4.Env, rec: p4.Recipe) -> None:
    """15 fold-runs (3 seeds x 5 folds): per-epoch held-out macro-F1, one file per run."""
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    reuse: dict[str, Any] | None = None
    if reuse_enabled(env, rec.key):
        reuse = check_curve_reuse(env, rec)
        print_reuse_report(reuse)
        if not reuse["match"]:
            print(f"[curves] {rec.key}: 4c curves NOT reused (mismatch above); training all 15")
    reuse_seed = int(env.pcfg["reuse_4c"]["seed"])
    for seed in model_seeds(env):
        es = seed_env(env, rec.key, seed)
        tcfg = p4.train_cfg_for(es, rec, seed, None)
        for f in env.folds:
            fp = curve_path(env, rec.key, seed, f)
            if fp.exists():
                continue
            if reuse is not None and reuse["match"] and seed == reuse_seed:
                src = Path(env.pcfg["reuse_4c"]["results_dir"]) / "guard"
                src = src / f"{reuse['from']}_f{f}.json"
                frec = json.loads(src.read_text(encoding="utf-8"))
                frec |= {"candidate": rec.key, "seed": seed, "reused_from": str(src),
                         "config_check": reuse["per_fold"][f]}  # fmt: skip
                write_json(fp, frec)
                print(f"[curves] {rec.key} s{seed} f{f}: reused {src}")
                continue
            tr, ev = p4.fold_data(es, frame, f)
            ex = p4.build_extras(
                rec, es.inputs if rec.factors else None, es.pcfg, tr, p4.cv_forbid(ev)
            )
            meta = {"run_id": f"p4e_curve_{rec.key}_s{seed}_f{f}", "fold": f, "candidate": rec.key}
            with gpu_section(
                int(env.pcfg["gpu_lock_expected_s"]),
                f"intent-router phase4e curve {rec.key} s{seed} f{f}",
                float(env.pcfg["gpu_lock_poll_s"]),
            ):
                res = train_fold(
                    tcfg, tr, ev, meta, oe_texts=ex.oe_texts, extra_train_rows=ex.extra_rows
                )
            p4._free_gpu()  # noqa: SLF001
            frec = {
                "candidate": rec.key, "seed": seed, "fold": f, "extras": ex.info,
                "macro_f1": [e["macro_f1"] for e in res.epochs],
                "accuracy": [e["accuracy"] for e in res.epochs],
                "wall_clock_s": res.wall_clock_s, "peak_vram_mb": res.peak_vram_mb,
                "wandb_url": res.wandb_url, "nan_detected": res.nan_detected,
                "gpu_exclusive": res.gpu_exclusive, "config": res.config,
            }  # fmt: skip
            write_json(fp, frec)
            print(f"[curves] {rec.key} s{seed} f{f}: best-epoch F1 {max(frec['macro_f1']):.4f}")


# ======================================================================================= epoch
def load_curve_matrix(env: p4.Env, cand: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """[n_runs, n_epochs] macro-F1 curves of every (seed, fold) run, ordered by seed then fold."""
    rows, keys = [], []
    for seed in model_seeds(env):
        for f in env.folds:
            fp = curve_path(env, cand, seed, f)
            if not fp.exists():
                raise SystemExit(f"missing curve {fp}: run --stage curves first")
            rows.append(json.loads(fp.read_text(encoding="utf-8"))["macro_f1"])
            keys.append((seed, f))
    return np.array(rows, dtype=float), keys


def epoch_summary(curves: np.ndarray, fixed_epoch: int | None = None) -> dict[str, Any]:
    """Pooled mean-curve argmax (ties -> earliest epoch) and the epoch actually used.

    fixed_epoch (v1's shipped epoch 9) overrides e*; the own argmax and the pooled F1 at both
    epochs are always recorded (the axis-(a) sensitivity check for v1 needs the argmax one).
    """
    mean = curves.mean(axis=0)
    own = int(np.argmax(mean)) + 1  # np.argmax returns the first maximum => ties go to the earliest
    if fixed_epoch is not None and not 1 <= fixed_epoch <= len(mean):
        raise ValueError(f"fixed epoch {fixed_epoch} outside the {len(mean)}-epoch curve")
    e_star = own if fixed_epoch is None else int(fixed_epoch)
    return {
        "e_star": e_star, "own_argmax": own, "fixed": fixed_epoch is not None,
        "f1_at_e_star": float(mean[e_star - 1]), "f1_at_own_argmax": float(mean[own - 1]),
        "mean_curve": mean.tolist(), "n_runs": int(curves.shape[0]),
    }  # fmt: skip


def stage_epoch(env: p4.Env, rec: p4.Recipe) -> None:
    out = epoch_path(env, rec.key)
    if out.exists():
        print(f"[epoch] {rec.key}: epoch.json exists, skipping")
        return
    curves, keys = load_curve_matrix(env, rec.key)
    fixed = int(env.pcfg["v1"]["deployed_epoch"]) if rec.key == "v1" and not env.smoke else None
    summ = epoch_summary(curves, fixed)
    per_seed = {
        str(s): int(np.argmax(curves[[i for i, k in enumerate(keys) if k[0] == s]].mean(0))) + 1
        for s in model_seeds(env)
    }
    write_json(out, {"candidate": rec.key, **summ, "per_seed_own_argmax": per_seed,
                     "smoke": env.smoke, "git_sha": git_sha()})  # fmt: skip
    print(f"[epoch] {rec.key}: e*={summ['e_star']} (own argmax {summ['own_argmax']}) "
          f"F1@e*={summ['f1_at_e_star']:.4f}")  # fmt: skip


# ======================================================================================= folds
def delete_fold_model(fdir: Path) -> bool:
    """Remove one fold-model directory. Refuses (AssertionError) anything that is not
    .../phase4e*/.../fold_models/fold<k>; returns whether something was deleted."""
    p = fdir.resolve()
    if not any(part in ("phase4e", "phase4e_smoke") for part in p.parts):
        raise AssertionError(f"refusing to delete outside outputs/phase4e*: {p}")
    if p.parent.name != "fold_models" or not p.name.startswith("fold"):
        raise AssertionError(f"refusing to delete a non fold-model path: {p}")
    if not p.exists():
        return False
    shutil.rmtree(p)
    return True


def part_paths(parts_dir: Path, fold: int) -> dict[str, Path]:
    return {n: parts_dir / f"f{fold}_{n}.csv" for n in (*OOF_NAMES, "meta")} | {
        "meta": parts_dir / f"f{fold}_meta.json"
    }


def summarize_oof(env: p4.Env, cand: str, seed: int, e_star: int, frames: Mapping[str, Any],
                  extra: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The per-(candidate, seed) folds.json summary (same fields as phase4c's folds.json)."""
    clean = frames["clean"]
    return {
        "candidate": cand, "seed": seed, "deployed_epoch": e_star, "smoke": env.smoke,
        "oof_accuracy": accuracy(clean["gold"], clean["pred"]),
        "oof_macro_f1": macro_f1(clean["gold"].to_numpy(), clean["pred"].to_numpy()),
        "n_rows": len(clean), "swap": p4.swap_summary(frames["swap"]),
        "noise": p4.noise_summary(frames["noise"]),
        "translation": p4.mt_summary(
            p4.mt_with_bool(frames["mt"]), env.pcfg["translation"]["langs"]
        ),
        **extra, "git_sha": git_sha(),
    }  # fmt: skip


def check_oof_reuse(
    env: p4.Env, rec: p4.Recipe, seed: int, e_star: int | None, dry_run: bool = False
) -> dict[str, Any]:
    """Gates for copying the 4c seed-0 OOF csvs instead of retraining (all must hold):
    seed == 4c's seed; 4e e* == the 4c deployed epoch; fold_seed / model_seed columns equal 4e's;
    every 4c fold model's run_meta has stop_epoch == e*, model_seed, eager attention and (where
    recorded) the same recipe; and, where the 4e curves exist, the 4c fold model's final-epoch
    macro-F1 equals the 4e curve value at e* (a determinism cross-check, tolerance 1e-6)."""
    rc = env.pcfg["reuse_4c"]
    key4c = rc["keys"].get(rec.key)
    out: dict[str, Any] = {"candidate": rec.key, "seed": seed, "from": key4c, "ok": False}
    if env.smoke or key4c is None or seed != int(rc["seed"]):
        out["why"] = "not a 4c seed-0 counterpart"
        return out
    d4 = Path(rc["results_dir"]) / key4c
    folds = p4.read_json(d4 / "folds.json")
    if folds is None or not all((d4 / f"oof_{n}.csv").exists() for n in OOF_NAMES):
        out["why"] = "4c OOF files missing"
        return out
    out["deployed_epoch_4c"] = int(folds["deployed_epoch"])
    out["e_star_4e"] = e_star
    checks: dict[str, Any] = {
        "epoch_equal": e_star is not None and e_star == folds["deployed_epoch"]
    }
    clean = pd.read_csv(d4 / "oof_clean.csv", usecols=["fold_seed", "model_seed"])
    g = env.pcfg["guard"]
    checks["fold_seed_equal"] = set(clean["fold_seed"]) == {int(g["fold_seed_idx"])}
    checks["model_seed_equal"] = set(clean["model_seed"]) == {seed}
    meta_ok, cross = True, []
    for fm in folds["folds"]:
        m = fm["run_meta"]
        if m is None or e_star is None:
            meta_ok = False
            continue
        meta_ok &= int(m["stop_epoch"]) == e_star and int(m["model_seed"]) == seed
        meta_ok &= m["attn_implementation"] == "eager"
        if "recipe" in m:
            meta_ok &= m["recipe"] == {f: getattr(rec, f) for f in p4.FACTORS}
        cp = curve_path(env, rec.key, seed, int(fm["fold"]))
        if not cp.exists() and dry_run:  # 4c guard curve == what the curves stage would copy
            cp = Path(rc["results_dir"]) / "guard" / f"{key4c}_f{fm['fold']}.json"
        if cp.exists():
            curve = json.loads(cp.read_text(encoding="utf-8"))["macro_f1"]
            cross.append(abs(curve[e_star - 1] - m["final_epoch_eval"]["macro_f1"]))
    checks["run_meta_equal"] = bool(meta_ok)
    checks["curve_vs_fold_model_max_abs_diff"] = max(cross) if cross else None
    checks["curve_cross_check_ok"] = bool(cross) and max(cross) <= 1e-6
    out["checks"] = checks
    out["ok"] = bool(
        checks["epoch_equal"] and checks["fold_seed_equal"] and checks["model_seed_equal"]
        and checks["run_meta_equal"] and checks["curve_cross_check_ok"]
    )  # fmt: skip
    return out


def stage_folds_seed(env: p4.Env, rec: p4.Recipe, seed: int) -> None:
    """OOF clean / swap / noise / MT predictions of the 15 -> (per seed) 5 fold models at e*.

    Per-fold prediction parts are written under outputs/ first; the fold model is deleted right
    after (disk), so a crash never costs more than one fold. folds.json is the done marker."""
    import torch

    es = seed_env(env, rec.key, seed)
    out_dir = seed_results(env, rec.key, seed)
    if (out_dir / "folds.json").exists():
        print(f"[folds] {rec.key} s{seed}: folds.json exists, skipping")
        return
    e_star = e_star_of(env, rec.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    reuse = check_oof_reuse(env, rec, seed, e_star)
    if reuse["ok"]:
        src = Path(env.pcfg["reuse_4c"]["results_dir"]) / reuse["from"]
        for n in OOF_NAMES:
            shutil.copyfile(src / f"oof_{n}.csv", out_dir / f"oof_{n}.csv")
        frames = {n: pd.read_csv(out_dir / f"oof_{n}.csv") for n in OOF_NAMES}
        write_json(out_dir / "folds.json", summarize_oof(
            env, rec.key, seed, e_star, frames, {"reused_from": str(src), "reuse_check": reuse}
        ))  # fmt: skip
        print(f"[folds] {rec.key} s{seed}: reused 4c OOF predictions from {src}")
        return
    if reuse.get("why") != "not a 4c seed-0 counterpart":
        print(f"[folds] {rec.key} s{seed}: 4c OOF NOT reused: {reuse}")

    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    t = env.fcfg["train"]
    bs = int(env.pcfg["predict_batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts_dir = seed_outputs(env, rec.key, seed) / "oof_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    models_root = seed_outputs(env, rec.key, seed) / "fold_models"
    for k in env.folds:
        pp = part_paths(parts_dir, k)
        fdir = models_root / f"fold{k}"
        if not all(p.exists() for p in pp.values()):
            fdir = p4.ensure_fold_model(
                es, rec, frame, k, e_star, tag="p4e", label_prefix="phase4e", fdir=fdir
            )
            _, ev = p4.fold_data(es, frame, k)
            with gpu_section(600, f"intent-router phase4e eval {rec.key} s{seed} f{k}",
                             float(env.pcfg["gpu_lock_poll_s"])):  # fmt: skip
                tok, model = rb._load_classifier(fdir, device)  # noqa: SLF001
                try:
                    got = p4.eval_fold(model, tok, t["model_name"], int(t["max_len"]), bs, ev,
                                       es.inputs.eval_mt, es.pcfg, k)  # fmt: skip
                finally:
                    model = None
                    p4._free_gpu()  # noqa: SLF001
            for n, df in got.items():
                df.to_csv(pp[n], index=False, float_format="%.9g")
            rm = p4.read_json(fdir / "run_meta.json") or {}
            write_json(pp["meta"], {"fold": k, "run_meta": rm})  # meta last = part set complete
        if delete_fold_model(fdir):
            print(f"[folds] {rec.key} s{seed} fold {k}: fold model deleted after OOF extraction")
    frames, metas = {}, []
    for n in OOF_NAMES:
        dfs = [pd.read_csv(part_paths(parts_dir, k)[n]) for k in env.folds]
        keep = [d for d in dfs if len(d)] or dfs[:1]  # concat of empty frames upcasts dtypes
        frames[n] = pd.concat(keep, ignore_index=True)
        frames[n].to_csv(out_dir / f"oof_{n}.csv", index=False, float_format="%.9g")
    cross = []
    for k in env.folds:
        m = json.loads(part_paths(parts_dir, k)["meta"].read_text(encoding="utf-8"))
        metas.append(m)
        fe = (m["run_meta"] or {}).get("final_epoch_eval")
        cp = curve_path(env, rec.key, seed, k)
        if fe and cp.exists():
            cross.append(abs(json.loads(cp.read_text(encoding="utf-8"))["macro_f1"][e_star - 1]
                             - fe["macro_f1"]))  # fmt: skip
    summ = summarize_oof(env, rec.key, seed, e_star, frames, {
        "folds": metas, "reuse_check": reuse,
        "curve_vs_fold_model_max_abs_diff": max(cross) if cross else None,
    })  # fmt: skip
    write_json(out_dir / "folds.json", summ)
    print(
        f"[folds] {rec.key} s{seed}: OOF acc {summ['oof_accuracy']:.4f} "
        f"flip {summ['swap']['pooled']['flip_rate']} drop {summ['noise']['drop']:.4f}"
    )


def stage_folds(env: p4.Env, rec: p4.Recipe) -> None:
    for seed in model_seeds(env):
        stage_folds_seed(env, rec, seed)


# ====================================================================================== trackb
def stage_trackb(env: p4.Env, rec: p4.Recipe) -> None:
    """5 DEV LOCO runs per model seed (15), stopped at e*, scored with every method (CSV last).

    The per-run npz arrays (about 9 MB each, measured on the 4c runs) are kept: 60 runs are about
    0.55 GB and keeping them lets any score be recomputed without retraining."""
    e = e_star_of(env, rec.key)
    for seed in model_seeds(env):
        es = seed_env(env, rec.key, seed)
        cand = p4.candidate_for(es, rec.key, e)
        for run in ti.plan(es.ctx, cand, es.dev_classes, [], [seed], "phase4e_trackb"):
            p4.run_one_p4c(es, rec, run)


# ================================================================================ pure statistics
def safe_gain(point: float, samples: np.ndarray, level: float) -> dict[str, Any]:
    """phase4c.gain_record, but NaN-tolerant: no finite resample => NaN CI (fails closed)."""
    if not np.isfinite(point) or not np.isfinite(samples).any():
        nan = float("nan")
        return {"point": float(point), "lo": nan, "hi": nan, "ci_excludes_zero_positive": False,
                "n_resamples": 0}  # fmt: skip
    return p4.gain_record(point, samples, level)


def fit_calibration(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Temperature (NLL minimisation on log of the saved OOF probabilities), the temperature-scaled
    confidence of every row, correctness, and the ECE (n_bins equal-width bins)."""
    logp = np.log(np.clip(probs, 1e-300, None))
    temp = fit_temperature(logp, gold)
    cal = softmax(logp, temp)
    ece, _ = ece_bins(cal, gold, n_bins)
    return {"T": float(temp), "conf": cal.max(axis=1), "correct": cal.argmax(1) == gold,
            "ece": float(ece)}  # fmt: skip


def ece_samples(
    conf: np.ndarray, correct: np.ndarray, idx: np.ndarray, n_bins: int = 15
) -> np.ndarray:
    """ECE of each resample (idx [B, n] row indices) with the same binning as evaluate.ece_bins:
    ECE = sum over bins of |sum(correct) - sum(conf)| / n."""
    c, ok = conf[idx], correct[idx].astype(float)
    b = np.clip(np.ceil(c * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = np.zeros(idx.shape[0])
    for k in range(n_bins):
        m = b == k
        total += np.abs((ok * m).sum(axis=1) - (c * m).sum(axis=1))
    return total / idx.shape[1]


@dataclass
class CandData:
    """Everything the five axes need from one candidate's saved outputs (arrays only)."""

    key: str
    n_factors: int
    e_star: int
    f1: np.ndarray  # [S, F] held-out macro-F1 at e* per (seed, fold)
    tb_units: list[list[ti.BootUnit]]  # [DEV class][seed]
    flips: np.ndarray  # [S, n_clusters] flipped swap instances per id cluster
    inst: np.ndarray  # [n_clusters] swap instances per id cluster (identical across seeds)
    agree: np.ndarray  # [S, n_ids, L] translation agreement with the English prediction
    kept: np.ndarray  # [n_ids, L] LaBSE-kept mask
    probs: np.ndarray  # [S, n_rows, K] OOF probabilities
    gold: np.ndarray  # [n_rows]
    keys: dict[str, Any] = dataclasses.field(default_factory=dict)  # row-identity fingerprints


def flip_rate_samples(flips: np.ndarray, inst: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per resample: mean over seeds of (sum flips / sum instances) over the drawn id clusters."""
    den = inst[idx].sum(axis=1).astype(float)
    den = np.where(den > 0, den, np.nan)
    return np.mean([flips[s][idx].sum(axis=1) / den for s in range(flips.shape[0])], axis=0)


def flip_rate_point(flips: np.ndarray, inst: np.ndarray) -> float:
    return float("nan") if inst.sum() == 0 else float(np.mean(flips.sum(axis=1) / inst.sum()))


def agreement_samples(agree: np.ndarray, kept: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per resample: mean over seeds and languages of the kept-row agreement (ids resampled)."""
    n_seeds, _, n_langs = agree.shape
    vals = []
    for s in range(n_seeds):
        for j in range(n_langs):
            den = kept[:, j][idx].sum(axis=1).astype(float)
            vals.append(
                (agree[s, :, j] * kept[:, j])[idx].sum(axis=1) / np.where(den > 0, den, np.nan)
            )
    return np.mean(vals, axis=0)


def agreement_point(agree: np.ndarray, kept: np.ndarray) -> float:
    vals = []
    for s in range(agree.shape[0]):
        for j in range(agree.shape[2]):
            k = kept[:, j]
            vals.append(float((agree[s, :, j] * k).sum() / k.sum()) if k.sum() else float("nan"))
    return float(np.mean(vals))


def class_seeds(n_classes: int, seed: int) -> list[int]:
    """One independent, deterministic bootstrap seed per DEV class, derived from `seed` (42), so
    classes are resampled independently while every candidate shares each class's draws."""
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n_classes)]


def b_stats(cd: CandData, n_boot: int, seeds: Sequence[int], level: float) -> dict[str, Any]:
    """DEV-mean AUROC / strict rejection@95: point and per-resample samples. Rows are resampled
    within each class stratified known/unknown (draws shared by the seeds of a class), seeds are
    averaged inside each resample, the DEV mean is taken per resample."""
    res = [ti.bootstrap_mean_ci(units, n_boot, s, True, level)
           for units, s in zip(cd.tb_units, seeds, strict=True)]  # fmt: skip
    out: dict[str, Any] = {}
    for name, m in (("auroc", "auroc"), ("rej95", "strict_recall_95")):
        out[name] = (float(np.mean([r["point"][m] for r in res])),
                     np.mean([r["samples"][m] for r in res], axis=0))  # fmt: skip
    return out


def paired_record(
    cand: tuple[float, np.ndarray], ref: tuple[float, np.ndarray], higher_is_better: bool,
    level: float,
) -> dict[str, Any]:  # fmt: skip
    """Candidate-vs-v1 improvement record. `delta` is cand - ref; the CI is of the improvement
    (delta when higher is better, ref - cand = the reduction when lower is better)."""
    sign = 1.0 if higher_is_better else -1.0
    imp_pt = sign * (cand[0] - ref[0])
    rec = safe_gain(imp_pt, sign * (cand[1] - ref[1]), level)
    return {
        "candidate": cand[0], "v1": ref[0], "delta": cand[0] - ref[0],
        "improvement": rec["point"], "lo": rec["lo"], "hi": rec["hi"],
        "ci_of": "delta" if higher_is_better else "reduction (v1 - candidate)",
        "ci_excludes_zero_positive": rec["ci_excludes_zero_positive"],
        "improved": bool(rec["ci_excludes_zero_positive"] and imp_pt > 0),
        "n_resamples": rec["n_resamples"],
    }  # fmt: skip


def eligible_drop(delta: float, max_drop: float) -> bool:
    """delta >= -max_drop (fails closed on NaN)."""
    return bool(np.isfinite(delta) and delta >= -max_drop - EPS)


def eligible_rise(cand: float, ref: float, max_rise: float) -> bool:
    """cand <= ref + max_rise (fails closed on NaN)."""
    return bool(np.isfinite(cand) and np.isfinite(ref) and cand <= ref + max_rise + EPS)


def check_aligned(ref: CandData, cd: CandData) -> None:
    """The paired axes need identical rows: refuse otherwise (never silently mis-pair)."""
    if cd.keys != ref.keys:
        raise ValueError(f"{cd.key} and {ref.key} are not row-aligned: {cd.keys} vs {ref.keys}")
    if cd.f1.shape != ref.f1.shape or cd.flips.shape != ref.flips.shape:
        raise ValueError(f"{cd.key} and {ref.key} differ in (seed, fold) / cluster shape")


def axis_a(ref_f1: np.ndarray, cd: CandData, idx: np.ndarray, th: Mapping[str, float],
           level: float) -> dict[str, Any]:  # fmt: skip
    """Axis (a): mean macro-F1 over the matched (seed, fold) pairs; resample the pairs."""
    d = (cd.f1 - ref_f1).reshape(-1)
    rec = safe_gain(float(d.mean()), d[idx].mean(axis=1), level)
    pt = float(d.mean())
    return {
        "candidate": float(cd.f1.mean()), "v1": float(ref_f1.mean()), "delta": pt,
        "improvement": pt, "lo": rec["lo"], "hi": rec["hi"], "ci_of": "delta",
        "ci_excludes_zero_positive": rec["ci_excludes_zero_positive"],
        "improved": bool(rec["ci_excludes_zero_positive"] and pt > 0),
        "eligible": eligible_drop(pt, th["a_max_drop"]), "n_resamples": rec["n_resamples"],
        "rule": f"delta >= -{th['a_max_drop']}",
    }  # fmt: skip


def compute_axes(
    data: Mapping[str, CandData], th: Mapping[str, float], n_boot: int, seed: int, level: float,
    n_bins: int, ref: str = "v1", ref_f1_alt: np.ndarray | None = None,
) -> dict[str, Any]:  # fmt: skip
    """All five axes of every non-ref candidate vs `ref` (v1), plus ref's absolute values.

    Resamples are drawn once per axis from default_rng(seed) and shared by every candidate and
    seed (paired). ref_f1_alt: v1's per-(seed, fold) F1 at its own argmax epoch, for the axis-(a)
    sensitivity block.
    """
    r = data[ref]
    for cd in data.values():
        check_aligned(r, cd)
    n_pairs = r.f1.size
    idx_a = p4.boot_indices(n_pairs, n_boot, seed)
    idx_c = p4.boot_indices(len(r.inst), n_boot, seed)
    idx_d = p4.boot_indices(r.kept.shape[0], n_boot, seed)
    idx_e = p4.boot_indices(len(r.gold), n_boot, seed)
    cls_seeds = class_seeds(len(r.tb_units), seed)

    def stats(cd: CandData) -> dict[str, Any]:
        cal = [fit_calibration(cd.probs[s], cd.gold, n_bins) for s in range(cd.probs.shape[0])]
        return {
            "b": b_stats(cd, n_boot, cls_seeds, level),
            "c": (flip_rate_point(cd.flips, cd.inst), flip_rate_samples(cd.flips, cd.inst, idx_c)),
            "d": (agreement_point(cd.agree, cd.kept), agreement_samples(cd.agree, cd.kept, idx_d)),
            "e": (float(np.mean([c["ece"] for c in cal])),
                  np.mean(
                [ece_samples(c["conf"], c["correct"], idx_e, n_bins) for c in cal], axis=0
            )),
            "T": [c["T"] for c in cal],
        }  # fmt: skip

    st = {k: stats(cd) for k, cd in data.items()}
    out: dict[str, Any] = {"v1": {
        "f1": float(r.f1.mean()), "auroc": st[ref]["b"]["auroc"][0],
        "rej95": st[ref]["b"]["rej95"][0],
        "flip_rate": st[ref]["c"][0], "agreement": st[ref]["d"][0], "ece": st[ref]["e"][0],
        "temperature_per_seed": st[ref]["T"], "e_star": r.e_star,
    }, "candidates": {}}  # fmt: skip
    for key, cd in data.items():
        if key == ref:
            continue
        ax: dict[str, Any] = {"a": axis_a(r.f1, cd, idx_a, th, level)}
        b_au = paired_record(st[key]["b"]["auroc"], st[ref]["b"]["auroc"], True, level)
        b_rj = paired_record(st[key]["b"]["rej95"], st[ref]["b"]["rej95"], True, level)
        ax["b"] = {
            "auroc": b_au, "rej95": b_rj,
            "eligible": eligible_drop(b_au["delta"], th["b_max_auroc_drop"])
            and eligible_drop(b_rj["delta"], th["b_max_rej95_drop"]),
            "improved": bool(b_au["improved"] or b_rj["improved"]),
            "rule": f"dAUROC >= -{th['b_max_auroc_drop']} and dRej95 >= -{th['b_max_rej95_drop']}",
        }  # fmt: skip
        for name, rise in (("c", th["c_max_flip_increase"]), ("e", th["e_max_ece_increase"])):
            rec = paired_record(st[key][name], st[ref][name], False, level)
            rec |= {"eligible": eligible_rise(rec["candidate"], rec["v1"], rise),
                    "rule": f"candidate <= v1 + {rise}"}  # fmt: skip
            ax[name] = rec
        rec = paired_record(st[key]["d"], st[ref]["d"], True, level)
        rec |= {"eligible": eligible_drop(rec["delta"], th["d_max_agreement_drop"]),
                "rule": f"delta >= -{th['d_max_agreement_drop']}"}  # fmt: skip
        ax["d"] = rec
        improved = [a for a in AXES if ax[a]["improved"]]
        cand_out: dict[str, Any] = {
            "e_star": cd.e_star, "n_factors": cd.n_factors, "axes": ax,
            "eligible": all(ax[a]["eligible"] for a in AXES),
            "ineligible_axes": [a for a in AXES if not ax[a]["eligible"]],
            "improved_axes": improved, "n_improved": len(improved),
            "temperature_per_seed": st[key]["T"],
        }  # fmt: skip
        if ref_f1_alt is not None:
            cand_out["sensitivity_a_v1_at_own_argmax"] = axis_a(ref_f1_alt, cd, idx_a, th, level)
        out["candidates"][key] = cand_out
    return out


# ================================================================================ selection rule
@dataclass(frozen=True)
class SelRow:
    """One candidate as the selection rule sees it."""

    key: str
    eligible: bool
    improved: tuple[str, ...]  # improved axes (axis b counted once)
    n_factors: int
    axis_a: float  # axis-(a) point estimate (CV macro-F1 delta)


def select_candidate(rows: Sequence[SelRow], ref: str = "v1") -> dict[str, Any]:
    """Pre-registered "Selection" rule: among eligible non-ref candidates with >= 1 improved axis,
    the most improved axes; ties -> fewer factors; remaining ties -> higher axis-(a) point estimate.
    Nobody qualifies -> ref. Pure; `path` records every narrowing step."""
    pool = [r for r in rows if r.key != ref and r.eligible and r.improved]
    path: list[dict[str, Any]] = [{"step": "eligible_with_>=1_improved_axis",
                                   "kept": [r.key for r in pool]}]  # fmt: skip
    if not pool:
        return {"chosen": ref, "path": path, "unresolved_tie": False,
                "reason": "no eligible candidate with an improved axis; v1 ships"}  # fmt: skip
    for step, key_fn, best in (
        ("most_improved_axes", lambda r: len(r.improved), max),
        ("fewer_factors", lambda r: r.n_factors, min),
        ("higher_axis_a", lambda r: r.axis_a, max),
    ):
        target = best(key_fn(r) for r in pool)
        pool = [r for r in pool if key_fn(r) == target]
        path.append({"step": step, "value": target, "kept": [r.key for r in pool]})
    tie = len(pool) > 1
    return {"chosen": pool[0].key, "path": path, "unresolved_tie": tie,
            "reason": "unresolved exact tie; first in candidate order" if tie else "selected",
            }  # fmt: skip


def rows_from_axes(axes: Mapping[str, Any], v1_alt: bool = False) -> list[SelRow]:
    """SelRows from a saved axes.json; v1_alt uses axis (a) vs v1 at its own argmax epoch."""
    rows = []
    for key, c in axes["candidates"].items():
        a = c["sensitivity_a_v1_at_own_argmax"] if v1_alt else c["axes"]["a"]
        elig = all(c["axes"][x]["eligible"] for x in AXES if x != "a") and a["eligible"]
        imp = [x for x in AXES if x != "a" and c["axes"][x]["improved"]] + (
            ["a"] if a["improved"] else []
        )
        rows.append(SelRow(key, bool(elig), tuple(sorted(imp)), int(c["n_factors"]),
                           float(c["axes"]["a"]["delta"])))  # fmt: skip
    return rows


def stage_select(env: p4.Env) -> None:
    """selection.json from axes.json only (the pure rule + the v1-epoch sensitivity run)."""
    axes = p4.read_json(env.P.results / "axes.json")
    if axes is None:
        raise SystemExit("run --stage analyze first")
    main = select_candidate(rows_from_axes(axes))
    alt = None
    if all("sensitivity_a_v1_at_own_argmax" in c for c in axes["candidates"].values()):
        alt = select_candidate(rows_from_axes(axes, v1_alt=True))
    rows = rows_from_axes(axes)
    out = {
        "chosen": main["chosen"], "path": main["path"], "unresolved_tie": main["unresolved_tie"],
        "reason": main["reason"],
        "eligible": [r.key for r in rows if r.eligible],
        "ineligible": {k: c["ineligible_axes"] for k, c in axes["candidates"].items()
                       if not c["eligible"]},  # fmt: skip
        "improved_axes": {r.key: list(r.improved) for r in rows},
        "n_improved_axes": {r.key: len(r.improved) for r in rows},
        "sensitivity_v1_at_own_argmax": None if alt is None else {
            "chosen": alt["chosen"], "path": alt["path"],
            "same_outcome": alt["chosen"] == main["chosen"],
            "flagged": alt["chosen"] != main["chosen"],
            "v1_own_argmax_epoch": axes["v1_epoch"]["own_argmax"],
            "v1_deployed_epoch": axes["v1_epoch"]["e_star"],
        },
        "axes": axes, "git_sha": git_sha(), "smoke": env.smoke,
    }  # fmt: skip
    write_json(env.P.results / "selection.json", out)
    print(f"[select] chosen: {out['chosen']} (eligible {out['eligible']}, "
          f"improved axes {out['n_improved_axes']})")  # fmt: skip
    if out["sensitivity_v1_at_own_argmax"]:
        s = out["sensitivity_v1_at_own_argmax"]
        print(
            f"[select] v1-epoch sensitivity: same_outcome={s['same_outcome']} "
            f"flagged={s['flagged']}"
        )


# ===================================================================================== analysis
def load_cand_data(env: p4.Env, key: str) -> CandData:
    """Read one candidate's saved curves, OOF csvs and Track B score tables (no model, no GPU)."""
    rec = recipe_for(key)
    e = e_star_of(env, key)
    seeds = model_seeds(env)
    f1 = np.array([[json.loads(curve_path(env, key, s, f).read_text(encoding="utf-8"))["macro_f1"][
                        e - 1]
                    for f in env.folds] for s in seeds])  # fmt: skip
    langs = list(env.pcfg["translation"]["langs"])
    oof = {s: {n: pd.read_csv(seed_results(env, key, s) / f"oof_{n}.csv") for n in OOF_NAMES}
           for s in seeds}  # fmt: skip
    # --- calibration inputs
    pcols = sorted((c for c in oof[seeds[0]]["clean"].columns if c.startswith("prob_")),
                   key=lambda c: int(c.split("_")[1]))  # fmt: skip
    clean0 = oof[seeds[0]]["clean"].sort_values("id").reset_index(drop=True)
    probs, gold = [], clean0["gold"].to_numpy()
    for s in seeds:
        c = oof[s]["clean"].sort_values("id").reset_index(drop=True)
        if not (c["id"].equals(clean0["id"]) and (c["gold"].to_numpy() == gold).all()):
            raise ValueError(f"{key}: OOF clean rows differ between seeds")
        probs.append(c[pcols].to_numpy(dtype=np.float64))
    # --- swap flips per id cluster
    clusters = sorted(set(oof[seeds[0]]["swap"]["id"]))
    flips, inst = [], None
    for s in seeds:
        sw = oof[s]["swap"].assign(flip=lambda d: (d["swap_pred"] != d["clean_pred"]).astype(float))
        g = sw.groupby("id").agg(flips=("flip", "sum"), n=("flip", "size")).reindex(clusters)
        if sorted(set(sw["id"])) != clusters:
            raise ValueError(f"{key}: swap id clusters differ between seeds")
        n = g["n"].to_numpy(dtype=float)
        if inst is not None and not np.array_equal(inst, n):
            raise ValueError(f"{key}: swap instance counts differ between seeds")
        inst = n
        flips.append(g["flips"].to_numpy(dtype=float))
    # --- translation agreement (kept rows only enter via the mask)
    mt_ids = sorted(set(oof[seeds[0]]["mt"]["id"]))
    agree, kept = [], None
    for s in seeds:
        mt = p4.mt_with_bool(oof[s]["mt"])
        a = np.zeros((len(mt_ids), len(langs)))
        k = np.zeros((len(mt_ids), len(langs)))
        pos = {i: n for n, i in enumerate(mt_ids)}
        for r in mt.itertuples(index=False):
            a[pos[r.id], langs.index(r.lang)] = float(r.mt_pred == r.en_pred)
            k[pos[r.id], langs.index(r.lang)] = float(r.kept)
        if kept is not None and not np.array_equal(kept, k):
            raise ValueError(f"{key}: LaBSE-kept mask differs between seeds")
        kept = k
        agree.append(a)
    # --- Track B DEV units [class][seed]
    method = env.pcfg["trackb"]["score"]
    ret = float(env.pcfg["trackb"]["retention"])
    safe = env.ctx.cfg["ood"]["safe_labels"]
    units: list[list[ti.BootUnit]] = [[] for _ in env.dev_classes]
    tb_fp: list[Any] = [None] * len(env.dev_classes)
    for s in seeds:
        es = seed_env(env, key, s)
        cand = p4.candidate_for(es, key, e)
        for ci, cls in enumerate(env.dev_classes):
            table = ti.read_scores(es.ctx, ti.run_id_for(cand, "loco", s, cls))
            fp = table[["id", "set"]].astype(str).agg("|".join, axis=1).tolist()
            if tb_fp[ci] is not None and tb_fp[ci] != fp:
                raise ValueError(f"{key}: Track B rows of {cls} differ between seeds")
            tb_fp[ci] = fp
            units[ci].append(ti.boot_unit(table, method, [ret], safe))
    keys = {"clean": tuple(clean0["id"]), "swap": tuple(clusters), "mt": tuple(mt_ids),
            "tb": tuple(hash_ids(x) for x in tb_fp)}  # fmt: skip
    return CandData(key, len(rec.factors), e, f1, units, np.array(flips), inst, np.array(agree),
                    kept, np.array(probs), gold, keys)  # fmt: skip


def hash_ids(ids: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def render_axes_table(axes: Mapping[str, Any]) -> str:
    """Markdown summary of every candidate x axis (numbers only)."""

    def f(v: Any, nd: int = 4) -> str:
        return "-" if v is None else f"{v:+.{nd}f}"

    def cell(a: Mapping[str, Any], nd: int = 4, elig: bool | None = None) -> str:
        elig = a["eligible"] if elig is None else elig  # axis (b): eligibility is joint
        mark = ("ok" if elig else "INELIGIBLE") + (", improved" if a["improved"] else "")
        return f"{f(a['delta'], nd)} [{f(a['lo'], nd)}, {f(a['hi'], nd)}] {mark}"

    v1 = axes["v1"]
    lines = [
        "## Multi-axis selection axes: candidate - v1 "
        "(CI of the improvement; reductions for flip rate / ECE)",
        "",
        f"v1 (e*={v1['e_star']}): F1 {v1['f1']:.4f}, AUROC {v1['auroc']:.4f}, rej@95 "
        f"{v1['rej95']:.4f}, flip {v1['flip_rate']:.4f}, agreement {v1['agreement']:.4f}, "
        f"ECE {v1['ece']:.4f}",
        "",
        "| candidate | e* | (a) CV F1 | (b) dAUROC | (b) dRej@95 | (c) flip | (d) agreement "
        "| (e) ECE | eligible | improved axes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in axes["candidates"].items():
        a = c["axes"]
        be = a["b"]["eligible"]
        cells = [
            cell(a["a"]), cell(a["b"]["auroc"], 4, be), cell(a["b"]["rej95"], 4, be),
            cell(a["c"]), cell(a["d"]), cell(a["e"]),
        ]  # fmt: skip
        lines.append(
            f"| {key} | {c['e_star']} | {' | '.join(cells)} | {c['eligible']} | "
            f"{','.join(c['improved_axes']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def stage_analyze(env: p4.Env) -> None:
    """Five axes of every candidate vs v1 from the saved outputs -> axes.json + axes_table.md."""
    pc = env.pcfg
    keys = list(pc["candidates"])
    data = {k: load_cand_data(env, k) for k in keys}
    bs = pc["bootstrap"]
    v1_ep = p4.read_json(epoch_path(env, "v1")) or {}
    alt = None
    if v1_ep and v1_ep["own_argmax"] != v1_ep["e_star"]:
        alt = np.array([
            [json.loads(curve_path(env, "v1", s, f).read_text(encoding="utf-8"))["macro_f1"][
                v1_ep["own_argmax"] - 1] for f in env.folds] for s in model_seeds(env)
        ])  # fmt: skip
    elif v1_ep:
        alt = data["v1"].f1
    axes = compute_axes(data, pc["thresholds"], env.n_boot, int(bs["seed"]), float(bs["level"]),
                        int(pc["calibration"]["ece_bins"]), "v1", alt)  # fmt: skip
    axes |= {
        "v1_epoch": {k: v1_ep.get(k) for k in ("e_star", "own_argmax", "f1_at_e_star",
                                               "f1_at_own_argmax")},
        "bootstrap": bs | {"n_resamples_used": env.n_boot}, "seeds": model_seeds(env),
        "thresholds": pc["thresholds"], "git_sha": git_sha(), "smoke": env.smoke,
        "label": "measured; paired bootstrap, seeds averaged inside each resample",
    }  # fmt: skip
    write_json(env.P.results / "axes.json", axes)
    (env.P.results / "axes_table.md").write_text(render_axes_table(axes), encoding="utf-8")
    print(f"[analyze] wrote {env.P.results / 'axes.json'}")


# ========================================================================================= main
def run_stage(env: p4.Env, stage: str, cands: Sequence[str]) -> None:
    if stage == "all":
        for s in ("curves", "epoch", "folds", "trackb", "analyze", "select"):
            run_stage(env, s, cands)
        return
    if stage in FINAL_STAGES:
        from intent_router import phase4e_final as pf  # lazy: it imports this module

        {"confirm_final": pf.stage_confirm_final, "final_retrain": pf.stage_final_retrain}[stage](
            env
        )
        return
    if stage == "analyze":
        stage_analyze(env)
        return
    if stage == "select":
        stage_select(env)
        return
    for c in cands:
        rec = recipe_for(c)
        if stage == "reuse_check":
            dry_run_reuse(env, rec)
        elif stage == "curves":
            run_curves(env, rec)
        elif stage == "epoch":
            stage_epoch(env, rec)
        elif stage == "folds":
            stage_folds(env, rec)
        elif stage == "trackb":
            stage_trackb(env, rec)


def dry_run_reuse(env: p4.Env, rec: p4.Recipe) -> None:
    """Read-only: print the curve-reuse comparison and the OOF-reuse gates (no training/writes)."""
    if rec.key not in env.pcfg["reuse_4c"]["keys"]:
        print(f"[reuse] {rec.key}: no 4c counterpart; trained in full")
        return
    print_reuse_report(check_curve_reuse(env, rec))
    ep = p4.read_json(epoch_path(env, rec.key))
    key4c = env.pcfg["reuse_4c"]["keys"][rec.key]
    dep = p4.read_json(Path(env.pcfg["reuse_4c"]["results_dir"]) / key4c / "folds.json") or {}
    e = None if ep is None else int(ep["e_star"])
    if rec.key == "v1" and e is None:
        e = int(env.pcfg["v1"]["deployed_epoch"])
    hypothetical = e is None  # no 4e e* yet: assume it equals 4c's, to exercise the other gates
    if hypothetical:
        e = int(dep["deployed_epoch"])
    rep = check_oof_reuse(env, rec, int(env.pcfg["reuse_4c"]["seed"]), e, dry_run=True)
    print(
        f"[reuse] {rec.key} OOF: 4c deployed epoch {dep.get('deployed_epoch')}, 4e e* "
        f"{'pending (HYPOTHETICAL: assumed equal to 4c)' if hypothetical else e}; "
        f"ok={rep['ok']} {rep.get('checks')}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Multi-axis selection")
    ap.add_argument("--config", default="configs/phase4e.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--cand", default=None, choices=CANDIDATES,
                    help="one candidate (default: all four, in config order)")  # fmt: skip
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: fold 0, 1st DEV class, 1 epoch, stand-in inputs; "
                    "writes only under outputs/phase4e_smoke")  # fmt: skip
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    env = make_env(args)
    cands = [args.cand] if args.cand else list(env.pcfg["candidates"])
    run_stage(env, args.stage, cands)


if __name__ == "__main__":
    main()
