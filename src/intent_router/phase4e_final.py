"""Multi-axis selection post-selection stages (run only when the selection is not v1).

    python -m intent_router.phase4e --stage confirm_final [--smoke] [--no-wandb]
    python -m intent_router.phase4e --stage final_retrain [--smoke] [--no-wandb]

confirm_final: Track B CONFIRM (5 classes, seed 42) + headline (seeds 42/43/44) for the selected
candidate stopped at e*, scored with maha_ft, paired bootstrap CIs against v1's existing Open-set
scorer comparison
runs (results/phase4e/confirm_<cand>.json, same shape as phase4c's confirm_comb.json).

final_retrain: archive v1 (copy), derive the final config, train on the train split only, run THE
ONE logged test evaluation under the new model fingerprint, refit the maha_ft threshold on val,
regenerate the analysis artifacts (analysis in a subprocess), side-by-side numbers and robustness.

The candidate and its e* are read from results/phase4e/selection.json and <cand>/epoch.json (never
hard-coded). Building blocks come from phase4c.py (recipes, extras, patched training, one Track B
run, the OOD threshold); phase4c's own v2 stage functions hard-code its `final_v2` config block and
are left untouched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import final as final_mod
from intent_router import phase4c as p4
from intent_router import phase4e as p5
from intent_router import robustness as rb
from intent_router import trackb_improve as ti
from intent_router.evaluate import git_sha

CONFIRM_CALL_TYPE = "phase4e_confirm"
V1_VERSION = "v1"


# ============================================================================== selection / guard
def resolve_selection(results_dir: Path) -> tuple[p4.Recipe, int, dict[str, Any]]:
    """(recipe, e*, selection.json) of the selected candidate; refuses anything but a real non-v1.

    Reads results_dir/selection.json and results_dir/<chosen>/epoch.json. Exits when the selection
    is missing, comes from a smoke run, chose v1 (nothing to confirm or retrain: v1 ships) or the
    chosen candidate has no (non-smoke) epoch.json.
    """
    sel = p4.read_json(results_dir / "selection.json")
    if sel is None:
        raise SystemExit(f"no {results_dir / 'selection.json'}: run phase4e --stage select first")
    if sel.get("smoke"):
        raise SystemExit("selection.json was written by a smoke run; refusing to use it")
    chosen = str(sel["chosen"])
    if chosen == V1_VERSION:
        raise SystemExit("selection chose v1: nothing to confirm or retrain, v1 ships unchanged")
    rec = p5.recipe_for(chosen)
    ep = p4.read_json(results_dir / chosen / "epoch.json")
    if ep is None or ep.get("smoke"):
        raise SystemExit(f"no real (non-smoke) {results_dir / chosen / 'epoch.json'}")
    return rec, int(ep["e_star"]), sel


# ======================================================================================= confirm
def confirm_env(env: p4.Env, rec: p4.Recipe) -> p4.Env:
    """Env whose Track B score tables / run jsons / npz / test log live under the confirm dirs."""
    cc = env.pcfg["confirm_final"]
    results = env.P.results / cc["results_subdir"]
    outputs = env.P.outputs / cc["outputs_subdir"]
    paths = ti.Paths(results, outputs / "trackb_arrays", env.P.results / cc["test_log"], env.smoke)
    return p4.Env(
        env.pcfg, p4.P4Paths(results, outputs, env.smoke), replace(env.ctx, P=paths), env.smoke,
        None, env.inputs if rec.factors else None,
    )  # fmt: skip


def v1_comparator_ctx(env: p4.Env) -> ti.Ctx:
    """
    Context reading v1's Open-set scorer comparison base score tables (CONFIRM classes s42, headline
    s42/43/44).
    """
    paths = ti.Paths(
        Path(env.pcfg["confirm_final"]["v1_trackb_dir"]), env.P.outputs / "unused",
        env.P.results / "unused.jsonl", env.smoke,
    )  # fmt: skip
    return replace(env.ctx, P=paths)


def missing_v1_tables(ctx: ti.Ctx) -> list[str]:
    """v1 comparator score tables that are absent (CONFIRM classes at loco_seed, headline seeds)."""
    base = ti.base_candidate(ctx)
    ids = [ti.run_id_for(base, "loco", int(ctx.cfg["loco_seed"]), c)
           for c in ctx.cfg["confirm_classes"]]  # fmt: skip
    ids += [ti.run_id_for(base, "headline", int(s)) for s in ctx.cfg["headline_seeds"]]
    return [i for i in ids if not (ctx.P.scores / f"{i}.csv").exists()]


def stage_confirm_final(env: p4.Env) -> None:
    """CONFIRM + headline runs of the selected candidate, then paired CIs vs v1's existing runs."""
    rec, e_star, _ = resolve_selection(Path(env.pcfg["results_dir"]))
    cenv = confirm_env(env, rec)
    cfg = cenv.ctx.cfg
    cur_ctx = v1_comparator_ctx(env)
    if not env.smoke:
        missing = missing_v1_tables(cur_ctx)
        if missing:  # fail before any GPU time or test-split inference is spent
            raise SystemExit(f"v1 comparator score tables missing in {cur_ctx.P.scores}: {missing}")
    cand = p4.candidate_for(cenv, rec.key, e_star)
    runs = ti.plan(
        cenv.ctx, cand, cfg["confirm_classes"] if not env.smoke else cenv.dev_classes,
        cfg["headline_seeds"] if not env.smoke else [], [int(cfg["loco_seed"])], CONFIRM_CALL_TYPE,
    )  # fmt: skip
    for r in runs:  # LOCO + headline runs also make logged test-split inferences (call_type above)
        p4.run_one_p4c(cenv, rec, r)
    if env.smoke:
        print("[confirm_final] smoke: runs only (no CONFIRM tables)")
        return
    method = str(env.pcfg["trackb"]["score"])
    e_cur = ti.confirm_entry(cur_ctx, ti.base_candidate(cur_ctx), method)
    e_new = ti.confirm_entry(cenv.ctx, cand, method)
    level = float(env.pcfg["bootstrap"]["level"])
    deltas = {part: ti.paired_delta_ci(e_new[part]["_samples"], e_cur[part]["_samples"], level)
              for part in ("confirm", "headline")}  # fmt: skip
    for e in (e_cur, e_new):
        for part in ("confirm", "headline"):
            e[part].pop("_samples")
    out = Path(env.pcfg["results_dir"]) / f"confirm_{rec.key}.json"
    p5.write_json(out, {
        "label": "measured, CONFIRM LOCO classes (5, seed 42) and headline holdout (seeds "
        f"{cfg['headline_seeds']}); thresholds from calibration-known rows",
        "candidate": {"key": rec.key, "factors": list(rec.factors), "e_star": e_star,
                      "score": method, **e_new},
        "v1": {"score": method, **e_cur},
        "paired_delta_candidate_minus_v1": deltas,
        "ci_note": "AUPR has a point estimate only; resampled: AUROC, FPR@95TPR, recall, retention",
        "bootstrap": {"n_resamples": int(cfg["bootstrap"]["n_resamples"]),
                      "seed": int(cfg["bootstrap"]["seed"]), "level": level},
        "v1_comparator": str(cur_ctx.P.results), "test_log": str(cenv.ctx.P.test_log),
        "git_sha": git_sha(),
    })  # fmt: skip
    print(f"[confirm_final] wrote {out}")


# ================================================================================ final config
def derive_final_cfg(
    fcfg: Mapping[str, Any], rec: p4.Recipe, e_star: int, pcfg: Mapping[str, Any], smoke: bool
) -> dict[str, Any]:
    """final.yaml with the selected recipe at e* and the v3 model dir / CV stand-in paths.

    results_dir, figures_dir, outputs_dir and the shared test_eval_log are deliberately unchanged:
    the new artifacts replace results/final (v1 is archived first) and the append-only log stays.
    """
    cfg = json.loads(json.dumps(fcfg))  # deep copy of plain yaml data
    v3 = pcfg["final_v3"]
    cfg["train"] = cfg["train"] | {"stop_epoch": e_star, **p4.recipe_overrides(rec, pcfg)}
    cfg["model_dir"] = v3["model_dir"]
    cfg["determinism"]["sdpa_timing_run"] = bool(v3["sdpa_timing_run"])
    cfg["wandb"]["group"] = f"phase4e-final-{v3['version']}"
    if not smoke:
        d = Path(v3["results_dir"])
        cfg["selection_path"] = (d / f"selection_{v3['version']}.json").as_posix()
        cfg["analysis"]["oof_dir"] = (d / "oof").as_posix()
        cfg["analysis"]["oof_config_id"] = f"p4e_{rec.key}"
        cfg["analysis"]["oof_predictions_per_id"] = int(v3["oof_predictions_per_id"])
    return cfg


def build_stand_in_oof(frames: Sequence[pd.DataFrame], n_per_id: int) -> pd.DataFrame:
    """Concatenate per-seed OOF clean predictions into the analysis-module layout.

    Every id must appear exactly n_per_id times with one gold label, and every frame must carry
    the same columns (analysis.load_oof_mean raises otherwise; failing here is earlier and clearer).
    """
    cols = list(frames[0].columns)
    if any(list(f.columns) != cols for f in frames):
        raise ValueError("OOF frames have different columns")
    out = pd.concat(frames, ignore_index=True)
    counts = out.groupby("id").size()
    if not (counts == n_per_id).all():
        raise ValueError(f"{int((counts != n_per_id).sum())} ids without {n_per_id} predictions")
    if (out.groupby("id")["gold"].nunique() != 1).any():
        raise ValueError("gold label differs between predictions of one id")
    return out


def selection_stand_in(
    cid: str, e_star: int, f1: np.ndarray, acc: np.ndarray, note: str
) -> dict[str, Any]:
    """The selection.json shape final.stage_evaluate reads: CV macro-F1/accuracy at e* per run."""
    f1, acc = np.asarray(f1, dtype=float), np.asarray(acc, dtype=float)
    return {"candidates": {cid: {
        "chosen_epoch": int(e_star), "n_runs": int(f1.size),
        "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(np.std(f1, ddof=1)),
        "accuracy_mean": float(acc.mean()), "accuracy_std": float(np.std(acc, ddof=1)),
        "note": note,
    }}}  # fmt: skip


def curve_values(env: p4.Env, key: str, e_star: int) -> tuple[np.ndarray, np.ndarray]:
    """(macro-F1, accuracy) at e* of every (seed, fold) 4e curve file; raises if one is missing."""
    f1, acc = [], []
    for seed in p5.model_seeds(env):
        for f in env.folds:
            rec = p4.read_json(p5.curve_path(env, key, seed, f))
            if rec is None:
                raise SystemExit(f"missing curve {p5.curve_path(env, key, seed, f)}")
            f1.append(rec["macro_f1"][e_star - 1])
            acc.append(rec["accuracy"][e_star - 1])
    return np.array(f1), np.array(acc)


def oof_clean_paths(env: p4.Env, key: str) -> list[Path]:
    return [p5.seed_results(env, key, s) / "oof_clean.csv" for s in p5.model_seeds(env)]


def write_cv_stand_ins(env: p4.Env, rec: p4.Recipe, cfg: Mapping[str, Any], e_star: int) -> None:
    """Selection json + OOF csv of the 4e fold models (3 model seeds x fold-seed s0, 5 folds)."""
    cid = cfg["analysis"]["oof_config_id"]
    f1, acc = curve_values(env, rec.key, e_star)
    seeds = p5.model_seeds(env)
    note = (f"Multi-axis selection config {rec.key}: 5 folds x model seeds {seeds}, fold-seed s0 "
            f"({f1.size} fold-runs); v1 used 9 fold-runs per id")  # fmt: skip
    p5.write_json(Path(cfg["selection_path"]), selection_stand_in(cid, e_star, f1, acc, note))
    oof = build_stand_in_oof(
        [pd.read_csv(p) for p in oof_clean_paths(env, rec.key)],
        int(cfg["analysis"]["oof_predictions_per_id"]),
    )
    oof_dir = Path(cfg["analysis"]["oof_dir"])
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_dir / f"{cid}.csv", index=False, float_format="%.9g")


def preflight_stand_ins(env: p4.Env, rec: p4.Recipe, e_star: int) -> None:
    """Everything write_cv_stand_ins reads must exist BEFORE v1 is archived or anything trains."""
    curve_values(env, rec.key, e_star)
    missing = [str(p) for p in oof_clean_paths(env, rec.key) if not p.exists()]
    if missing:
        raise SystemExit(f"4e OOF predictions missing: {missing}")


# ====================================================================================== archive
def _copy_atomic(src: Path, dst: Path, ignore: Any = None) -> None:
    """copytree into <dst>.partial, then rename: an interrupted copy never looks like an archive."""
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists (interrupted copy): inspect it and remove it manually")
    shutil.copytree(src, tmp, ignore=ignore)
    tmp.rename(dst)


def version_of(results: Path) -> str | None:
    mv = p4.read_json(results / "model_version.json")
    return None if mv is None else str(mv.get("version"))


def archive_v1(
    src_results: Path, dst_results: Path, figures_dir: Path, src_outputs: Path, dst_outputs: Path
) -> str:
    """Copy (never move, never overwrite) v1's results, figures and outputs.

    Returns "archived" or "exists". An existing dst_results is left untouched (but must itself
    be v1, else exit); a missing one is only filled when src_results is v1 (else exit): v3 files
    must never be archived under the v1 name. The shared test_eval_log.jsonl is not copied.
    """
    if dst_results.exists():
        if version_of(dst_results) != V1_VERSION:
            raise SystemExit(f"{dst_results} exists but is not v1: refusing to overwrite or reuse")
        print(f"[final] {dst_results} exists and is v1: not touched")
        return "exists"
    if version_of(src_results) != V1_VERSION:
        raise SystemExit(f"{src_results} is not v1 (model_version.json): refusing to archive it")
    if dst_outputs.exists():
        raise SystemExit(f"{dst_outputs} exists but {dst_results} does not: resolve manually")
    _copy_atomic(src_outputs, dst_outputs)
    _copy_atomic(src_results, dst_results, shutil.ignore_patterns("test_eval_log.jsonl"))
    fig_dst = dst_results / "figures"
    fig_dst.mkdir(exist_ok=True)
    for p in figures_dir.glob("final_*.png"):
        shutil.copyfile(p, fig_dst / p.name)
    print(f"[final] archived v1 to {dst_results} and {dst_outputs}")
    return "archived"


def training_done(results: Path, model_dir: Path, v1_fingerprint: str | None) -> bool:
    """True when results/final already holds a finished v3 train stage (resume without retraining).

    A second `evaluation` for the same fingerprint is refused by the test-eval guard, so after a
    crash past the logged evaluation the train stage must not run again.
    """
    ts = p4.read_json(results / "train_summary.json")
    if ts is None or not ts.get("model_fingerprint") or ts["model_fingerprint"] == v1_fingerprint:
        return False
    return (
        Path(str(ts.get("model_dir"))).as_posix() == model_dir.as_posix()
        and (model_dir / "config.json").exists()
    )


# ===================================================================== reporting (saved files only)
def pick(d: Mapping[str, Any]) -> dict[str, Any]:
    """Headline numbers of one final model's track_a.json."""
    t = d["test"]
    return {"model_fingerprint": d["model_fingerprint"], "macro_f1": t["macro_f1"],
            "accuracy": t["accuracy"], "temperature": d["calibration"]["temperature"],
            "ece_test_after": d["calibration"]["test"]["ece_after"],
            "selective_at_threshold": t["selective"]["at_threshold"]}  # fmt: skip


def write_side_by_side(env: p4.Env, version: str) -> None:
    """v1 (archived) vs the new final model, plus v2 for the record: one logged evaluation each."""
    v3 = env.pcfg["final_v3"]
    old = p4.read_json(Path(v3["archive_results"]) / "track_a.json")
    new = p4.read_json(Path(env.fcfg["results_dir"]) / "track_a.json")
    v2 = p4.read_json(Path(v3["v2_results"]) / "track_a.json")
    if old is None or new is None:
        print("[final] side-by-side skipped: track_a.json missing")
        return
    out: dict[str, Any] = {"v1": pick(old), version: pick(new)}
    if v2 is not None:
        out["v2"] = pick(v2)
    out["label"] = "test split, one logged evaluation each (v2 for the record, not shipped)"
    p5.write_json(Path(v3["side_by_side"]), out)
    print(f"[final] v1 test macro-F1 {old['test']['macro_f1']['point']:.4f} -> "
          f"{version} {new['test']['macro_f1']['point']:.4f}")  # fmt: skip


def patch_cv_note(track_a_path: Path, n_per_id: int, seeds: Sequence[int]) -> None:
    """final.stage_evaluate hard-codes "9 OOF predictions per id" in a note; state the truth."""
    ta = json.loads(track_a_path.read_text(encoding="utf-8"))
    ta["cv"]["oof_probability_averaged_426_rows"]["note"] = (
        f"probabilities averaged over the {n_per_id} OOF predictions per id (model seeds "
        f"{list(seeds)} x fold-seed s0), then scored once"
    )
    final_mod.write_json(track_a_path, ta)


def oof_robustness_from_4e(env: p4.Env, keys: Sequence[str]) -> dict[str, Any]:
    """Per-seed OOF swap / noise / translation summaries of the 4e fold models (saved folds.json).

    These use the 4e definitions (3 neutral swaps, char_mixed_10, m2m100 + LaBSE filter), NOT the
    robustness stage definitions, so they are comparable across 4e candidates but not with 4a's OOF.
    """
    out: dict[str, Any] = {
        "note": "from results/phase4e/<cand>/s*/folds.json (4e definitions); the fold models were "
        "deleted after OOF extraction, so rb.stage_* cannot re-run on OOF rows",
    }
    for key in keys:
        per_seed = {}
        for s in p5.model_seeds(env):
            fj = p4.read_json(p5.seed_results(env, key, s) / "folds.json") or {}
            per_seed[str(s)] = {
                k: fj.get(k)
                for k in ("oof_accuracy", "oof_macro_f1", "swap", "noise", "translation")
            }
        out[key] = per_seed
    return out


def robustness_v3(env: p4.Env, final_cfg_path: Path) -> None:
    """ID-swap / noise / translation on the new final model's TEST rows (logged
    `robustness_inference` calls, via robustness.py). The OOF side of rb.stage_* needs fold models
    that 4e deleted, so it is skipped (sources: [test]) and covered by oof_robustness_from_4e."""
    v3 = env.pcfg["final_v3"]
    rcfg = yaml.safe_load(Path(env.pcfg["robustness_config"]).read_text(encoding="utf-8"))
    out = env.P.outputs / "final_v3"
    res_dir = Path(v3["robustness_results"])
    rcfg |= {"final_config": str(final_cfg_path), "results_dir": str(res_dir),
             "figures_dir": str(res_dir / "figures"), "outputs_dir": str(out),
             "sources": ["test"]}  # fmt: skip
    rcfg["translation"] = rcfg["translation"] | {"include_oof": False}
    cache = out / "translations.csv"
    src = Path("outputs/robustness/translations.csv")  # same NLLB greedy translations as 4a
    if not cache.exists() and src.exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cache)
    r_path = Path(v3["results_dir"]) / "robustness_v3.yaml"
    r_path.write_text(yaml.safe_dump(rcfg, sort_keys=False), encoding="utf-8")
    rcfg, fcfg = rb.load_configs(str(r_path))
    P = rb.Paths(rcfg, False)
    rb.stage_idswap(rcfg, fcfg, P)
    rb.stage_noise(rcfg, fcfg, P)
    rb.stage_translate(rcfg, fcfg, P)
    rb.stage_report(rcfg, P, False)


# ================================================================================= final_retrain
def stage_final_retrain(env: p4.Env) -> None:
    """Retrain with the selected recipe, ONE logged test evaluation, redo the analysis artifacts."""
    real = Path(env.pcfg["results_dir"])
    rec, e_star, sel = resolve_selection(real)
    v3 = env.pcfg["final_v3"]
    version = str(v3["version"])
    if not env.smoke and not (real / f"confirm_{rec.key}.json").exists():
        raise SystemExit("run --stage confirm_final first (CONFIRM + headline vs v1)")
    cfg = derive_final_cfg(env.fcfg, rec, e_star, env.pcfg, env.smoke)
    inp = env.inputs if rec.factors else None
    method = str(env.pcfg["trackb"]["score"])
    v1_fp: str | None = None
    if env.smoke:
        root = env.P.results.parent / "final_v3"
        P = final_mod.Paths(root / "results", root / "figures", root / "outputs", root / "model",
                            root / "test_eval_log.jsonl", True)  # fmt: skip
        use_wandb = False
    else:
        preflight_stand_ins(env, rec, e_star)
        archive_v1(Path(env.fcfg["results_dir"]), Path(v3["archive_results"]),
                   Path(env.fcfg["figures_dir"]), Path(env.fcfg["outputs_dir"]),
                   Path(v3["archive_outputs"]))  # fmt: skip
        v1_fp = (p4.read_json(Path(v3["archive_results"]) / "model_version.json") or {}).get(
            "model_fingerprint"
        )
        write_cv_stand_ins(env, rec, cfg, e_star)
        P = final_mod.make_paths(cfg, False)
        use_wandb = bool(cfg["wandb"]["enabled"]) and bool(env.ctx.use_wandb)
        d = Path(v3["results_dir"])
        d.mkdir(parents=True, exist_ok=True)
        (d / f"final_{version}_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    record: dict[str, Any] = {}

    def build(tr: pd.DataFrame, _ev: pd.DataFrame | None) -> p4.Extras:
        # No holdout in the 12-class run: the synthetic pool is unfiltered; A3 = train ids only.
        return p4.build_extras(rec, inp, env.pcfg, tr, set())

    if not env.smoke and training_done(P.results, P.model_dir, v1_fp):
        print("[final] train stage already finished for this model: not retraining")
        train_df = data_mod.get_frame(["train"], cfg["data_path"], cfg["splits_path"])
        record.update(build(train_df.sort_values("id").reset_index(drop=True), None).info)
    else:
        with p4.patched_training([final_mod], build, record):
            final_mod.stage_train(cfg, P, env.smoke)
    final_mod.stage_evaluate(cfg, P, env.smoke, use_wandb)
    tsum = p4.read_json(P.results / "train_summary.json") or {}
    ood = p4.shipped_ood_threshold(
        cfg, P.outputs / "features_logits.npz", method, float(env.ctx.tb["ood"]["retention"]),
        env.ctx.cfg["c1"]["fusion_components"],
    )  # fmt: skip
    p5.write_json(P.results / "ood_shipped.json", ood)
    flag = sel.get("sensitivity_v1_at_own_argmax")
    p5.write_json(P.results / "model_version.json", {
        "version": version, "factors": list(rec.factors), "deployed_epoch": e_star,
        "train_config": cfg["train"], "extras": record,
        "model_fingerprint": tsum.get("model_fingerprint"), "model_dir": str(P.model_dir),
        "shipped_ood_score": method,
        "archived_v1": v3["archive_results"], "archived_v2": v3["v2_results"],
        "selection": (real / "selection.json").as_posix(),
        "selection_sensitivity_flagged": None if flag is None else bool(flag["flagged"]),
        "selection_note": "selected by the pre-registered rule with v1 at its shipped epoch 9; "
        "against v1 at its own argmax epoch the margin on axis (a) is outside -0.010 (see "
        "selection.json sensitivity_v1_at_own_argmax); the decision is not robust to that choice",
        "git_sha": git_sha(), "smoke": env.smoke,
    })  # fmt: skip
    if env.smoke:
        print("[final] smoke: trained + evaluated on val stand-in; analysis/robustness skipped")
        return
    cfg_path = Path(v3["results_dir"]) / f"final_{version}_config.yaml"
    subprocess.run(  # noqa: S603 - analysis refuses to run in a process that loaded ood/baselines
        [sys.executable, "-m", "intent_router.analysis", "--config", str(cfg_path)],
        check=True, env={**os.environ, "PYTHONPATH": "src"},
    )  # fmt: skip
    patch_cv_note(P.results / "track_a.json", int(cfg["analysis"]["oof_predictions_per_id"]),
                  p5.model_seeds(env))  # fmt: skip
    write_side_by_side(env, version)
    p5.write_json(Path(v3["results_dir"]) / "oof_robustness_4e.json",
                  oof_robustness_from_4e(env, ["v1", rec.key]))  # fmt: skip
    robustness_v3(env, cfg_path)
