"""Open-set improvement round post-selection stages: confirm_report and final_retrain.

    python -m intent_router.phase6a --stage confirm_report [--smoke]
    python -m intent_router.phase6a --stage final_retrain  [--smoke [--smoke-cand <key>]]

confirm_report: ONLY after `select`. CONFIRM (5 classes, seed 42) + headline holdout (seeds
42/43/44) for the selected candidate and for ref, raw AND ID-neutral, paired bootstrap CIs
(candidate - ref), plus the adaptive-reuse look counts tallied by code from the artifacts of the
earlier phases. If ref is selected its own numbers and the counts are still reported. These runs
read test rows (known side) and are logged under call_type phase6a_confirm in
results/phase6a/confirm_test_inference_log.jsonl.

final_retrain: refuses unless selection.json (a real, non-smoke selection) chose a non-ref
candidate and confirm_report.json exists. Deterministic training on the train split at the
candidate's e* (soup members: member seeds 0..4 = replicate 0, trained on the full train split,
greedy criterion = val macro-F1), ONE logged test evaluation under the new fingerprint (final.py's
guarded door), temperature and the OOD threshold refit on val, the previous results/final archived
(copy, never move) to results/final_v1_pre6a, analysis artifacts regenerated. Implemented here;
NOT run in the build session.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import data as data_mod
from intent_router import final as final_mod
from intent_router import ood_variants as ov
from intent_router import phase4c as p4
from intent_router import phase4e as p5
from intent_router import phase4e_final as pf
from intent_router import phase6a as p6
from intent_router import phase6a_select as sel
from intent_router import soup
from intent_router import trackb_improve as ti
from intent_router.evaluate import git_sha, softmax, threshold_at_retention
from intent_router.models import predict
from intent_router.phase6a_views import local_gold, view_dir
from intent_router.stats import accuracy, macro_f1
from intent_router.train import FoldResult

CONFIRM_CALL_TYPE = "phase6a_confirm"
REF = p6.REF


# ============================================================================ selection guard
def read_selection(env: p4.Env) -> dict[str, Any]:
    """selection.json of this run's results dir; real runs refuse a smoke selection."""
    s = p4.read_json(env.P.results / "selection.json")
    if s is None:
        raise SystemExit(f"no {env.P.results / 'selection.json'}: run --stage select first")
    if s.get("smoke") and not env.smoke:
        raise SystemExit("selection.json was written by a smoke run; refusing to use it")
    return s


# ======================================================================================= looks
def look_counts(pcfg: Mapping[str, Any]) -> dict[str, Any]:
    """Decision-relevant looks per set, tallied from the configured artifacts of earlier phases
    (an existing artifact = that phase's look happened) + this 6a look."""
    out: dict[str, Any] = {}
    for part, entries in pcfg["confirm_report"]["looks"].items():
        rows = [
            {"phase": e["phase"], "artifact": e["path"], "exists": Path(e["path"]).exists()}
            for e in entries
        ]
        n_prior = len({r["phase"] for r in rows if r["exists"]})
        out[part] = {
            "prior_looks": n_prior,
            "phases": sorted({r["phase"] for r in rows if r["exists"]}),
            "including_6a": n_prior + 1,
            "artifacts": rows,
        }
    out["note"] = (
        "count of distinct earlier phases whose artifacts show a CONFIRM / headline look "
        "(adaptive reuse: selections after the first look are not independent of it)"
    )
    return out


# ================================================================================ confirm runs
def confirm_plan(env: p4.Env, weights: str) -> list[p6.RunDef]:
    """CONFIRM classes at the LOCO seed + headline seeds (smoke: 1 class, 1 headline seed)."""
    cfg = env.ctx.cfg
    loco_seed = int(cfg["loco_seed"])
    classes = list(cfg["confirm_classes"]) if not env.smoke else env.dev_classes
    seeds = [int(s) for s in cfg["headline_seeds"]]
    seeds = seeds[:1] if env.smoke else seeds
    runs = [
        p6.RunDef(
            f"{weights}_loco_{c}_s{loco_seed}",
            "confirm",
            loco_seed,
            0,
            "loco",
            (c,),
            f"loco_{c}",
            CONFIRM_CALL_TYPE,
        )
        for c in classes
    ]
    runs += [
        p6.RunDef(
            f"{weights}_headline_s{s}",
            "confirm",
            s,
            s - min(int(x) for x in cfg["headline_seeds"]),
            "headline",
            env.ctx.headline_holdout,
            "headline",
            CONFIRM_CALL_TYPE,
        )
        for s in seeds
    ]
    return runs


def confirm_env(env: p4.Env) -> p4.Env:
    """Env whose test log is the confirm log (call_type phase6a_confirm)."""
    cc = env.pcfg["confirm_report"]
    paths = replace(env.ctx.P, test_log=env.P.results / cc["test_log"])
    return replace(env, ctx=replace(env.ctx, P=paths))


def pool_for(env: p4.Env, weights: str) -> p6.Pool | None:
    """The soup pool that produces `weights` (None for plain models)."""
    if weights in ("i3u", "i3g"):
        return p6.i3_pool(env)
    if weights == "i3i4":
        ch = sel.read_choice(env.P.results)
        if ch is None or ch["i4_pick"] is None:
            raise SystemExit("i3i4 weights need a formed combos choice")
        return p6.i3i4_pool(env, float(env.pcfg["i4"]["lambdas"][ch["i4_pick"]]), ch["soup_type"])
    return None


def ensure_confirm_runs(env: p4.Env, weights: str) -> list[p6.RunDef]:
    """Train + extract every CONFIRM / headline run of one weight set (resumable)."""
    cenv = confirm_env(env)
    runs = confirm_plan(cenv, weights)
    pool = pool_for(cenv, weights)
    if pool is None:
        spec = p6.std_specs(cenv)[weights]
        e_star = p6.e_star_of(cenv, weights)
        for r in runs:
            p6.run_loco_std(cenv, spec, e_star, r)
    else:
        p6.stage_pool_loco(cenv, pool, "confirm", {weights: runs})
    return runs


# ================================================================================ confirm stats
def units_of(
    env: p4.Env, name: str, weights: str, mode: str, runs: list[p6.RunDef], root: Path
) -> dict[str, list[dict[str, ti.BootUnit]]]:
    """part -> unit triplets of the saved confirm view tables of `name`."""
    safe = list(env.ctx.cfg["ood"]["safe_labels"])
    out: dict[str, list[dict[str, ti.BootUnit]]] = {"confirm": [], "headline": []}
    for r in runs:
        stem = f"{name}_{r.run_id[len(weights) + 1 :]}"
        t = pd.read_csv(view_dir(root, name, r.seed, mode) / f"{stem}.csv")
        out["confirm" if r.kind == "loco" else "headline"].append(sel.unit_triplet(t, safe))
    return out


def boot_part(
    units: list[dict[str, ti.BootUnit]],
    n: int,
    seed: int,
    shared: bool,
    level: float,
) -> dict[str, Any]:
    """Mean over units of AUROC / strict rejection@95 / @90 / retention@95, CI + resample samples.
    The three unit kinds share one seed and one set of row counts, hence identical draws."""
    s = ti.bootstrap_mean_ci([u["score"] for u in units], n, seed, shared, level)
    m95 = ti.bootstrap_mean_ci([u["m95"] for u in units], n, seed, shared, level)
    m90 = ti.bootstrap_mean_ci([u["m90"] for u in units], n, seed, shared, level)
    pick = {
        "auroc": (s, "auroc"),
        "rej95": (m95, "strict_recall_95"),
        "rej90": (m90, "strict_recall_90"),
        "retention95": (m95, "retention_95"),
    }
    out: dict[str, Any] = {"ci95": {}, "_samples": {}}
    for k, (b, m) in pick.items():
        out["ci95"][k] = {"point": b["point"][m], "lo": b["lo"][m], "hi": b["hi"][m]}
        out["_samples"][k] = b["samples"][m]
    out["n_units"] = len(units)
    return out


def stage_confirm_report(env: p4.Env) -> None:
    """CONFIRM + headline for ref and the selected candidate; paired CIs; look counts."""
    s = read_selection(env)
    chosen = str(s["chosen"])
    spec = s.get("chosen_spec")
    names = {REF: (REF, "maha_ft", "global")}
    if chosen != REF:
        names[chosen] = (spec["weights"], spec["scorer"], spec["thr"])
    cenv = confirm_env(env)
    root = env.P.results / env.pcfg["confirm_report"]["results_subdir"]
    plans = {w: ensure_confirm_runs(env, w) for w in sorted({v[0] for v in names.values()})}
    for name, (w, scorer, thr) in names.items():
        p6.build_views_for(cenv, name, w, scorer, thr, plans[w], root)
    bs, level = env.pcfg["bootstrap"], float(env.pcfg["bootstrap"]["level"])
    n, seed = env.n_boot, int(bs["seed"])
    res: dict[str, Any] = {}
    for name, (w, _sc, _t) in names.items():
        res[name] = {}
        for mode in ("raw", "neutral"):
            u = units_of(env, name, w, mode, plans[w], root)
            res[name][mode] = {
                "confirm": boot_part(u["confirm"], n, seed, False, level),
                "headline": boot_part(u["headline"], n, seed, True, level),
            }
    deltas: dict[str, Any] = {}
    if chosen != REF:
        for mode in ("raw", "neutral"):
            deltas[mode] = {
                part: ti.paired_delta_ci(
                    res[chosen][mode][part]["_samples"], res[REF][mode][part]["_samples"], level
                )
                for part in ("confirm", "headline")
            }
    for name in res:
        for mode in res[name]:
            for part in res[name][mode]:
                res[name][mode][part].pop("_samples")
    out = {
        "label": "measured, CONFIRM LOCO classes (seed 42) and headline holdout (seeds 42/43/44); "
        "thresholds from calibration-known rows; known side reads test rows",
        "selected": chosen,
        "selected_spec": spec,
        "ref": res[REF],
        "candidate": res.get(chosen) if chosen != REF else None,
        "paired_delta_candidate_minus_ref": deltas or None,
        "adaptive_reuse_looks": look_counts(env.pcfg),
        "bootstrap": {"n_resamples": n, "seed": seed, "level": level},
        "runs": {w: [r.run_id for r in p] for w, p in plans.items()},
        "test_log": str(cenv.ctx.P.test_log),
        "git_sha": git_sha(),
        "smoke": env.smoke,
    }
    p5.write_json(env.P.results / "confirm_report.json", out)
    print(f"[confirm_report] wrote {env.P.results / 'confirm_report.json'} (selected: {chosen})")


# ================================================================================ final_retrain
def final_env(env: p4.Env) -> p4.Env:
    """Env whose pcfg also carries the final_v4 block under the name phase4e_final reads."""
    return replace(env, pcfg={**env.pcfg, "final_v3": env.pcfg["final_v4"]})


def shipped_ood_variant(
    cfg: Mapping[str, Any],
    npz_path: Path,
    scorer: str,
    thr_rule: str,
    retention: float,
    tb: Mapping[str, Any],
) -> dict[str, Any]:
    """The shipped open-set scoring variant on the final model's saved arrays (train / val / test;
    no new test inference): scorer fitted on train features, threshold rule on val (the calibration
    rows of the 12-class model); I6b cross-fits over train + val features."""
    z = np.load(npz_path)
    labels = list(data_mod.LABELS)
    full = data_mod.load_data(cfg["data_path"]).set_index("id")["label"]
    l2i = {lab: i for i, lab in enumerate(labels)}
    y_tr = np.array([l2i[full[i]] for i in z["train_ids"]])
    y_val = np.array([l2i[full[i]] for i in z["val_ids"]])
    fit = ov.fit_feature_scorer(scorer, z["train_features"], y_tr, len(labels))
    s_val, s_test = fit(z["val_features"]), fit(z["test_features"])
    pred_val, pred_test = z["val_logits"].argmax(1), z["test_logits"].argmax(1)
    out: dict[str, Any] = {
        "method": f"{scorer}/{thr_rule}",
        "scorer": scorer,
        "thr_rule": thr_rule,
        "retention_target": retention,
        "n_val": len(s_val),
        "n_test": len(s_test),
    }
    t_global = threshold_at_retention(s_val, retention)
    if thr_rule == "global":
        vec_val, vec_test = np.full(len(s_val), t_global), np.full(len(s_test), t_global)
        out["threshold"] = float(t_global)
    elif thr_rule == "i6a":
        pc = ov.fit_per_class_thresholds(
            s_val, pred_val, len(labels), retention, int(tb["per_class_min_rows"])
        )
        vec_val, vec_test = pc.for_pred(pred_val), pc.for_pred(pred_test)
        out |= {
            "per_class_thresholds": {labels[c]: float(t) for c, t in enumerate(pc.thresholds)},
            "global_threshold": float(t_global),
            "fallback_classes": [labels[c] for c in pc.fallback],
        }
    else:
        feats = np.concatenate([z["train_features"], z["val_features"]])
        oof = ov.crossfit_scores(
            ov.scorer_fit_fn(scorer),
            feats,
            np.concatenate([y_tr, y_val]),
            len(labels),
            int(tb["crossfit_folds"]),
            int(tb["crossfit_seed"]),
        )
        t_cf = threshold_at_retention(oof, retention)
        vec_val, vec_test = np.full(len(s_val), t_cf), np.full(len(s_test), t_cf)
        out |= {"threshold": float(t_cf), "n_crossfit_rows": int(len(oof))}
    out |= {
        "val_retention_achieved": float(np.mean(s_val >= vec_val)),
        "test_coverage_at_threshold": float(np.mean(s_test >= vec_test)),
        "note": "12 classes known: fitted on train / val, applied to the saved test arrays "
        "(no new test inference)",
    }
    return out


def soup_trainer(env: p4.Env, pool: p6.Pool, soup_type: str, e_star: int, record: dict[str, Any]):
    """A `train_model` replacement for final.py: trains the replicate-0 members on the FULL train
    split (member seeds 0..n-1, shared head init), averages them (greedy criterion = val macro-F1),
    and returns (FoldResult, model carrying the soup weights, tokenizer)."""

    def train_soup(
        cfg_t: Any,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        run_meta: dict[str, Any],
        labels: list[str] | None = None,
    ) -> tuple[FoldResult, Any, Any]:
        from intent_router.train import train_model

        members = p6.soup_members(env, 0)
        sds: list[dict[str, Any]] = []
        res = model = tok = None
        t0, f1_val, train_losses = time.perf_counter(), [], []
        bs = int(env.pcfg["predict_batch_size"])
        lab = list(data_mod.LABELS) if labels is None else list(labels)
        for m in members:
            model = None  # drop the previous member before the next one trains (8 GB card)
            p6.free_gpu()
            tcfg = replace(cfg_t, model_seed=m)
            res, model, tok = train_model(
                tcfg,
                train_df,
                eval_df,
                {**run_meta, "member": m},
                labels=labels,
                hooks=pool.train.hooks(),
            )
            sds.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            train_losses.append(res.epochs[-1]["train_loss"])
            f1_val.append(p6.f1_of(model, tok, eval_df, lab, tcfg, bs))
        assert res is not None and model is not None
        if soup_type == "uniform":
            chosen, info = list(range(len(sds))), {"type": "uniform"}
        else:

            def eval_subset(idx: list[int]) -> float:
                model.load_state_dict(soup.average_state_dicts([sds[j] for j in idx]))
                return p6.f1_of(model, tok, eval_df, lab, cfg_t, bs)

            g = soup.greedy_soup(f1_val, eval_subset)
            chosen, info = (
                g["selected"],
                {
                    "type": "greedy",
                    "trace": g["trace"],
                    "criterion": "val macro-F1 of the val split",
                },
            )
        model.load_state_dict(soup.average_state_dicts([sds[j] for j in chosen]))
        f1 = p6.f1_of(model, tok, eval_df, lab, cfg_t, bs)
        record.update(
            {
                "soup": info
                | {
                    "members": members,
                    "selected": [members[j] for j in chosen],
                    "member_val_f1": f1_val,
                    "soup_val_f1": f1,
                    "train_spec": pool.train.info(),
                }
            }
        )
        logits = predict(model, tok, eval_df["text"].tolist(), cfg_t.model_name, cfg_t.max_len, bs)[
            0
        ]
        y_val = local_gold(eval_df, lab)
        # Finite, deterministic epoch record (NaN != NaN would break final.py's determinism check):
        # train_loss = mean of the members' last-epoch train losses, eval_loss = soup CE on val.
        ep = [
            {
                "epoch": e_star,
                "train_loss": float(np.mean(train_losses)),
                "eval_loss": float(-np.log(softmax(logits)[np.arange(len(y_val)), y_val]).mean()),
                "macro_f1": f1,
                "accuracy": accuracy(y_val, logits.argmax(1)),
            }
        ]
        out = dataclasses.replace(
            res,
            epochs=ep,
            probs=np.zeros((1, 0, 0), dtype=np.float32),
            wall_clock_s=time.perf_counter() - t0,
        )
        return out, model, tok

    return train_soup


def std_trainer(spec: p6.TrainSpec, record: dict[str, Any]):
    """A `train_model` replacement for final.py: the plain recipe + the spec's hooks (I4)."""

    def train(
        cfg_t: Any,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        run_meta: dict[str, Any],
        labels: list[str] | None = None,
    ) -> tuple[FoldResult, Any, Any]:
        from intent_router.train import train_model

        record["train_spec"] = spec.info()
        return train_model(cfg_t, train_df, eval_df, run_meta, labels, hooks=spec.hooks())

    return train


def patched_final_training(trainer: Any) -> Any:
    """Context manager making final.py's `train_model` the given trainer (restored afterwards)."""
    import contextlib

    @contextlib.contextmanager
    def cm() -> Any:
        orig = final_mod.train_model
        final_mod.train_model = trainer
        try:
            yield
        finally:
            final_mod.train_model = orig

    return cm()


def cv_stand_ins(env: p4.Env, key: str, cfg: Mapping[str, Any], e_star: int) -> None:
    """selection json + OOF csv the final evaluation reads: per-(seed, fold) CV macro-F1 / acc. of
    the e*-stopped fold models, and the 3 per-seed OOF clean predictions of every row."""
    cid = cfg["analysis"]["oof_config_id"]
    f1, acc, frames = [], [], []
    for s in p6.model_seeds(env):
        clean = pd.read_csv(p6.wdir(env, key, s) / "oof_clean.csv")
        frames.append(clean)
        for k in env.folds:
            g = clean[clean["fold"] == k]
            f1.append(macro_f1(g["gold"].to_numpy(), g["pred"].to_numpy()))
            acc.append(accuracy(g["gold"], g["pred"]))
    note = (
        f"Open-set improvement round weights {key}: {len(env.folds)} folds x model seeds / "
        f"replicates "
        f"{p6.model_seeds(env)}, fold-seed s0 ({len(f1)} fold-runs); v1 used 9 fold-runs per id"
    )
    p5.write_json(
        Path(cfg["selection_path"]),
        pf.selection_stand_in(cid, e_star, np.array(f1), np.array(acc), note),
    )
    oof = pf.build_stand_in_oof(frames, int(cfg["analysis"]["oof_predictions_per_id"]))
    d = Path(cfg["analysis"]["oof_dir"])
    d.mkdir(parents=True, exist_ok=True)
    oof.to_csv(d / f"{cid}.csv", index=False, float_format="%.9g")


def stage_final_retrain(env: p4.Env, force_cand: str | None = None) -> None:
    """Retrain the selected candidate as the final model (see the module docstring)."""
    s = read_selection(env)
    chosen = str(force_cand) if (env.smoke and force_cand) else str(s["chosen"])
    if chosen == REF:
        raise SystemExit("selection chose ref (v1): nothing to retrain, v1 ships unchanged")
    spec = next(
        (
            x
            for x in sel.resolve_candidates(env.pcfg, sel.read_choice(env.P.results))
            if x["key"] == chosen
        ),
        None,
    )
    if spec is None or not spec["formed"]:
        raise SystemExit(f"{chosen!r} is not a formed candidate")
    if not env.smoke and not (env.P.results / "confirm_report.json").exists():
        raise SystemExit("run --stage confirm_report first (CONFIRM + headline vs ref)")
    w, scorer, thr = spec["weights"], spec["scorer"], spec["thr"]
    e_star = p6.e_star_of(env, w)
    fenv = final_env(env)
    v4 = env.pcfg["final_v4"]
    version = str(v4["version"])
    pool = pool_for(env, w)
    train_spec = p6.std_specs(env).get(w) if pool is None else pool.train
    assert train_spec is not None
    rec = p4.Recipe(w, a1=train_spec.a1)
    cfg = pf.derive_final_cfg(env.fcfg, rec, e_star, fenv.pcfg, env.smoke)
    cfg["wandb"]["group"] = f"phase6a-final-{version}"
    if not env.smoke:
        cfg["analysis"]["oof_config_id"] = f"p6a_{w}"
    v1_fp: str | None = None
    if env.smoke:
        root = env.P.results.parent / "final_v4"
        P = final_mod.Paths(
            root / "results",
            root / "figures",
            root / "outputs",
            root / "model",
            root / "test_eval_log.jsonl",
            True,
        )
        use_wandb = False
    else:
        for sd in p6.model_seeds(env):  # preflight: everything the stand-ins read must exist
            if not (p6.wdir(env, w, sd) / "oof_clean.csv").exists():
                raise SystemExit(f"OOF predictions of {w} s{sd} missing")
        pf.archive_v1(
            Path(env.fcfg["results_dir"]),
            Path(v4["archive_results"]),
            Path(env.fcfg["figures_dir"]),
            Path(env.fcfg["outputs_dir"]),
            Path(v4["archive_outputs"]),
        )
        v1_fp = (p4.read_json(Path(v4["archive_results"]) / "model_version.json") or {}).get(
            "model_fingerprint"
        )
        cv_stand_ins(env, w, cfg, e_star)
        P = final_mod.make_paths(cfg, False)
        use_wandb = bool(cfg["wandb"]["enabled"]) and bool(env.ctx.use_wandb)
        d = Path(v4["results_dir"])
        d.mkdir(parents=True, exist_ok=True)
        (d / f"final_{version}_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    record: dict[str, Any] = {}
    trainer = (
        soup_trainer(env, pool, spec_soup_type(env, w), e_star, record)
        if pool is not None
        else std_trainer(train_spec, record)
    )
    if not env.smoke and pf.training_done(P.results, P.model_dir, v1_fp):
        print("[final] train stage already finished for this model: not retraining")
    else:
        with patched_final_training(trainer):
            final_mod.stage_train(cfg, P, env.smoke)
    final_mod.stage_evaluate(cfg, P, env.smoke, use_wandb)
    tsum = p4.read_json(P.results / "train_summary.json") or {}
    ood = shipped_ood_variant(
        cfg,
        P.outputs / "features_logits.npz",
        scorer,
        thr,
        float(env.pcfg["trackb"]["retention"]),
        env.pcfg["trackb"],
    )
    p5.write_json(P.results / "ood_shipped.json", ood)
    p5.write_json(
        P.results / "model_version.json",
        {
            "version": version,
            "candidate": chosen,
            "weights": w,
            "scorer": scorer,
            "thr_rule": thr,
            "deployed_epoch": e_star,
            "train_config": cfg["train"],
            "extras": record,
            "model_fingerprint": tsum.get("model_fingerprint"),
            "model_dir": str(P.model_dir),
            "shipped_ood_score": f"{scorer}/{thr}",
            "archived_previous": v4["archive_results"],
            "selection": (env.P.results / "selection.json").as_posix(),
            "git_sha": git_sha(),
            "smoke": env.smoke,
        },
    )
    if env.smoke:
        print("[final] smoke: trained + evaluated on val stand-in; analysis/robustness skipped")
        return
    cfg_path = Path(v4["results_dir"]) / f"final_{version}_config.yaml"
    subprocess.run(  # noqa: S603 - analysis refuses to run in a process that loaded ood/baselines
        [sys.executable, "-m", "intent_router.analysis", "--config", str(cfg_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    pf.patch_cv_note(
        P.results / "track_a.json",
        int(cfg["analysis"]["oof_predictions_per_id"]),
        p6.model_seeds(env),
    )
    pf.write_side_by_side(fenv, version)
    p5.write_json(
        Path(v4["results_dir"]) / "oof_robustness_6a.json", pf.oof_robustness_from_4e(env, [REF, w])
    )
    pf.robustness_v3(fenv, cfg_path)


def spec_soup_type(env: p4.Env, weights: str) -> str:
    """Soup type of a pooled weight set: i3u uniform, i3g greedy, i3i4 from the combos choice."""
    if weights == "i3u":
        return "uniform"
    if weights == "i3g":
        return "greedy"
    ch = sel.read_choice(env.P.results)
    assert ch is not None
    return str(ch["soup_type"])
