"""Open-set improvement round (pre-registered 2026-10-03).

    python -m intent_router.phase6a --config configs/phase6a.yaml --stage <stage> [--smoke]

Stages (each resumable: a result file's existence is the "done" marker; sequential, exclusive
GPU access):
  reuse_check     read-only: would the 4e v1 curves be reused for ref?
  ref             curves (4e v1 curves reused ONLY if config-identical) -> epoch -> OOF -> DEV LOCO
  i4              same for I4a / I4b (A1 + lambda * symmetric KL with the prefix-swapped twin)
  i3              soup pool (7 members per fold and per DEV class): member curves -> e* -> members
                  -> I3u / I3g soups (OOF + DEV LOCO)
  combos          pre-listed pick rules -> I3+I4 pool (if formed) -> combo views
  views           (re)compute every score view from the saved arrays (CPU)
  select          axes + selection (phase6a_select.py; reads DEV / CV outputs only)
  confirm_report  AFTER select: CONFIRM + headline for the selected candidate and ref
  final_retrain   AFTER select, only when selection != ref (phase6a_post.py); not run here
  status          what exists on disk

Outputs: results/phase6a/<weights>/s<seed>/{curves, oof_*.csv, folds.json, trackb/runs/*.json,
views/<mode>/*.csv}; outputs/phase6a/<weights>/s<seed>/trackb_arrays/*.npz (raw AND ID-neutral
logits / features of the train / cal / eval-known / eval-unknown rows; neutral arrays carry an
`n_` prefix). Only ids and numbers go under results/. Fold models are never saved; soup members
are deleted once their soups are built (disk).

IMPLEMENTATION NOTES (readings chosen where the rules fixed before the runs are ambiguous; most
conservative each)
 1. CV macro-F1 (axis a) of every candidate comes from its OOF clean predictions (the e*-stopped
    fold models), uniformly; ref's reused 4e curves only decide e*. folds.json records the
    curve-vs-model difference. ref's own argmax is computed from the 15 curves (expected 8, the
    value fixed beforehand is recorded, not forced).
 2. I4 twin: from the A1-randomised training text EVERY ID prefix is redrawn from {PO, LD, REF,
    random 2-4 uppercase}; loss = lambda * sum_rows(KL(p||q) + KL(q||p)) / batch size (no 1/2
    factor), rows without an ID contribute 0, both sides carry gradient. I4 curves use it too.
 3. Soups: members share the head init (seed 0) and differ in member seed 0..6 (data order,
    dropout); they train on 85% of the fold-train / LOCO-train rows (`soup.inner_split`, seeded
    per unit, one split for the whole pool); member curves are the held-out-fold curves; e* is the
    argmax of the mean of all member curves; members are retrained and stopped at e*. Greedy:
    members sorted by inner-holdout macro-F1, best first, a member is added iff the averaged
    soup's inner-holdout F1 does not drop (>=). Replicate r <-> model seed r (DEV) and <-> seed
    42 + r (headline); CONFIRM classes use replicate 0 (seed 42).
 4. ID-neutral mode: calibration, eval-known and eval-unknown TEXT is neutralised at inference
    time (a second forward pass over the same rows); train rows and training text are unchanged,
    the Gaussians are fitted on the raw train features. The known side reads test rows
    (counted in the report appendix); both passes are covered by ONE logged test-inference
    call per run (call_type phase6a_trackb).
 5. I6a groups calibration rows by the PREDICTED class (all that is known at serving time). I6b
    cross-fits only the Gaussian (5 stratified folds over known train + cal rows, seeded); the
    network weights are shared, so train rows are optimistic - DEV measures the consequence. For
    I6 candidates AUROC is the raw-score AUROC (thresholds do not change rankings); rejection
    uses the margins.
 6. Combo pick rules (stage `combos`, written to results/phase6a/combos_choice.json): lambda for
    I3+I4 = the I4 variant eligible on axis (a) (point estimate) with the lower neutral-swap flip
    rate (ties: smaller lambda); soup type for ALL combos = the I3 variant with the higher
    standalone CV macro-F1 (ties: uniform); the I1 scorer (I3+I1, reused for I3+I4+I1 and
    I3+I4+I1+I6) = the one with the higher ID-neutral DEV rejection@95 on the chosen I3 soup; the
    I6 variant = the one with the higher ID-neutral DEV rejection@95 as a standalone candidate on
    ref weights (ties: i6a). If no I4 variant is eligible on (a), the I4 combos are reported
    "not formed".
 7. Only the selection scorer(s) are scored (not the other 13 Track B methods); LOCO runs write
    npz + run json + view tables, no per-method score CSV.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import ood_variants as ov
from intent_router import phase4c as p4
from intent_router import phase4e as p5
from intent_router import phase6a_select as sel
from intent_router import phase6a_views as pv
from intent_router import robustness as rb
from intent_router import soup
from intent_router import trackb_improve as ti
from intent_router.cv import load_cv_frame
from intent_router.evaluate import git_sha
from intent_router.models import build_model, predict, state_dict_sha256
from intent_router.ood import HoldoutSets, build_holdout_sets
from intent_router.stats import macro_f1
from intent_router.trackb import gpu_section
from intent_router.train import TrainConfig, TrainHooks, train_fold
from intent_router.train import train_model as _train_model
from intent_router.twin_kl import make_twin_loss

STAGES = (
    "reuse_check",
    "ref",
    "i4",
    "i3",
    "combos",
    "views",
    "select",
    "confirm_report",
    "final_retrain",
    "status",
)
REF = "ref"
OOF_NAMES = p5.OOF_NAMES
write_json = p5.write_json
SOUP_TYPES = ("uniform", "greedy")


# ================================================================================== environment
@dataclass
class Env6(p4.Env):
    """phase4c.Env whose smoke runs use 2 folds (the base class has 1)."""

    @property
    def folds(self) -> list[int]:
        return [0, 1] if self.smoke else list(range(int(self.pcfg["guard"]["n_folds"])))


def make_env(args: argparse.Namespace) -> Env6:
    """phase4c.make_env on configs/phase6a.yaml, with the smoke sandbox redirected to 6a."""
    args.smoke_root = "outputs/phase6a_smoke"
    args.smoke_adopt = None
    e = p4.make_env(args)
    return Env6(e.pcfg, e.P, e.ctx, e.smoke, None, e._inputs)  # noqa: SLF001


def model_seeds(env: p4.Env) -> list[int]:
    return [int(s) for s in env.pcfg["model_seeds"]]


def seed_pcfg(env: p4.Env, seed: int) -> dict[str, Any]:
    """pcfg with guard.model_seed = seed (eval_fold writes it into the OOF csvs)."""
    return {**env.pcfg, "guard": {**env.pcfg["guard"], "model_seed": int(seed)}}


def wdir(env: p4.Env, key: str, seed: int) -> Path:
    return env.P.results / key / f"s{seed}"


def odir(env: p4.Env, key: str, seed: int) -> Path:
    return env.P.outputs / key / f"s{seed}"


def candidate_specs(env: p4.Env) -> list[dict[str, Any]]:
    return sel.resolve_candidates(env.pcfg, sel.read_choice(env.P.results))


def free_gpu() -> None:
    p4._free_gpu()  # noqa: SLF001


def lock(env: p4.Env, label: str, expected_s: int | None = None) -> Any:
    """GPU section (a no-op context on CPU-only runs)."""
    exp = int(env.pcfg["gpu_lock_expected_s"]) if expected_s is None else expected_s
    return gpu_section(exp, f"intent-router phase6a {label}", float(env.pcfg["gpu_lock_poll_s"]))


# ================================================================================ training specs
@dataclass(frozen=True)
class TrainSpec:
    """How one model of a weight set / soup pool is trained (the rest is final.yaml's recipe)."""

    key: str
    a1: bool = False
    kl_lambda: float = 0.0
    head_init_seed: int | None = None
    inner_frac: float = 0.0  # > 0: soup member (trains on the other 1 - frac of the rows)

    def recipe(self) -> p4.Recipe:
        return p4.Recipe(self.key, a1=self.a1)

    def hooks(self) -> TrainHooks | None:
        if self.head_init_seed is None and self.kl_lambda <= 0:
            return None
        return TrainHooks(
            self.head_init_seed, make_twin_loss(self.kl_lambda) if self.kl_lambda > 0 else None
        )

    def info(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "a1": self.a1,
            "kl_lambda": self.kl_lambda,
            "head_init_seed": self.head_init_seed,
            "inner_frac": self.inner_frac,
        }


def std_specs(env: p4.Env) -> dict[str, TrainSpec]:
    """ref (v1 recipe), I4a, I4b."""
    out = {REF: TrainSpec(REF)}
    for key, lam in env.pcfg["i4"]["lambdas"].items():
        out[key] = TrainSpec(key, a1=True, kl_lambda=float(lam))
    return out


@dataclass(frozen=True)
class Pool:
    """A soup pool: member training spec + the (weights key, soup type) pairs built from it."""

    key: str
    train: TrainSpec
    types: tuple[tuple[str, str], ...]

    @property
    def dir_key(self) -> str:
        return f"pool_{self.key}"


def i3_pool(env: p4.Env) -> Pool:
    s = env.pcfg["soup"]
    return Pool(
        "i3",
        TrainSpec(
            "i3", head_init_seed=int(s["head_init_seed"]), inner_frac=float(s["inner_holdout_frac"])
        ),
        (("i3u", "uniform"), ("i3g", "greedy")),
    )


def i3i4_pool(env: p4.Env, lam: float, soup_type: str) -> Pool:
    s = env.pcfg["soup"]
    return Pool(
        "i3i4",
        TrainSpec(
            "i3i4",
            a1=True,
            kl_lambda=lam,
            head_init_seed=int(s["head_init_seed"]),
            inner_frac=float(s["inner_holdout_frac"]),
        ),
        (("i3i4", soup_type),),
    )


def train_cfg6(env: p4.Env, spec: TrainSpec, seed: int, stop_epoch: int | None) -> TrainConfig:
    """final.yaml train block + the spec's A1 override; stop_epoch None => the full schedule."""
    return p4.train_cfg_for(env, spec.recipe(), seed, stop_epoch)


def pool_size(env: p4.Env) -> int:
    return 5 if env.smoke else int(env.pcfg["soup"]["pool_size"])


def soup_members(env: p4.Env, replicate: int) -> list[int]:
    s = env.pcfg["soup"]
    n = 3 if env.smoke else int(s["members_per_soup"])
    return soup.rotated_members(pool_size(env), n, replicate, int(s["rotate_shift"]))


# ==================================================================================== curves (std)
def curve_path(env: p4.Env, key: str, seed: int, fold: int) -> Path:
    return wdir(env, key, seed) / "curves" / f"f{fold}.json"


def epoch_path(env: p4.Env, key: str) -> Path:
    return env.P.results / key / "epoch.json"


def e_star_of(env: p4.Env, key: str) -> int:
    ep = p4.read_json(epoch_path(env, key))
    if ep is None:
        raise SystemExit(f"no epoch.json for {key}: run its curves/epoch stage first")
    return int(ep["e_star"])


def check_ref_reuse(env: p4.Env, spec: TrainSpec) -> dict[str, Any]:
    """Compare the 4e v1 curve files with what 6a would train for ref, field by field (4e's check):
    stored `config` == the TrainConfig built here (ignoring W&B group/project), same extras, a full
    20-epoch curve, no NaN, same (seed, fold), same fold seed / n_folds in both configs."""
    rc = env.pcfg["reuse_4e"]
    pc4 = yaml.safe_load(Path(rc["config"]).read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "match": False,
        "per_curve": [],
        "from": f"{rc['results_dir']}/{rc['key']}",
    }
    gd = {
        n: [a, b]
        for n, a, b in (
            (
                "guard.fold_seed_idx",
                pc4["guard"]["fold_seed_idx"],
                env.pcfg["guard"]["fold_seed_idx"],
            ),
            ("guard.n_folds", pc4["guard"]["n_folds"], env.pcfg["guard"]["n_folds"]),
            ("factors.a1", pc4["factors"]["a1"], env.pcfg["factors"]["a1"]),
        )
        if a != b
    }
    out["global_diff"] = gd
    ok = not gd
    for seed in rc["seed_keys"]:
        want = p5.expected_config(train_cfg6(env, spec, int(seed), None))
        for f in range(int(env.pcfg["guard"]["n_folds"])):
            src = Path(rc["results_dir"]) / rc["key"] / f"s{seed}" / "curves" / f"f{f}.json"
            rec: dict[str, Any] = {"seed": seed, "fold": f, "file": str(src)}
            if not src.exists():
                rec["why"] = "missing"
                ok = False
                out["per_curve"].append(rec)
                continue
            st = json.loads(src.read_text(encoding="utf-8"))
            diff = p5.config_diff(st["config"], want)
            rec |= {
                "config_diff": diff,
                "extras_equal": st["extras"] == {"factors": []},
                "n_epochs": len(st["macro_f1"]),
                "ids_ok": (st["seed"], st["fold"]) == (seed, f),
                "nan_detected": bool(st["nan_detected"]),
            }
            if not (
                not diff
                and rec["extras_equal"]
                and rec["n_epochs"] == 20
                and rec["ids_ok"]
                and not rec["nan_detected"]
            ):
                ok = False
            out["per_curve"].append(rec)
    out["match"] = bool(ok)
    return out


def print_reuse_report(rep: Mapping[str, Any]) -> None:
    print(f"[reuse] ref <- {rep['from']}: curves match={rep['match']}")
    for k, v in rep["global_diff"].items():
        print(f"[reuse]   MISMATCH {k}: 4e={v[0]} 6a={v[1]}")
    bad = [
        r
        for r in rep["per_curve"]
        if r.get("why")
        or r.get("config_diff")
        or not r.get("extras_equal", True)
        or r.get("n_epochs", 20) != 20
    ]
    for r in bad:
        print(f"[reuse]   s{r['seed']} f{r['fold']}: {r.get('why') or r.get('config_diff')}")
    print(f"[reuse]   {len(rep['per_curve']) - len(bad)}/{len(rep['per_curve'])} curves identical")


def run_curves_std(env: p4.Env, spec: TrainSpec) -> None:
    """15 fold-runs (3 seeds x 5 folds): per-epoch held-out macro-F1, one file per run."""
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    reuse: dict[str, Any] | None = None
    if spec.key == REF and not env.smoke:
        reuse = check_ref_reuse(env, spec)
        print_reuse_report(reuse)
        if not reuse["match"]:
            print("[curves] ref: 4e curves NOT reused (mismatch above); training all 15")
    for seed in model_seeds(env):
        tcfg = train_cfg6(env, spec, seed, None)
        for f in env.folds:
            fp = curve_path(env, spec.key, seed, f)
            if fp.exists():
                continue
            if reuse is not None and reuse["match"] and seed in env.pcfg["reuse_4e"]["seed_keys"]:
                src = Path(env.pcfg["reuse_4e"]["results_dir"]) / env.pcfg["reuse_4e"]["key"]
                src = src / f"s{seed}" / "curves" / f"f{f}.json"
                frec = json.loads(src.read_text(encoding="utf-8"))
                frec |= {"candidate": spec.key, "reused_from": str(src)}
                write_json(fp, frec)
                print(f"[curves] {spec.key} s{seed} f{f}: reused {src}")
                continue
            tr, ev = p4.fold_data(env, frame, f)
            meta = {
                "run_id": f"p6a_curve_{spec.key}_s{seed}_f{f}",
                "fold": f,
                "candidate": spec.key,
            }
            with lock(env, f"curve {spec.key} s{seed} f{f}"):
                res = train_fold(tcfg, tr, ev, meta, hooks=spec.hooks())
            free_gpu()
            frec = {
                "candidate": spec.key,
                "seed": seed,
                "fold": f,
                "extras": {"factors": list(spec.recipe().factors)},
                "macro_f1": [e["macro_f1"] for e in res.epochs],
                "accuracy": [e["accuracy"] for e in res.epochs],
                "wall_clock_s": res.wall_clock_s,
                "peak_vram_mb": res.peak_vram_mb,
                "wandb_url": res.wandb_url,
                "nan_detected": res.nan_detected,
                "gpu_exclusive": res.gpu_exclusive,
                "config": res.config,
                "train_spec": spec.info(),
            }
            write_json(fp, frec)
            print(f"[curves] {spec.key} s{seed} f{f}: best-epoch F1 {max(frec['macro_f1']):.4f}")


def load_curves(env: p4.Env, key: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows, keys = [], []
    for seed in model_seeds(env):
        for f in env.folds:
            fp = curve_path(env, key, seed, f)
            if not fp.exists():
                raise SystemExit(f"missing curve {fp}: run the curves stage first")
            rows.append(json.loads(fp.read_text(encoding="utf-8"))["macro_f1"])
            keys.append((seed, f))
    return np.array(rows, dtype=float), keys


def stage_epoch_std(env: p4.Env, spec: TrainSpec) -> None:
    out = epoch_path(env, spec.key)
    if out.exists():
        print(f"[epoch] {spec.key}: epoch.json exists, skipping")
        return
    curves, keys = load_curves(env, spec.key)
    summ = p5.epoch_summary(curves)
    per_seed = {
        str(s): int(np.argmax(curves[[i for i, k in enumerate(keys) if k[0] == s]].mean(0))) + 1
        for s in model_seeds(env)
    }
    extra: dict[str, Any] = {}
    if spec.key == REF:
        pre = int(env.pcfg["ref"]["preregistered_epoch"])
        extra = {"preregistered_epoch": pre, "matches_preregistered_epoch": summ["e_star"] == pre}
        if summ["e_star"] != pre and not env.smoke:
            print(f"[epoch] WARNING ref own argmax {summ['e_star']} != pre-registered {pre}")
    reused = sum(
        "reused_from" in json.loads(curve_path(env, spec.key, s, f).read_text("utf-8"))
        for s, f in keys
    )
    write_json(
        out,
        {
            "candidate": spec.key,
            **summ,
            "per_seed_own_argmax": per_seed,
            "n_curves_reused_from_4e": reused,
            "smoke": env.smoke,
            "git_sha": git_sha(),
            **extra,
        },
    )
    print(f"[epoch] {spec.key}: e*={summ['e_star']} F1@e*={summ['f1_at_e_star']:.4f}")


# ============================================================================ OOF (std + soups)
def delete_dir_guarded(path: Path, parent_name: str) -> bool:
    """Remove a transient model directory. Refuses (AssertionError) anything that is not
    .../phase6a*/.../<parent_name>/<child>; returns whether something was deleted."""
    p = path.resolve()
    if not any(part in ("phase6a", "phase6a_smoke") for part in p.parts):
        raise AssertionError(f"refusing to delete outside outputs/phase6a*: {p}")
    if p.parent.name != parent_name:
        raise AssertionError(f"refusing to delete {p}: parent is not {parent_name!r}")
    if not p.exists():
        return False
    shutil.rmtree(p)
    return True


def eval_fold_frames(
    env: p4.Env,
    model: Any,
    tok: Any,
    ev: pd.DataFrame,
    seed: int,
    fold: int,
) -> dict[str, pd.DataFrame]:
    t = env.fcfg["train"]
    return p4.eval_fold(
        model,
        tok,
        t["model_name"],
        int(t["max_len"]),
        int(env.pcfg["predict_batch_size"]),
        ev,
        env.inputs.eval_mt,
        seed_pcfg(env, seed),
        fold,
    )


def write_parts(
    parts: Mapping[str, Path],
    got: Mapping[str, pd.DataFrame],
    meta: dict[str, Any],
) -> None:
    for n, df in got.items():
        df.to_csv(parts[n], index=False, float_format="%.9g")
    write_json(parts["meta"], meta)  # meta last = part set complete


def parts_complete(parts: Mapping[str, Path]) -> bool:
    return all(p.exists() for p in parts.values())


def finish_oof(env: p4.Env, key: str, seed: int, e_star: int, extra: Mapping[str, Any]) -> None:
    """Concatenate the per-fold parts into oof_*.csv + folds.json (the done marker)."""
    parts_dir = odir(env, key, seed) / "oof_parts"
    out_dir = wdir(env, key, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for n in OOF_NAMES:
        dfs = [pd.read_csv(p5.part_paths(parts_dir, k)[n]) for k in env.folds]
        keep = [d for d in dfs if len(d)] or dfs[:1]
        frames[n] = pd.concat(keep, ignore_index=True)
        frames[n].to_csv(out_dir / f"oof_{n}.csv", index=False, float_format="%.9g")
    metas = [
        json.loads(p5.part_paths(parts_dir, k)["meta"].read_text(encoding="utf-8"))
        for k in env.folds
    ]
    write_json(
        out_dir / "folds.json",
        p5.summarize_oof(env, key, seed, e_star, frames, {"folds": metas, **extra}),
    )
    s = json.loads((out_dir / "folds.json").read_text(encoding="utf-8"))
    print(
        f"[folds] {key} s{seed}: OOF acc {s['oof_accuracy']:.4f} flip "
        f"{s['swap']['pooled']['flip_rate']} drop {s['noise']['drop']:.4f}"
    )


def stage_folds_std(env: p4.Env, spec: TrainSpec) -> None:
    """OOF clean / swap / noise / MT predictions of the e*-stopped fold models (retrained; the model
    is evaluated in memory under the same lock section and never saved)."""
    e_star = e_star_of(env, spec.key)
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    for seed in model_seeds(env):
        if (wdir(env, spec.key, seed) / "folds.json").exists():
            print(f"[folds] {spec.key} s{seed}: folds.json exists, skipping")
            continue
        parts_dir = odir(env, spec.key, seed) / "oof_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        cross: list[float] = []
        for k in env.folds:
            pp = p5.part_paths(parts_dir, k)
            if parts_complete(pp):
                continue
            tr, ev = p4.fold_data(env, frame, k)
            tcfg = train_cfg6(env, spec, seed, e_star)
            meta = {"run_id": f"p6a_fold_{spec.key}_s{seed}_f{k}", "fold": k}
            print(f"[folds] {spec.key} s{seed} fold {k}: training {len(tr)} rows to epoch {e_star}")
            with rb.wandb_disabled(), lock(env, f"fold {spec.key} s{seed} f{k}") as lk:
                res, model, tok = _train_model(tcfg, tr, ev, meta, hooks=spec.hooks())
                try:
                    got = eval_fold_frames(env, model, tok, ev, seed, k)
                finally:
                    model = None
                    free_gpu()
            cp = curve_path(env, spec.key, seed, k)
            diff = None
            if cp.exists():
                diff = abs(
                    json.loads(cp.read_text("utf-8"))["macro_f1"][e_star - 1]
                    - res.epochs[-1]["macro_f1"]
                )
            write_parts(
                pp,
                got,
                {
                    "fold": k,
                    "stop_epoch": e_star,
                    "final_epoch_eval": res.epochs[-1],
                    "curve_vs_fold_model_abs_diff": diff,
                    "wall_clock_s": res.wall_clock_s,
                    "gpu_exclusive": res.gpu_exclusive,
                    "lock_waited_s": lk.get("waited_s"),
                    "train_spec": spec.info(),
                },
            )
        for k in env.folds:
            m = json.loads(p5.part_paths(parts_dir, k)["meta"].read_text(encoding="utf-8"))
            if m.get("curve_vs_fold_model_abs_diff") is not None:
                cross.append(m["curve_vs_fold_model_abs_diff"])
        finish_oof(
            env,
            spec.key,
            seed,
            e_star,
            {
                "curve_vs_fold_model_max_abs_diff": max(cross) if cross else None,
                "train_spec": spec.info(),
            },
        )


# ====================================================================== Track B arrays (shared)
def neutral_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(text=frame["text"].map(ov.neutralize_ids))


def extract_arrays(
    ctx: ti.Ctx,
    model: Any,
    tok: Any,
    sets: HoldoutSets,
    tcfg: TrainConfig,
    run_id: str,
    call_type: str,
) -> tuple[dict[str, np.ndarray], str]:
    """Logits + penultimate features of the train / cal / eval-known / eval-unknown rows on raw text
    and (n_ prefix) the cal / eval rows on ID-neutralised text. Mirrors ti.train_and_extract's row
    handling (non-test unknowns, then ONE call over all test rows); the test call is logged
    first."""
    bs = int(ctx.fcfg["predict_batch_size"])

    def infer(frame: pd.DataFrame, neutral: bool = False) -> tuple[np.ndarray, np.ndarray]:
        texts = frame["text"].tolist()
        if neutral:
            texts = [ov.neutralize_ids(t) for t in texts]
        return predict(model, tok, texts, tcfg.model_name, tcfg.max_len, bs)

    sha = state_dict_sha256(model)
    unk_nt = sets.eval_unknown[sets.eval_unknown["split"] != "test"]
    test_all = (
        None
        if ctx.smoke
        else ctx.df_all[ctx.df_all["split"] == "test"].sort_values("id").reset_index(drop=True)
    )
    outs: dict[str, dict[str, Any]] = {}
    outs["raw"] = {"tr": infer(sets.train), "cal": infer(sets.cal), "unk_nt": infer(unk_nt)}
    outs["neutral"] = {"cal": infer(sets.cal, True), "unk_nt": infer(unk_nt, True)}
    if test_all is not None:
        ti.log_test_call(
            ctx.P.test_log,
            f"{run_id}:{sha}",
            call_type,
            {
                "run_id": run_id,
                "n_rows": len(test_all),
                "role": "phase6a",
                "passes": ["raw", "neutral"],
            },
        )
        outs["raw"]["test"] = infer(test_all)
        outs["neutral"]["test"] = infer(test_all, True)

    def assemble(o: Mapping[str, Any]) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        unk_at = {i: [x[n] for x in o["unk_nt"]] for n, i in enumerate(unk_nt["id"])}
        if "test" in o:
            assert test_all is not None
            pos = {i: n for n, i in enumerate(test_all["id"])}
            ek = tuple(x[[pos[i] for i in sets.eval_known["id"]]] for x in o["test"])
            for i in sets.eval_unknown["id"]:
                if i in pos:
                    unk_at[i] = [x[pos[i]] for x in o["test"]]
        else:  # smoke: no test inference; the known side is the calibration rows
            ek = o["cal"]
        unk = tuple(np.stack([unk_at[i][j] for i in sets.eval_unknown["id"]]) for j in range(2))
        return ek, unk

    arrays: dict[str, np.ndarray] = {}
    for part, frame in (
        ("train", sets.train),
        ("cal", sets.cal),
        ("ek", sets.eval_known),
        ("unk", sets.eval_unknown),
    ):
        arrays[f"{part}_ids"] = frame["id"].to_numpy(dtype=str)
    arrays["train_logits"], arrays["train_features"] = outs["raw"]["tr"]
    for pre, mode in (("", "raw"), ("n_", "neutral")):
        ek, unk = assemble(outs[mode])
        for part, vals in (("cal", outs[mode]["cal"]), ("ek", ek), ("unk", unk)):
            arrays[f"{pre}{part}_logits"], arrays[f"{pre}{part}_features"] = vals
    return arrays, sha


@dataclass(frozen=True)
class RunDef:
    """One Track B run: where it lives and which holdout / seed it uses."""

    run_id: str
    ns: str  # "dev" | "confirm"
    seed: int  # run seed (path + model seed): DEV 0..2, CONFIRM 42, headline 42..44
    replicate: int  # soup replicate
    kind: str  # "loco" | "headline"
    holdout: tuple[str, ...]
    unit: str  # soup-pool unit: loco_<class> | headline
    call_type: str


def dev_runs(env: p4.Env, key: str) -> list[RunDef]:
    return [
        RunDef(
            ti.run_id_for(p4.candidate_for(env, key, None), "loco", s, c),
            "dev",
            s,
            s,
            "loco",
            (c,),
            f"loco_{c}",
            "phase6a_trackb",
        )
        for s in model_seeds(env)
        for c in env.dev_classes
    ]


def run_files(env: p4.Env, ns: str, key: str, seed: int, run_id: str) -> tuple[Path, Path]:
    """(npz path, run json path); the json is written last (the done marker)."""
    if ns == "dev":
        rr, oo = wdir(env, key, seed), odir(env, key, seed)
    else:
        cc = env.pcfg["confirm_report"]
        rr = env.P.results / cc["results_subdir"] / key / f"s{seed}"
        oo = env.P.outputs / cc["outputs_subdir"] / key / f"s{seed}"
    return oo / "trackb_arrays" / f"{run_id}.npz", rr / "trackb" / "runs" / f"{run_id}.json"


def run_done(env: p4.Env, key: str, run: RunDef) -> bool:
    npz, js = run_files(env, run.ns, key, run.seed, run.run_id)
    return npz.exists() and js.exists()


def save_run(
    env: p4.Env,
    key: str,
    run: RunDef,
    sets: HoldoutSets,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> None:
    npz, js = run_files(env, run.ns, key, run.seed, run.run_id)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **arrays)
    write_json(
        js,
        {
            "run_id": run.run_id,
            "weights": key,
            "kind": run.kind,
            "holdout": list(run.holdout),
            "seed": run.seed,
            "replicate": run.replicate,
            "labels": sets.labels,
            "call_type": run.call_type,
            "git_sha": git_sha(),
            "smoke": env.smoke,
            "audit": sets.audit,
            **meta,
        },
    )


# ================================================================================ DEV LOCO (std)
def run_loco_std(env: p4.Env, spec: TrainSpec, e_star: int, run: RunDef) -> None:
    """One LOCO / headline run of a plain model (ref, I4) stopped at e*; npz first, json last."""
    if run_done(env, spec.key, run):
        print(f"[trackb] {run.run_id}: done, skipping", flush=True)
        return
    ctx = env.ctx
    t0 = time.perf_counter()
    sets = build_holdout_sets(ctx.df_all, list(run.holdout), ctx.smoke)
    tcfg = train_cfg6(env, spec, run.seed, e_star)
    meta_run = {
        "run_id": run.run_id,
        "track": "phase6a",
        "kind": run.kind,
        "holdout": list(run.holdout),
        "n_labels": len(sets.labels),
        "candidate": spec.key,
        "git_sha": git_sha(),
        "smoke": ctx.smoke,
    }
    with lock(env, f"trackB {run.run_id}") as lk:
        res, model, tok = _train_model(
            tcfg, sets.train, sets.cal, meta_run, labels=sets.labels, hooks=spec.hooks()
        )
        try:
            arrays, sha = extract_arrays(ctx, model, tok, sets, tcfg, run.run_id, run.call_type)
        finally:
            model = None
            free_gpu()
    save_run(
        env,
        spec.key,
        run,
        sets,
        arrays,
        {
            "train_spec": spec.info(),
            "training": {
                "config": res.config,
                "epochs": res.epochs,
                "wall_clock_s": res.wall_clock_s,
                "peak_vram_mb": res.peak_vram_mb,
                "gpu_exclusive": res.gpu_exclusive,
                "gpu_foreign_seen": res.gpu_foreign_seen,
                "nan_detected": res.nan_detected,
                "fingerprint": sha,
                "gpu_lock_waited_s": lk.get("waited_s"),
                "wandb_url": res.wandb_url,
                "run_total_s": time.perf_counter() - t0,
            },
        },
    )
    print(
        f"[trackb] {run.run_id}: train {res.wall_clock_s:.0f}s total "
        f"{time.perf_counter() - t0:.0f}s gpu_exclusive={res.gpu_exclusive}",
        flush=True,
    )


def stage_trackb_std(env: p4.Env, spec: TrainSpec) -> None:
    e_star = e_star_of(env, spec.key)
    for run in dev_runs(env, spec.key):
        run_loco_std(env, spec, e_star, run)


# ==================================================================================== soup pools
def pool_curve_path(env: p4.Env, pool: Pool, m: int, fold: int) -> Path:
    return env.P.results / pool.dir_key / "curves" / f"m{m}_f{fold}.json"


def member_dir(env: p4.Env, pool: Pool, ns: str, unit: str) -> Path:
    return env.P.outputs / pool.dir_key / "members" / f"{ns}_{unit}"


def member_meta_path(env: p4.Env, pool: Pool, ns: str, unit: str, m: int) -> Path:
    return env.P.results / pool.dir_key / "members_meta" / f"{ns}_{unit}_m{m}.json"


def run_pool_curves(env: p4.Env, pool: Pool) -> None:
    """Member curves: every member trained on the inner 85% of each fold's train rows for the full
    schedule, per-epoch macro-F1 on the held-out fold (pool_size x folds files)."""
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    sc = env.pcfg["soup"]
    for f in env.folds:
        tr, ev = p4.fold_data(env, frame, f)
        fit, _ih = soup.inner_split(
            tr, f"cv_f{f}", float(sc["inner_holdout_frac"]), int(sc["inner_seed"])
        )
        for m in range(pool_size(env)):
            fp = pool_curve_path(env, pool, m, f)
            if fp.exists():
                continue
            tcfg = train_cfg6(env, pool.train, m, None)
            meta = {"run_id": f"p6a_poolcurve_{pool.key}_m{m}_f{f}", "fold": f}
            with rb.wandb_disabled(), lock(env, f"pool curve {pool.key} m{m} f{f}"):
                res = train_fold(tcfg, fit, ev, meta, hooks=pool.train.hooks())
            free_gpu()
            write_json(
                fp,
                {
                    "pool": pool.key,
                    "member": m,
                    "fold": f,
                    "n_fit": len(fit),
                    "macro_f1": [e["macro_f1"] for e in res.epochs],
                    "accuracy": [e["accuracy"] for e in res.epochs],
                    "wall_clock_s": res.wall_clock_s,
                    "peak_vram_mb": res.peak_vram_mb,
                    "nan_detected": res.nan_detected,
                    "gpu_exclusive": res.gpu_exclusive,
                    "config": res.config,
                    "train_spec": pool.train.info(),
                },
            )
            best = max(e["macro_f1"] for e in res.epochs)
            print(f"[pool-curves] {pool.key} m{m} f{f}: best-epoch F1 {best:.4f}")


def stage_pool_epoch(env: p4.Env, pool: Pool) -> None:
    """e* = argmax of the mean of ALL member curves (ties -> earliest); copied per weight key."""
    out = env.P.results / pool.dir_key / "epoch.json"
    if out.exists():
        print(f"[epoch] {pool.key}: epoch.json exists, skipping")
        return
    rows, keys = [], []
    for m in range(pool_size(env)):
        for f in env.folds:
            fp = pool_curve_path(env, pool, m, f)
            if not fp.exists():
                raise SystemExit(f"missing member curve {fp}: run the pool curves first")
            rows.append(json.loads(fp.read_text("utf-8"))["macro_f1"])
            keys.append((m, f))
    summ = p5.epoch_summary(np.array(rows, dtype=float))
    rec = {
        "candidate": pool.key,
        **summ,
        "n_member_curves": len(rows),
        "smoke": env.smoke,
        "git_sha": git_sha(),
    }
    write_json(out, rec)
    for wkey, _t in pool.types:
        write_json(epoch_path(env, wkey), rec | {"candidate": wkey, "pool": pool.key})
    print(f"[epoch] {pool.key}: e*={summ['e_star']} F1@e*={summ['f1_at_e_star']:.4f}")


def f1_of(
    model: Any, tok: Any, df: pd.DataFrame, labels: list[str], tcfg: TrainConfig, bs: int
) -> float:
    """Macro-F1 of a model on rows (label space = `labels`)."""
    logits, _ = predict(model, tok, df["text"].tolist(), tcfg.model_name, tcfg.max_len, bs)
    return macro_f1(pv.local_gold(df, labels), logits.argmax(1), len(labels))


def ensure_members(
    env: p4.Env,
    pool: Pool,
    ns: str,
    unit: str,
    fit: pd.DataFrame,
    ev: pd.DataFrame,
    ih: pd.DataFrame,
    labels: list[str],
    e_star: int,
) -> None:
    """Train + save (safetensors) every pool member of one unit, stopped at e*; each member's
    inner-holdout and eval-set macro-F1 go to its meta json (the done marker, written last)."""
    bs = int(env.pcfg["predict_batch_size"])
    for m in range(pool_size(env)):
        mp = member_meta_path(env, pool, ns, unit, m)
        path = member_dir(env, pool, ns, unit) / f"m{m}.safetensors"
        if mp.exists() and path.exists():
            continue
        tcfg = train_cfg6(env, pool.train, m, e_star)
        meta_run = {"run_id": f"p6a_member_{pool.key}_{ns}_{unit}_m{m}", "member": m}
        t0 = time.perf_counter()
        with rb.wandb_disabled(), lock(env, f"member {pool.key} {ns}_{unit} m{m}"):
            res, model, tok = _train_model(
                tcfg, fit, ev, meta_run, labels=labels, hooks=pool.train.hooks()
            )
            try:
                soup.save_member(model.state_dict(), path)
                fp = state_dict_sha256(model)
                f1_ih = f1_of(model, tok, ih, labels, tcfg, bs)
            finally:
                model = None
                free_gpu()
        write_json(
            mp,
            {
                "member": m,
                "unit": f"{ns}_{unit}",
                "n_fit": len(fit),
                "n_inner": len(ih),
                "f1_inner_holdout": f1_ih,
                "f1_eval_set": res.epochs[-1]["macro_f1"],
                "stop_epoch": e_star,
                "fingerprint": fp,
                "wall_clock_s": res.wall_clock_s,
                "total_s": time.perf_counter() - t0,
                "gpu_exclusive": res.gpu_exclusive,
                "nan_detected": res.nan_detected,
                "train_spec": pool.train.info(),
            },
        )
        print(
            f"[members] {pool.key} {ns}_{unit} m{m}: inner-F1 {f1_ih:.4f} "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )


@dataclass
class SoupBuild:
    """One built soup: state dict + provenance."""

    sd: dict[str, Any]
    info: dict[str, Any]


def build_soups(
    env: p4.Env,
    pool: Pool,
    ns: str,
    unit: str,
    replicate: int,
    skel: Any,
    tok: Any,
    ih: pd.DataFrame,
    labels: list[str],
    tcfg: TrainConfig,
    types: Sequence[str],
) -> dict[str, SoupBuild]:
    """Soup(s) of one replicate. Greedy decisions use the inner-holdout rows `ih` ONLY (a subset of
    the training rows; the evaluated fold / eval sets are never passed in)."""
    bs = int(env.pcfg["predict_batch_size"])
    members = soup_members(env, replicate)
    sds = [soup.load_member(member_dir(env, pool, ns, unit) / f"m{m}.safetensors") for m in members]
    metas = [
        json.loads(member_meta_path(env, pool, ns, unit, m).read_text("utf-8")) for m in members
    ]
    out: dict[str, SoupBuild] = {}
    for typ in types:
        if typ == "uniform":
            out[typ] = SoupBuild(
                soup.average_state_dicts(sds),
                {"type": typ, "members": members, "selected": members},
            )
            continue

        def eval_subset(idx: list[int]) -> float:
            skel.load_state_dict(soup.average_state_dicts([sds[j] for j in idx]))
            return f1_of(skel, tok, ih, labels, tcfg, bs)

        g = soup.greedy_soup([x["f1_inner_holdout"] for x in metas], eval_subset)
        sel_members = [members[j] for j in g["selected"]]
        out[typ] = SoupBuild(
            soup.average_state_dicts([sds[j] for j in g["selected"]]),
            {
                "type": typ,
                "members": members,
                "selected": sel_members,
                "order": [members[j] for j in g["order"]],
                "inner_f1_final": g["final_score"],
                "trace": [{**t, "member": members[t["member"]]} for t in g["trace"]],
            },
        )
    for b in out.values():
        b.info["member_inner_f1"] = {
            str(m): x["f1_inner_holdout"] for m, x in zip(members, metas, strict=True)
        }
        b.info["members_gpu_exclusive"] = all(x["gpu_exclusive"] for x in metas)
        b.info["members_wall_clock_s"] = float(sum(x["wall_clock_s"] for x in metas))
    return out


def make_skeleton(env: p4.Env, labels: list[str]) -> tuple[Any, Any]:
    import torch

    tok, model = build_model(env.fcfg["train"]["model_name"], labels, "eager")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model.to(dev)


def stage_pool_cv(env: p4.Env, pool: Pool) -> None:
    """Members per fold -> soups per replicate -> OOF parts -> oof csvs / folds.json per weights key
    and replicate. Members are deleted once every soup of the fold has been evaluated."""
    e_star = e_star_of(env, pool.types[0][0])
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    sc = env.pcfg["soup"]
    labels = rb._label_names()  # noqa: SLF001
    seeds = model_seeds(env)
    for k in env.folds:
        want = [(w, t, r) for w, t in pool.types for r in seeds]
        if all(
            parts_complete(p5.part_paths(odir(env, w, r) / "oof_parts", k)) for w, _t, r in want
        ):
            continue
        tr, ev = p4.fold_data(env, frame, k)
        fit, ih = soup.inner_split(
            tr, f"cv_f{k}", float(sc["inner_holdout_frac"]), int(sc["inner_seed"])
        )
        ensure_members(env, pool, "cv", f"f{k}", fit, ev, ih, labels, e_star)
        tcfg = train_cfg6(env, pool.train, 0, e_star)
        with lock(env, f"soups {pool.key} f{k}", 600):
            tok, skel = make_skeleton(env, labels)
            try:
                for r in seeds:
                    built = build_soups(
                        env,
                        pool,
                        "cv",
                        f"f{k}",
                        r,
                        skel,
                        tok,
                        ih,
                        labels,
                        tcfg,
                        sorted({t for _w, t in pool.types}),
                    )
                    for w, typ in pool.types:
                        pp = p5.part_paths(odir(env, w, r) / "oof_parts", k)
                        if parts_complete(pp):
                            continue
                        pp["meta"].parent.mkdir(parents=True, exist_ok=True)
                        skel.load_state_dict(built[typ].sd)
                        got = eval_fold_frames(env, skel, tok, ev, r, k)
                        write_parts(
                            pp,
                            got,
                            {
                                "fold": k,
                                "stop_epoch": e_star,
                                "soup": built[typ].info,
                                "train_spec": pool.train.info(),
                            },
                        )
            finally:
                skel = None
                free_gpu()
        delete_dir_guarded(member_dir(env, pool, "cv", f"f{k}"), "members")
        print(f"[soups] {pool.key} fold {k}: parts written, members deleted", flush=True)
    for w, _t in pool.types:
        for r in seeds:
            if not (wdir(env, w, r) / "folds.json").exists():
                finish_oof(env, w, r, e_star, {"pool": pool.key, "train_spec": pool.train.info()})


def stage_pool_loco(
    env: p4.Env, pool: Pool, ns: str = "dev", runs_by_type: Mapping[str, list[RunDef]] | None = None
) -> None:
    """LOCO (or confirm / headline) runs of every soup: per unit (class / headline holdout) train
    pool members on the inner 85% of the training rows, build the soups, extract the arrays, delete
    the members. runs_by_type: weights key -> its RunDefs (default: the DEV plan)."""
    e_star = e_star_of(env, pool.types[0][0])
    sc = env.pcfg["soup"]
    plan = runs_by_type or {w: dev_runs(env, w) for w, _t in pool.types}
    units: dict[str, list[tuple[str, str, RunDef]]] = {}
    for w, typ in pool.types:
        for run in plan.get(w, []):
            units.setdefault(run.unit, []).append((w, typ, run))
    ctx = env.ctx
    for unit, items in units.items():
        todo = [(w, t, r) for w, t, r in items if not run_done(env, w, r)]
        if not todo:
            continue
        holdout = items[0][2].holdout
        ns = items[0][2].ns
        sets = build_holdout_sets(ctx.df_all, list(holdout), ctx.smoke)
        fit, ih = soup.inner_split(
            sets.train, f"{ns}_{unit}", float(sc["inner_holdout_frac"]), int(sc["inner_seed"])
        )
        ensure_members(env, pool, ns, unit, fit, sets.cal, ih, sets.labels, e_star)
        with lock(env, f"soup runs {pool.key} {ns}_{unit}", 900):
            tok, skel = make_skeleton(env, sets.labels)
            try:
                reps = sorted({r.replicate for _w, _t, r in todo})
                for rep in reps:
                    types = sorted({t for _w, t, r in todo if r.replicate == rep})
                    rep_seed = next(r.seed for _w, _t, r in todo if r.replicate == rep)
                    tcfg = train_cfg6(env, pool.train, rep_seed, e_star)
                    t0 = time.perf_counter()
                    built = build_soups(
                        env, pool, ns, unit, rep, skel, tok, ih, sets.labels, tcfg, types
                    )
                    for w, typ, run in todo:
                        if run.replicate != rep:
                            continue
                        skel.load_state_dict(built[typ].sd)
                        arrays, sha = extract_arrays(
                            ctx, skel, tok, sets, tcfg, run.run_id, run.call_type
                        )
                        save_run(
                            env,
                            w,
                            run,
                            sets,
                            arrays,
                            {
                                "train_spec": pool.train.info(),
                                "soup": built[typ].info,
                                "training": {
                                    "config": dataclasses.asdict(tcfg),
                                    "wall_clock_s": built[typ].info["members_wall_clock_s"],
                                    "gpu_exclusive": built[typ].info["members_gpu_exclusive"],
                                    "fingerprint": sha,
                                    "run_total_s": time.perf_counter() - t0,
                                },
                            },
                        )
                        print(
                            f"[soup-run] {run.run_id}: {typ}, selected "
                            f"{built[typ].info['selected']}",
                            flush=True,
                        )
            finally:
                skel = None
                free_gpu()
        delete_dir_guarded(member_dir(env, pool, ns, unit), "members")


def stage_pool(env: p4.Env, pool: Pool) -> None:
    run_pool_curves(env, pool)
    stage_pool_epoch(env, pool)
    stage_pool_cv(env, pool)
    stage_pool_loco(env, pool)


# ==================================================================================== views
def view_name_dir(env: p4.Env, name: str, seed: int, mode: str, root: Path | None = None) -> Path:
    return pv.view_dir(root or env.P.results, name, seed, mode)


def build_views_for(
    env: p4.Env,
    name: str,
    weights: str,
    scorer: str,
    thr: str,
    runs: Sequence[RunDef],
    root: Path | None = None,
) -> int:
    """Write the (raw, neutral) view tables of candidate `name` for every run whose arrays exist.
    Returns the number of tables written. Idempotent (existing tables are kept)."""
    tb = env.pcfg["trackb"]
    n = 0
    for run in runs:
        npz_path, js = run_files(env, run.ns, weights, run.seed, run.run_id)
        if not (npz_path.exists() and js.exists()):
            continue
        stem = f"{name}_{run.run_id[len(weights) + 1 :]}"  # <view>_loco_<class>_s<seed>
        todo = [
            m
            for m in pv.MODES
            if not (view_name_dir(env, name, run.seed, m, root) / f"{stem}.csv").exists()
        ]
        if not todo:
            continue
        A = dict(np.load(npz_path))
        sets = build_holdout_sets(env.ctx.df_all, list(run.holdout), env.ctx.smoke)
        fit = ov.fit_feature_scorer(
            scorer, A["train_features"], pv.local_gold(sets.train, sets.labels), len(sets.labels)
        )
        for mode in todo:
            table, info = pv.build_view(
                sets,
                A,
                scorer,
                thr,
                mode,
                int(tb["per_class_min_rows"]),
                int(tb["crossfit_folds"]),
                int(tb["crossfit_seed"]),
                fit,
            )
            d = view_name_dir(env, name, run.seed, mode, root)
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / f"{stem}.thr.json", {**info, "weights": weights, "run_id": run.run_id})
            ti._save_csv(table, d / f"{stem}.csv")  # noqa: SLF001 - csv last (done marker)
            n += 1
    return n


def weights_available(env: p4.Env, key: str) -> bool:
    return all(run_done(env, key, r) for r in dev_runs(env, key))


def all_view_defs(env: p4.Env) -> dict[str, tuple[str, str, str]]:
    """name -> (weights, scorer, thr) of every view that can be built now: ref, the formed
    candidates, and the combo helper views."""
    out: dict[str, tuple[str, str, str]] = {REF: (REF, "maha_ft", "global")}
    for spec in candidate_specs(env):
        if spec["formed"]:
            out[spec["key"]] = (spec["weights"], spec["scorer"], spec["thr"])
    ch = sel.read_choice(env.P.results)
    if ch and ch.get("i3_type"):
        for sc_ in ("i1a", "i1b"):
            out[f"h_i3_{sc_}"] = (ch["i3_type"], sc_, "global")
    return out


def stage_views(env: p4.Env) -> None:
    for name, (w, scorer, thr) in all_view_defs(env).items():
        if not weights_available(env, w):
            continue
        n = build_views_for(env, name, w, scorer, thr, dev_runs(env, w))
        if n:
            print(f"[views] {name} ({w}/{scorer}/{thr}): wrote {n} tables")


# ==================================================================================== combos
def cand_cv_f1_and_flip(env: p4.Env, key: str) -> tuple[float, float]:
    """(mean CV macro-F1, pooled neutral-swap flip rate) of a weight set from its saved OOF."""
    seeds = model_seeds(env)
    sc_ = sel.SelCtx(
        env.P.results,
        env.pcfg,
        env.dev_classes,
        seeds,
        env.folds,
        list(env.ctx.cfg["ood"]["safe_labels"]),
        env.n_boot,
        env.smoke,
    )
    cv = sel.load_cv(sc_, key)
    return float(cv.f1.mean()), p5.flip_rate_point(cv.flips, cv.inst)


def dev_neutral_rej95(env: p4.Env, name: str) -> float:
    """Mean over DEV classes and seeds of the ID-neutral strict rejection@95 of a view (point)."""
    vals = []
    for s in model_seeds(env):
        for c in env.dev_classes:
            t = pd.read_csv(view_name_dir(env, name, s, "neutral") / f"{name}_loco_{c}_s{s}.csv")
            ev = t[(t["set"] == "eval") & t["is_unknown"].astype(bool)]
            vals.append(float((ev["margin95"] < 0).mean()))
    return float(np.mean(vals))


def choose_soup_type(f1_uniform: float, f1_greedy: float) -> str:
    """The I3 variant with the higher standalone CV macro-F1 (ties -> uniform)."""
    return "i3g" if f1_greedy > f1_uniform else "i3u"


def choose_combos(
    i3_type: str,
    f1_ref: float,
    i4: Mapping[str, Mapping[str, float]],
    lam_of: Mapping[str, float],
    a_max_drop: float,
    rej_i1: Mapping[str, float],
    rej_i6: Mapping[str, float],
) -> dict[str, Any]:
    """Pure pick rules of the pre-listed combos (module notes, item 6).

    i4[k] = {"f1", "flip"} of the standalone I4 variants; lambda for the I4 combos = the variant
    eligible on axis (a) (point estimate) with the lower flip rate (ties: smaller lambda), none
    eligible -> the I4 combos are not formed. rej_i1 / rej_i6: ID-neutral DEV rejection@95 of the I1
    scorers on the chosen I3 soup / of the standalone I6 variants; the higher wins, ties go to the
    first (i1a / i6a)."""
    elig = [k for k in lam_of if p5.eligible_drop(i4[k]["f1"] - f1_ref, a_max_drop)]
    i4_pick = min(elig, key=lambda k: (i4[k]["flip"], lam_of[k])) if elig else None
    i1 = "i1a" if rej_i1["i1a"] >= rej_i1["i1b"] else "i1b"
    i6 = "i6a" if rej_i6["i6a"] >= rej_i6["i6b"] else "i6b"
    no_i4 = "no I4 variant is eligible on axis (a)"
    formed = i4_pick is not None

    def combo(weights: str, scorer: str, thr: str, ok: bool) -> dict[str, Any]:
        return {
            "formed": ok,
            "weights": weights,
            "scorer": scorer,
            "thr": thr,
            "reason": None if ok else no_i4,
        }

    return {
        "i4_standalone": {k: dict(v) | {"eligible_a": k in elig} for k, v in i4.items()},
        "i4_pick": i4_pick,
        "i3_type": i3_type,
        "soup_type": "greedy" if i3_type == "i3g" else "uniform",
        "i1_scorer": i1,
        "i1_neutral_rej95_on_i3": dict(rej_i1),
        "i6_variant": i6,
        "i6_neutral_rej95_standalone": dict(rej_i6),
        "combos": {
            "i3i4": combo("i3i4", "maha_ft", "global", formed),
            "i3i1": combo(i3_type, i1, "global", True),
            "i3i4i1": combo("i3i4", i1, "global", formed),
            "i3i4i1i6": combo("i3i4", i1, i6, formed),
        },
    }


def pick_combos(env: p4.Env) -> dict[str, Any]:
    """Gather the standalone numbers and apply `choose_combos`."""
    f1_ref, _ = cand_cv_f1_and_flip(env, REF)
    lam_of = {k: float(v) for k, v in env.pcfg["i4"]["lambdas"].items()}
    i4 = {}
    for k in lam_of:
        f1, flip = cand_cv_f1_and_flip(env, k)
        i4[k] = {"f1": f1, "flip": flip}
    f1u, _ = cand_cv_f1_and_flip(env, "i3u")
    f1g, _ = cand_cv_f1_and_flip(env, "i3g")
    i3_type = choose_soup_type(f1u, f1g)
    # I1 scorer on the chosen I3 soup (helper views), I6 variant from the standalone I6 candidates.
    for name, scorer in (("h_i3_i1a", "i1a"), ("h_i3_i1b", "i1b")):
        build_views_for(env, name, i3_type, scorer, "global", dev_runs(env, i3_type))
    ch = choose_combos(
        i3_type,
        f1_ref,
        i4,
        lam_of,
        float(env.pcfg["thresholds"]["a_max_drop"]),
        {"i1a": dev_neutral_rej95(env, "h_i3_i1a"), "i1b": dev_neutral_rej95(env, "h_i3_i1b")},
        {"i6a": dev_neutral_rej95(env, "i6a"), "i6b": dev_neutral_rej95(env, "i6b")},
    )
    ch["i3_standalone_cv_f1"] = {"i3u": f1u, "i3g": f1g}
    ch |= {"git_sha": git_sha(), "smoke": env.smoke}
    return ch


def stage_combos(env: p4.Env) -> None:
    for w in ("i3u", "i3g", REF, "i4a", "i4b"):
        if (
            not weights_available(env, w)
            or not (wdir(env, w, model_seeds(env)[-1]) / "folds.json").exists()
        ):
            raise SystemExit(f"combos need the standalone results of {w}: run its stage first")
    stage_views(env)
    path = env.P.results / sel.CHOICE_FILE
    ch = p4.read_json(path)
    if ch is None:
        ch = pick_combos(env)
        write_json(path, ch)
        print(
            f"[combos] choice written: I4 {ch['i4_pick']}, soup {ch['soup_type']}, "
            f"I1 {ch['i1_scorer']}, I6 {ch['i6_variant']}"
        )
    else:
        print(f"[combos] {path} exists: keeping the recorded choice")
    if ch["i4_pick"] is not None:
        lam = float(env.pcfg["i4"]["lambdas"][ch["i4_pick"]])
        stage_pool(env, i3i4_pool(env, lam, ch["soup_type"]))
    stage_views(env)


# ===================================================================================== std stage
def stage_std(env: p4.Env, keys: Sequence[str]) -> None:
    specs = std_specs(env)
    for k in keys:
        spec = specs[k]
        run_curves_std(env, spec)
        stage_epoch_std(env, spec)
        stage_folds_std(env, spec)
        stage_trackb_std(env, spec)
        stage_views(env)


def stage_status(env: p4.Env) -> None:
    """What exists on disk (no GPU, no writes)."""
    for k in (REF, *env.pcfg["i4"]["lambdas"], "i3u", "i3g", "i3i4"):
        n_oof = sum((wdir(env, k, s) / "folds.json").exists() for s in model_seeds(env))
        n_run = sum(run_done(env, k, r) for r in dev_runs(env, k))
        ep = p4.read_json(epoch_path(env, k))
        print(
            f"[status] {k:5s} e*={ep['e_star'] if ep else '-'} OOF seeds {n_oof}/"
            f"{len(model_seeds(env))} DEV runs {n_run}/{len(dev_runs(env, k))}"
        )
    print(
        f"[status] combos choice: {'yes' if (env.P.results / sel.CHOICE_FILE).exists() else 'no'}"
        f"; selection: {'yes' if (env.P.results / 'selection.json').exists() else 'no'}"
    )


def sel_ctx(env: p4.Env) -> sel.SelCtx:
    return sel.SelCtx(
        env.P.results,
        env.pcfg,
        env.dev_classes,
        model_seeds(env),
        env.folds,
        list(env.ctx.cfg["ood"]["safe_labels"]),
        env.n_boot,
        env.smoke,
    )


def dir_size_gb(path: Path) -> float:
    """Total size of the files under `path` in GB (0 if it does not exist)."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def report_disk(env: p4.Env) -> dict[str, float]:
    """Print (and return) the 6a output sizes and the drive's free space (disk usage report)."""
    out = {
        "outputs_gb": dir_size_gb(env.P.outputs),
        "results_gb": dir_size_gb(env.P.results),
        "free_gb": shutil.disk_usage(env.P.outputs.anchor or ".").free / 1e9,
    }
    members = sum(dir_size_gb(d) for d in env.P.outputs.glob("pool_*/members"))
    print(
        f"[disk] {env.P.outputs}: {out['outputs_gb']:.2f} GB (soup members in flight "
        f"{members:.2f} GB), {env.P.results}: {out['results_gb']:.3f} GB, drive free "
        f"{out['free_gb']:.1f} GB"
    )
    return out


# ======================================================================================== main
def run_stage(env: p4.Env, stage: str, cand: str | None, smoke_cand: str | None = None) -> None:
    std_keys = [cand] if cand in std_specs(env) else list(std_specs(env))
    if stage == "reuse_check":
        print_reuse_report(check_ref_reuse(env, std_specs(env)[REF]))
    elif stage == "ref":
        stage_std(env, [REF])
    elif stage == "i4":
        stage_std(env, [k for k in std_keys if k != REF])
    elif stage == "i3":
        stage_pool(env, i3_pool(env))
        stage_views(env)
    elif stage == "combos":
        stage_combos(env)
    elif stage == "views":
        stage_views(env)
    elif stage == "select":
        sel.stage_select(sel_ctx(env), env.P.results)
    elif stage == "status":
        stage_status(env)
    elif stage in ("confirm_report", "final_retrain"):
        from intent_router import phase6a_post as post  # lazy: it imports this module

        if stage == "confirm_report":
            post.stage_confirm_report(env)
        else:
            post.stage_final_retrain(env, smoke_cand)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Open-set improvement round")
    ap.add_argument("--config", default="configs/phase6a.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--cand", default=None, help="stage i4 only: i4a or i4b")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny: 2 folds, 1st DEV class, 1 epoch, stand-in inputs, pool of 5; "
        "writes only under outputs/phase6a_smoke",
    )
    ap.add_argument(
        "--smoke-cand",
        default=None,
        help="smoke final_retrain only: candidate to retrain regardless of the selection",
    )
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke_cand and not args.smoke:
        raise SystemExit("--smoke-cand needs --smoke")
    env = make_env(args)
    run_stage(env, args.stage, args.cand, args.smoke_cand)
    report_disk(env)
    gc.collect()


if __name__ == "__main__":
    main()
