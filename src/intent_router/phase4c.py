"""Robustness and open-set training fixes (pre-registered).

Candidates: current (Open-set scorer comparison / 4a config), A1 (ID-prefix randomisation), A2
(outlier exposure),
A3 (translation + noise augmentation) and comb (all adopted factors). Entry point:

    python -m intent_router.phase4c --config configs/phase4c.yaml --stage <stage>

Every stage is resumable (a result file's existence is the "done" marker) and writes only ids and
numbers to results/phase4c/ (never dataset or synthetic text).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intent_router import final as final_mod
from intent_router import robustness as rb
from intent_router import trackb_improve as ti
from intent_router.cv import fold_split, load_cv_frame
from intent_router.evaluate import git_sha, softmax, threshold_at_retention
from intent_router.models import predict, state_dict_sha256
from intent_router.ood import build_holdout_sets, l2_normalize
from intent_router.stats import accuracy, macro_f1
from intent_router.trackb import gpu_section
from intent_router.train import TrainConfig, train_fold
from intent_router.train import train_model as _train_model

FACTORS = ("a1", "a2", "a3")
CANDIDATES = ("current", *FACTORS, "comb")
PIPELINE = ("guard", "folds", "trackb")
STAGES = (
    *[f"{p}_{c}" for p in PIPELINE for c in CANDIDATES],
    *[f"cand_{c}" for c in CANDIDATES],
    "decide_factors",
    "comb",
    "confirm_comb",
    "final_retrain",
)
NO_FACTORS_MSG = "no factors adopted; current model stays"
write_json = rb.write_json


# ================================================================================ recipes
@dataclass(frozen=True)
class Recipe:
    """Which factors a candidate trains with. key doubles as the artifact directory name."""

    key: str
    a1: bool = False
    a2: bool = False
    a3: bool = False

    @property
    def factors(self) -> tuple[str, ...]:
        return tuple(f for f in FACTORS if getattr(self, f))


def factor_recipe(key: str) -> Recipe:
    """Recipes of the four fixed candidates (comb is resolved from factor_decisions.json)."""
    if key == "current":
        return Recipe("current")
    if key in FACTORS:
        return Recipe(key, **{key: True})
    raise KeyError(f"{key!r} is not a fixed candidate; comb is resolved by comb_recipe")


def comb_recipe(adopted: Sequence[str]) -> Recipe:
    """The C-comb recipe. One adopted factor => identical to that factor alone, so its artifacts
    (same deterministic training) are reused under the factor's key instead of retraining."""
    if not adopted:
        raise ValueError(NO_FACTORS_MSG)
    flags = {f: f in adopted for f in FACTORS}
    return Recipe(adopted[0] if len(adopted) == 1 else "comb", **flags)


def recipe_overrides(rec: Recipe, pcfg: Mapping[str, Any]) -> dict[str, Any]:
    """TrainConfig field overrides of a recipe (A3 acts on the data, not the config)."""
    out: dict[str, Any] = {}
    if rec.a1:
        out["id_randomize_p"] = float(pcfg["factors"]["a1"]["id_randomize_p"])
    if rec.a2:
        out["oe_lambda"] = float(pcfg["factors"]["a2"]["oe_lambda"])
        out["oe_batch_size"] = int(pcfg["factors"]["a2"]["oe_batch_size"])
    return out


# =================================================================================== inputs
@dataclass
class Inputs:
    """Files written by phase4c_data.py (or smoke stand-ins)."""

    syn: pd.DataFrame
    syn_emb: np.ndarray
    aug: pd.DataFrame
    eval_mt: pd.DataFrame


def validate_synthetic(syn: pd.DataFrame, emb: np.ndarray) -> None:
    """Row-aligned embeddings, unique ids, L2-normalised rows."""
    need = {"syn_id", "text", "lang", "kind", "batch_seed"}
    if need - set(syn.columns):
        raise ValueError(f"synthetic_oe.csv is missing columns {sorted(need - set(syn.columns))}")
    if emb.ndim != 2 or emb.shape[0] != len(syn):
        raise ValueError(f"synthetic_oe_emb.npy shape {emb.shape} != {len(syn)} rows")
    if not syn["syn_id"].is_unique:
        raise ValueError("synthetic_oe.csv syn_id is not unique")
    if not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3):
        raise ValueError("synthetic_oe_emb.npy rows are not L2-normalised")


def validate_aug(
    aug: pd.DataFrame, cv_ids: Sequence[str], kinds: Sequence[str], full: bool
) -> None:
    """src_id within the CV (train+val) ids only; with full=True every id has every kind."""
    need = {"src_id", "kind", "lang", "text"}
    if need - set(aug.columns):
        raise ValueError(f"a3_aug.csv is missing columns {sorted(need - set(aug.columns))}")
    if not set(aug["kind"]) <= set(kinds):
        raise ValueError(f"a3_aug.csv kinds {sorted(set(aug['kind']))} not within {list(kinds)}")
    stray = set(aug["src_id"].astype(str)) - set(cv_ids)
    if stray:
        raise AssertionError(f"a3_aug.csv has {len(stray)} src_ids outside train+val (test leak?)")
    if aug.duplicated(["src_id", "kind"]).any():
        raise ValueError("a3_aug.csv has more than one copy per (src_id, kind)")
    if full:
        have = aug.groupby("kind")["src_id"].nunique().to_dict()
        short = {k: len(cv_ids) - have.get(k, 0) for k in kinds if have.get(k, 0) != len(cv_ids)}
        if short:
            raise ValueError(f"a3_aug.csv does not cover all {len(cv_ids)} train+val ids: {short}")


def validate_eval_mt(mt: pd.DataFrame, en_ids: Sequence[str], langs: Sequence[str]) -> None:
    """Rows only for English train+val ids and the 4 target languages; kept is boolean."""
    need = {"id", "lang", "text", "system", "labse_cos", "kept"}
    if need - set(mt.columns):
        raise ValueError(f"eval_mt.csv is missing columns {sorted(need - set(mt.columns))}")
    if not set(mt["lang"]) <= set(langs):
        raise ValueError(
            f"eval_mt.csv languages {sorted(set(mt['lang']))} not within {list(langs)}"
        )
    stray = set(mt["id"].astype(str)) - set(en_ids)
    if stray:
        raise AssertionError(
            f"eval_mt.csv has {len(stray)} ids that are not English train+val rows"
        )
    if mt.duplicated(["id", "lang"]).any():
        raise ValueError("eval_mt.csv has more than one row per (id, lang)")
    if mt["kept"].dtype != bool:
        raise ValueError("eval_mt.csv `kept` must be a boolean column")


def english_cv_ids(fcfg: Mapping[str, Any], cv_frame: pd.DataFrame) -> list[str]:
    """English (EDA primary language) train+val ids."""
    eda = pd.read_csv(fcfg["analysis"]["eda_rows_path"])[["id", "lang"]]
    en = set(eda.loc[eda["lang"] == "en", "id"])
    return sorted(i for i in cv_frame["id"] if i in en)


def read_text_csv(path: Path) -> pd.DataFrame:
    """read_csv with `text` as str; an empty cell stays "" (a failed MT output), never NaN."""
    df = pd.read_csv(path, dtype={"text": str})
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    return df


def load_inputs(
    pcfg: Mapping[str, Any], fcfg: Mapping[str, Any], smoke: bool, root: Path
) -> Inputs:
    """Read + validate the three input files; smoke runs build small stand-ins under root/data."""
    cv = load_cv_frame(fcfg["data_path"], fcfg["splits_path"])
    cv_ids = cv["id"].astype(str).tolist()
    en_ids = english_cv_ids(fcfg, cv)
    spec = pcfg["inputs"]
    if smoke:
        paths = make_smoke_inputs(cv, en_ids, root / "data")
    else:
        paths = {
            k: Path(spec[k]) for k in ("synthetic_oe", "synthetic_oe_emb", "a3_aug", "eval_mt")
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SystemExit(f"phase4c inputs missing (run phase4c_data first): {missing}")
    syn = read_text_csv(paths["synthetic_oe"])
    emb = np.load(paths["synthetic_oe_emb"])
    validate_synthetic(syn, emb)
    aug = read_text_csv(paths["a3_aug"])
    validate_aug(aug, cv_ids, pcfg["factors"]["a3"]["kinds"], full=True)
    if not smoke and len(cv_ids) != int(spec["expected_cv_rows"]):
        raise AssertionError(f"{len(cv_ids)} train+val rows != {spec['expected_cv_rows']}")
    mt = read_text_csv(paths["eval_mt"])
    if "labse_cos" not in mt.columns and "cos" in mt.columns:  # phase4c_data's column name
        mt = mt.rename(columns={"cos": "labse_cos"})
    if "kept" in mt.columns and mt["kept"].dtype != bool:  # csv booleans may arrive as strings
        mt["kept"] = mt["kept"].astype(str).str.lower().eq("true")
    validate_eval_mt(mt, en_ids, pcfg["translation"]["langs"])
    if not smoke and len(en_ids) != int(spec["expected_en_rows"]):
        raise AssertionError(f"{len(en_ids)} English rows != {spec['expected_en_rows']}")
    if not smoke and set(mt["id"]) != set(en_ids):
        raise AssertionError("eval_mt.csv does not cover every English train+val id")
    return Inputs(syn, emb.astype(np.float64), aug, mt)


def make_smoke_inputs(cv: pd.DataFrame, en_ids: Sequence[str], out: Path) -> dict[str, Path]:
    """Deterministic stand-ins (seed 42) with the real files' schema, for smoke runs only."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_syn = 48
    syn = pd.DataFrame(
        {
            "syn_id": [f"syn-{i:04d}" for i in range(n_syn)],
            "text": [
                f"standin outlier request number {i} about something unrelated"
                for i in range(n_syn)
            ],
            "lang": "en",
            "kind": ["logistics" if i % 4 else "generic" for i in range(n_syn)],
            "batch_seed": 42,
        }
    )
    emb = rng.normal(size=(n_syn, 768))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    aug_rows = []
    for rid, text in zip(cv["id"], cv["text"], strict=True):
        aug_rows.append((rid, "mt", "es", f"[es] {text}"))
        aug_rows.append((rid, "noise", "en", rb.apply_noise(text, "char_mixed_5", rid, 42)))
    aug = pd.DataFrame(aug_rows, columns=["src_id", "kind", "lang", "text"])
    text_of = dict(zip(cv["id"], cv["text"], strict=True))
    mt_rows = []
    for rid in en_ids:
        for lang in ("es", "fr", "de", "zh"):
            cos = float(rng.uniform(0.6, 0.95))
            mt_rows.append((rid, lang, f"[{lang}] {text_of[rid]}", "standin", cos, cos >= 0.8))
    mt = pd.DataFrame(mt_rows, columns=["id", "lang", "text", "system", "labse_cos", "kept"])
    paths = {
        "synthetic_oe": out / "synthetic_oe.csv",
        "synthetic_oe_emb": out / "synthetic_oe_emb.npy",
        "a3_aug": out / "a3_aug.csv",
        "eval_mt": out / "eval_mt.csv",
    }
    syn.to_csv(paths["synthetic_oe"], index=False)
    np.save(paths["synthetic_oe_emb"], emb)
    aug.to_csv(paths["a3_aug"], index=False)
    mt.to_csv(paths["eval_mt"], index=False)
    return paths


# ================================================================== training-time extras
@dataclass
class Extras:
    """What train_model gets besides the plain training rows."""

    oe_texts: list[str] | None
    extra_rows: pd.DataFrame | None
    info: dict[str, Any]


def a3_rows_for(
    aug: pd.DataFrame,
    train_df: pd.DataFrame,
    forbid_src_ids: set[str],
) -> tuple[pd.DataFrame, int]:
    """A3 augmented rows of the TRAINING ids only, as (id, text, label, src_id) rows.

    A copy whose source row is not in train_df is excluded (held-out fold rows, other splits).
    Raises if any kept copy's source is in forbid_src_ids (the held-out / evaluation ids).
    Returns (rows, number of aug rows excluded because their source is not a training row).
    """
    tr_ids = train_df["id"].astype(str)
    keep = aug["src_id"].astype(str).isin(set(tr_ids))
    sub = aug[keep]
    leaked = set(sub["src_id"].astype(str)) & forbid_src_ids
    if leaked:
        raise AssertionError(
            f"A3 copies of held-out rows would enter training: {sorted(leaked)[:5]}"
        )
    label_of = dict(zip(tr_ids, train_df["label"], strict=True))
    src = sub["src_id"].astype(str)
    rows = pd.DataFrame(
        {
            "id": [f"{s}|{k}" for s, k in zip(src, sub["kind"], strict=True)],
            "text": sub["text"].to_numpy(),
            "label": [label_of[s] for s in src],
            "src_id": src.to_numpy(),
        }
    )
    return rows, int((~keep).sum())


def oe_keep_mask(syn_emb: np.ndarray, heldout_emb: np.ndarray, max_cos: float) -> np.ndarray:
    """A2 leakage control: False for synthetic items with cosine > max_cos to ANY held-out row."""
    sims = l2_normalize(syn_emb) @ l2_normalize(heldout_emb).T
    return ~(sims.max(axis=1) > max_cos)


def build_extras(
    rec: Recipe,
    inp: Inputs | None,
    pcfg: Mapping[str, Any],
    train_df: pd.DataFrame,
    forbid_src_ids: set[str],
    heldout_emb: np.ndarray | None = None,
) -> Extras:
    """Synthetic OE pool and/or A3 rows for one training run (None where the factor is off)."""
    info: dict[str, Any] = {"factors": list(rec.factors)}
    if rec.factors and inp is None:
        raise ValueError("this recipe needs the Robustness and open-set training fixes input files")
    oe: list[str] | None = None
    extra: pd.DataFrame | None = None
    if rec.a2:
        keep = np.ones(len(inp.syn), dtype=bool)
        if heldout_emb is not None:
            keep = oe_keep_mask(
                inp.syn_emb, heldout_emb, float(pcfg["factors"]["a2"]["holdout_cos_drop"])
            )
        oe = inp.syn.loc[keep, "text"].tolist()
        info["oe"] = {"n_pool": len(inp.syn), "n_dropped": int((~keep).sum()), "n_kept": len(oe),
                      "filtered_by_holdout": heldout_emb is not None}  # fmt: skip
        if not oe:
            raise ValueError("every synthetic OE item was dropped by the holdout filter")
    if rec.a3:
        extra, n_excl = a3_rows_for(inp.aug, train_df, forbid_src_ids)
        info["a3"] = {"n_rows": len(extra), "n_excluded_not_train": n_excl}
    return Extras(oe, extra, info)


@contextlib.contextmanager
def patched_training(
    modules: Sequence[Any],
    build: Callable[[pd.DataFrame, pd.DataFrame], Extras],
    record: dict[str, Any],
) -> Iterator[None]:
    """Make `train_model` of the given modules pass this run's extras (OE pool / A3 rows).

    trackb_improve.train_and_extract and final._run_one call train_model(cfg, train, eval, meta,
    labels) with no hook for the Robustness and open-set training fixes extras; rather than editing
    those modules the name they
    resolve is wrapped for the duration of one call chain. `build(train_df, eval_df)` builds the
    extras from the actual training rows; its info lands in `record`.
    """

    def wrapped(
        cfg: Any,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        run_meta: dict[str, Any],
        labels: list[str] | None = None,
    ) -> Any:
        ex = build(train_df, eval_df)
        record.update(ex.info)
        return _train_model(
            cfg,
            train_df,
            eval_df,
            run_meta,
            labels,
            oe_texts=ex.oe_texts,
            extra_train_rows=ex.extra_rows,
        )

    saved = [(m, m.train_model) for m in modules]
    for m, _ in saved:
        m.train_model = wrapped
    try:
        yield
    finally:
        for m, orig in saved:
            m.train_model = orig


# ================================================================================ context
@dataclass(frozen=True)
class P4Paths:
    """Output locations; smoke runs are fully redirected under outputs/phase4c_smoke/."""

    results: Path
    outputs: Path
    smoke: bool

    def guard(self, key: str) -> Path:
        return self.results / "guard" / f"{key}.json"

    def cand(self, key: str) -> Path:
        return self.results / key


@dataclass
class Env:
    """Everything a stage needs."""

    pcfg: dict[str, Any]
    P: P4Paths
    ctx: ti.Ctx
    smoke: bool
    smoke_adopt: list[str] | None = None
    _inputs: Inputs | None = None

    @property
    def fcfg(self) -> dict[str, Any]:
        return self.ctx.fcfg

    @property
    def inputs(self) -> Inputs:
        if self._inputs is None:
            self._inputs = load_inputs(self.pcfg, self.fcfg, self.smoke, self.P.outputs.parent)
        return self._inputs

    @property
    def n_boot(self) -> int:
        return 200 if self.smoke else int(self.pcfg["bootstrap"]["n_resamples"])

    @property
    def dev_classes(self) -> list[str]:
        dev = list(self.ctx.cfg["dev_classes"])
        return dev[:1] if self.smoke else dev

    @property
    def folds(self) -> list[int]:
        return [0] if self.smoke else list(range(int(self.pcfg["guard"]["n_folds"])))

    def ctx_current(self) -> ti.Ctx:
        """
        Context whose score tables are the Open-set scorer comparison base runs (current method).
        """
        paths = ti.Paths(
            Path(self.pcfg["current"]["trackb_dir"]), self.P.outputs / "unused",
            self.P.results / "unused.jsonl", self.smoke,
        )  # fmt: skip
        return replace(self.ctx, P=paths)

    def reuses_phase3b_for_current(self) -> bool:
        """
        Real runs reuse the Open-set scorer comparison DEV score tables; smoke trains `current` like
        the rest.
        """
        return not self.smoke


def make_env(args: argparse.Namespace) -> Env:
    pcfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke = bool(args.smoke)
    if smoke:
        root = Path(getattr(args, "smoke_root", "outputs/phase4c_smoke"))  # 4e redirects its smoke
        P = P4Paths(root / "results", root / "outputs", True)
    else:
        P = P4Paths(Path(pcfg["results_dir"]), Path(pcfg["outputs_dir"]), False)
    no_wandb = bool(args.no_wandb) or not pcfg["wandb"]["enabled"]
    ti_args = argparse.Namespace(
        config=pcfg["trackb_improve_config"], poll_s=float(pcfg["gpu_lock_poll_s"]),
        smoke=smoke, no_wandb=no_wandb,
    )  # fmt: skip
    ctx = ti.load_ctx(ti_args)
    ctx.cfg["gpu_lock_expected_s"] = int(pcfg["gpu_lock_expected_s"])
    paths = ti.Paths(
        P.results, P.outputs / "trackb_arrays", P.results / "test_inference_log.jsonl", smoke
    )
    ctx = replace(ctx, P=paths)
    g = pcfg["guard"]
    if abs(float(g["threshold"]) - (float(g["comparator"]) - float(g["tolerance"]))) > 1e-9:
        raise AssertionError("guard threshold != comparator - tolerance")
    adopt = [x for x in str(args.smoke_adopt).split(",") if x] if args.smoke_adopt else None
    if adopt and (not smoke or not set(adopt) <= set(FACTORS)):
        raise SystemExit("--smoke-adopt needs --smoke and factors within a1,a2,a3")
    return Env(pcfg, P, ctx, smoke, adopt)


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def group_for(env: Env, key: str) -> str:
    return f"{env.pcfg['wandb']['group_prefix']}-{key}"


def candidate_for(env: Env, key: str, stop_epoch: int | None) -> ti.Candidate:
    """trackb_improve Candidate (run-id prefix `<key>_`) with the base model recipe."""
    t = env.fcfg["train"]
    return ti.Candidate(
        key,
        f"{key}_",
        "base",
        t["model_name"],
        float(t["lr"]),
        None,
        group_for(env, key),
        stop_epoch,
    )


def train_cfg_for(env: Env, rec: Recipe, seed: int, stop_epoch: int | None) -> TrainConfig:
    """final.yaml's train block + the recipe's overrides; stop_epoch None => the full schedule."""
    cand = candidate_for(env, rec.key, stop_epoch)
    tcfg = ti.train_config(env.ctx, cand, seed)
    tcfg = replace(tcfg, **recipe_overrides(rec, env.pcfg))
    if stop_epoch is None:
        tcfg = replace(tcfg, stop_epoch=1 if env.smoke else None)
    return tcfg


def deployed_epoch(env: Env, rec: Recipe) -> int:
    """
    e*: argmax of the 5-fold mean guard curve (`current` ships epoch 9, as in Open-set scorer
    comparison / 4a).
    """
    if env.smoke:
        return 1
    if rec.key == "current":
        return int(env.pcfg["current"]["deployed_epoch"])
    g = read_json(env.P.guard(rec.key))
    if g is None:
        raise SystemExit(f"no guard summary for {rec.key}: run --stage guard_{rec.key} first")
    return int(g["deployed_epoch"])


def _free_gpu() -> None:
    ti._free_gpu()  # noqa: SLF001 - same helper trackb_improve uses between runs


def cv_forbid(ev: pd.DataFrame) -> set[str]:
    return set(ev["id"].astype(str))


def fold_data(env: Env, frame: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, held-out) rows of one s0 fold; smoke runs use 64 train and 40 held-out rows."""
    tr, ev = fold_split(frame, int(env.pcfg["guard"]["fold_seed_idx"]), fold)
    if env.smoke:
        tr = tr.sample(64, random_state=42).reset_index(drop=True)
        ev = ev.iloc[:40].reset_index(drop=True)
    return tr, ev


# ==================================================================================== guard
def run_guard(env: Env, rec: Recipe) -> dict[str, Any]:
    """5-fold CV (train+val only), fold-seed s0, model seed 0, 20-epoch schedule, eager."""
    g = env.pcfg["guard"]
    out_path = env.P.guard(rec.key)
    if out_path.exists():
        print(f"[guard] {rec.key}: summary exists, skipping")
        return json.loads(out_path.read_text(encoding="utf-8"))
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    tcfg = train_cfg_for(env, rec, int(g["model_seed"]), None)
    gdir = env.P.results / "guard"
    f1_curves: list[list[float]] = []
    acc_curves: list[list[float]] = []
    for f in env.folds:
        fp = gdir / f"{rec.key}_f{f}.json"
        if fp.exists():
            frec = json.loads(fp.read_text(encoding="utf-8"))
        else:
            tr, ev = fold_data(env, frame, f)
            ex = build_extras(rec, env.inputs if rec.factors else None, env.pcfg, tr, cv_forbid(ev))
            meta = {"run_id": f"guard_{rec.key}_f{f}", "fold": f, "candidate": rec.key}
            with gpu_section(
                int(env.pcfg["gpu_lock_expected_s"]),
                f"intent-router phase4c guard {rec.key} f{f}",
                float(env.pcfg["gpu_lock_poll_s"]),
            ):
                res = train_fold(
                    tcfg, tr, ev, meta, oe_texts=ex.oe_texts, extra_train_rows=ex.extra_rows
                )
            _free_gpu()
            frec = {
                "candidate": rec.key, "fold": f, "extras": ex.info,
                "macro_f1": [e["macro_f1"] for e in res.epochs],
                "accuracy": [e["accuracy"] for e in res.epochs],
                "wall_clock_s": res.wall_clock_s, "peak_vram_mb": res.peak_vram_mb,
                "wandb_url": res.wandb_url, "nan_detected": res.nan_detected,
                "gpu_exclusive": res.gpu_exclusive, "config": res.config,
            }  # fmt: skip
            write_json(fp, frec)
            print(f"[guard] {rec.key} fold {f}: best-epoch F1 {max(frec['macro_f1']):.4f}")
        f1_curves.append(frec["macro_f1"])
        acc_curves.append(frec["accuracy"])
    out = guard_summary(rec.key, f1_curves, acc_curves, g, env.smoke, env.pcfg["current"])
    write_json(out_path, out)
    print(
        f"[guard] {rec.key}: epoch {out['argmax_epoch']} macro-F1 {out['macro_f1_at_argmax']:.4f} "
        f"passed={out['passed']}"
    )
    return out


def guard_summary(
    key: str,
    f1_curves: Sequence[Sequence[float]],
    acc_curves: Sequence[Sequence[float]],
    g: Mapping[str, Any],
    smoke: bool,
    current_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Mean-curve argmax e*, guard value and verdict (pure; ties go to the earliest epoch)."""
    mean_curve = np.mean(np.array(f1_curves), axis=0)
    mean_acc = np.mean(np.array(acc_curves), axis=0)
    argmax_epoch = int(np.argmax(mean_curve)) + 1
    value = float(mean_curve[argmax_epoch - 1])
    thr = float(g["threshold"])
    out: dict[str, Any] = {
        "candidate": key, "label": "measured, CV s0 5-fold, model seed 0, eager",
        "n_folds": len(f1_curves), "smoke": smoke, "mean_curve": mean_curve.tolist(),
        "mean_accuracy_curve": mean_acc.tolist(),
        "argmax_epoch": argmax_epoch, "macro_f1_at_argmax": value,
        "deployed_epoch": argmax_epoch, "macro_f1_at_deployed": value,
        "threshold": thr, "comparator": float(g["comparator"]),
        "passed": bool(value >= thr),
        "why": None
        if value >= thr
        else f"macro-F1 {value:.4f} < guard {thr:.4f} at epoch {argmax_epoch}",
    }  # fmt: skip
    if key == "current":
        # pre-registered: current ships epoch 9 (Open-set scorer comparison / 4a); the eager curve's
        # own argmax is
        # reported above and decides the guard, exactly as for every other candidate.
        e = int(current_cfg["deployed_epoch"])
        if e <= len(mean_curve):
            out["deployed_epoch"] = e
            out["macro_f1_at_deployed"] = float(mean_curve[e - 1])
    return out


# ==================================================================================== folds
def eval_fold(
    model: Any, tok: Any, name: str, max_len: int, bs: int, ev: pd.DataFrame,
    mt_rows: pd.DataFrame, pcfg: Mapping[str, Any], fold: int,
) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Clean / neutral-swap / char_mixed_10 / eval-MT predictions of one fold model (ids only)."""

    def logits(texts: list[str]) -> np.ndarray:
        return predict(model, tok, texts, name, max_len, bs)[0]

    gold = ev["label"].map({lab: i for i, lab in enumerate(rb._label_names())}).to_numpy()  # noqa: SLF001
    ids = ev["id"].astype(str).tolist()
    lg = logits(ev["text"].tolist())
    clean = lg.argmax(1)
    probs = softmax(lg)
    clean_df = pd.DataFrame(
        {"id": ids, "fold_seed": int(pcfg["guard"]["fold_seed_idx"]),
         "model_seed": int(pcfg["guard"]["model_seed"]), "fold": fold, "gold": gold, "pred": clean}
    )  # fmt: skip
    for i in range(probs.shape[1]):
        clean_df[f"prob_{i}"] = probs[:, i]
    clean_of = dict(zip(ids, clean, strict=True))
    gold_of = dict(zip(ids, gold, strict=True))

    swap_parts = []
    for name_s in pcfg["neutral_swaps"]:
        src, dst = rb.IDSWAPS[name_s]
        swapped = [rb.swap_ids(t, src, dst) for t in ev["text"]]
        keep = [i for i, (_, c) in enumerate(swapped) if c > 0]
        if not keep:
            continue
        pred = logits([swapped[i][0] for i in keep]).argmax(1)
        sub_ids = [ids[i] for i in keep]
        swap_parts.append(pd.DataFrame(
            {"id": sub_ids, "swap": name_s, "gold": [gold_of[i] for i in sub_ids],
             "clean_pred": [clean_of[i] for i in sub_ids], "swap_pred": pred}
        ))  # fmt: skip
    swap_df = pd.concat(swap_parts, ignore_index=True) if swap_parts else pd.DataFrame(
        columns=["id", "swap", "gold", "clean_pred", "swap_pred"])  # fmt: skip

    nz = pcfg["noise"]
    noisy = [
        rb.apply_noise(t, nz["perturbation"], i, int(nz["seed"]))
        for t, i in zip(ev["text"], ids, strict=True)
    ]
    noise_df = pd.DataFrame(
        {"id": ids, "gold": gold, "clean_pred": clean, "noise_pred": logits(noisy).argmax(1)}
    )

    mt_f = mt_rows[mt_rows["id"].astype(str).isin(set(ids))]
    mt_pred = logits(mt_f["text"].tolist()).argmax(1) if len(mt_f) else np.zeros(0, dtype=int)
    mt_ids = mt_f["id"].astype(str).tolist()
    mt_df = pd.DataFrame(
        {"id": mt_ids, "lang": mt_f["lang"].to_numpy(), "system": mt_f["system"].to_numpy(),
         "kept": mt_f["kept"].to_numpy(), "gold": [gold_of[i] for i in mt_ids],
         "en_pred": [clean_of[i] for i in mt_ids], "mt_pred": mt_pred}
    )  # fmt: skip
    return {"clean": clean_df, "swap": swap_df, "noise": noise_df, "mt": mt_df}


def _reusable_fold_dir(env: Env, rec: Recipe, k: int, e_star: int) -> Path | None:
    """`current`: the robustness fold model if its run_meta matches this config."""
    if rec.key != "current" or env.smoke:
        return None
    fdir = Path(env.pcfg["current"]["fold_models_dir"]) / f"fold{k}"
    meta = read_json(fdir / "run_meta.json")
    g = env.pcfg["guard"]
    ok = (
        meta is not None and int(meta["stop_epoch"]) == e_star
        and int(meta["model_seed"]) == int(g["model_seed"])
        and meta["attn_implementation"] == "eager"
    )  # fmt: skip
    return fdir if ok else None


def ensure_fold_model(
    env: Env, rec: Recipe, frame: pd.DataFrame, k: int, e_star: int,
    *, tag: str = "p4c", label_prefix: str = "phase4c", fdir: Path | None = None,
) -> Path:  # fmt: skip
    """Fold model stopped at e*: reused (current) or trained deterministically and saved.

    tag / label_prefix / fdir default to the Robustness and open-set training fixes names;
    Multi-axis selection passes its own so run ids,
    GPU-section labels and the model directory carry 4e's names (and per-seed paths).
    """
    reuse = _reusable_fold_dir(env, rec, k, e_star)
    if reuse is not None:
        print(f"[folds] {rec.key} fold {k}: reusing {reuse} (run_meta matches)")
        return reuse
    fdir = fdir or env.P.outputs / rec.key / "fold_models" / f"fold{k}"
    meta_path = fdir / "run_meta.json"
    want = {"stop_epoch": e_star, "model_seed": int(env.pcfg["guard"]["model_seed"]),
            "recipe": {f: getattr(rec, f) for f in FACTORS}}  # fmt: skip
    meta = read_json(meta_path)
    if meta is not None:
        if any(meta.get(a) != b for a, b in want.items()):
            raise AssertionError(
                f"cached fold {k} of {rec.key} differs from the config; delete {fdir}"
            )
        print(f"[folds] {rec.key} fold {k}: cached")
        return fdir
    g = env.pcfg["guard"]
    tr, ev = fold_data(env, frame, k)
    ex = build_extras(rec, env.inputs if rec.factors else None, env.pcfg, tr, cv_forbid(ev))
    tcfg = train_cfg_for(env, rec, int(g["model_seed"]), e_star)
    run_id = f"{tag}_{rec.key}_fs{g['fold_seed_idx']}_ms{g['model_seed']}_f{k}"
    print(f"[folds] {rec.key} fold {k}: training {len(tr)} rows to epoch {e_star} {ex.info}")
    with (
        rb.wandb_disabled(),  # fold models are evaluation artifacts, not W&B runs (as in 4a)
        gpu_section(int(env.pcfg["gpu_lock_expected_s"]), f"intent-router {label_prefix} {run_id}",
                    float(env.pcfg["gpu_lock_poll_s"])) as lock,
    ):  # fmt: skip
        res, model, tok = _train_model(
            tcfg, tr, ev, {"run_id": run_id, "fold": k},
            oe_texts=ex.oe_texts, extra_train_rows=ex.extra_rows,
        )  # fmt: skip
        try:
            fdir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(fdir)
            tok.save_pretrained(fdir)
            fingerprint = state_dict_sha256(model)
        finally:
            model = None
            gc.collect()
    write_json(meta_path, {
        "fold": k, "run_id": run_id, "n_train": len(tr), "n_eval": len(ev), **want,
        "attn_implementation": tcfg.attn_implementation, "fingerprint": fingerprint,
        "extras": ex.info, "final_epoch_eval": res.epochs[-1], "wall_clock_s": res.wall_clock_s,
        "gpu_exclusive": res.gpu_exclusive, "lock_waited_s": lock.get("waited_s"),
        "git_sha": git_sha(),
    })  # fmt: skip
    _free_gpu()
    return fdir


def stage_folds(env: Env, rec: Recipe) -> None:
    """Fold models at e*, then OOF clean / swap / noise / eval-MT predictions (ids only)."""
    import torch

    out_dir = env.P.cand(rec.key)
    done = out_dir / "folds.json"
    if done.exists():
        print(f"[folds] {rec.key}: folds.json exists, skipping")
        return
    e_star = deployed_epoch(env, rec)
    frame = load_cv_frame(env.fcfg["data_path"], env.fcfg["splits_path"])
    t = env.fcfg["train"]
    bs = int(env.pcfg["predict_batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts: dict[str, list[pd.DataFrame]] = {"clean": [], "swap": [], "noise": [], "mt": []}
    fold_meta = []
    for k in env.folds:
        fdir = ensure_fold_model(env, rec, frame, k, e_star)
        _, ev = fold_data(env, frame, k)
        with gpu_section(
            600, f"intent-router phase4c eval {rec.key} f{k}", float(env.pcfg["gpu_lock_poll_s"])
        ):
            tok, model = rb._load_classifier(fdir, device)  # noqa: SLF001
            try:
                got = eval_fold(model, tok, t["model_name"], int(t["max_len"]), bs, ev,
                                env.inputs.eval_mt, env.pcfg, k)  # fmt: skip
            finally:
                model = None
                _free_gpu()
        for name, df in got.items():
            parts[name].append(df)
        fold_meta.append({"fold": k, "dir": str(fdir),
                          "run_meta": read_json(fdir / "run_meta.json")})  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {n: pd.concat(p, ignore_index=True) for n, p in parts.items()}
    for n, df in frames.items():
        df.to_csv(out_dir / f"oof_{n}.csv", index=False, float_format="%.9g")
    summary: dict[str, Any] = {
        "candidate": rec.key, "deployed_epoch": e_star, "smoke": env.smoke,
        "oof_accuracy": accuracy(frames["clean"]["gold"], frames["clean"]["pred"]),
        "oof_macro_f1": macro_f1(
            frames["clean"]["gold"].to_numpy(), frames["clean"]["pred"].to_numpy()
        ),
        "n_rows": len(frames["clean"]),
        "swap": swap_summary(frames["swap"]),
        "noise": noise_summary(frames["noise"]),
        "translation": mt_summary(frames["mt"], env.pcfg["translation"]["langs"]),
        "folds": fold_meta, "git_sha": git_sha(),
    }  # fmt: skip
    if rec.key == "current" and not env.smoke:
        ref = pd.read_csv(env.pcfg["current"]["oof_clean_predictions"]).set_index("id")["pred"]
        got_pred = frames["clean"].set_index("id")["pred"]
        summary["clean_preds_equal_phase4a"] = bool((ref.loc[got_pred.index] == got_pred).all())
    write_json(done, summary)
    flip, drop = summary["swap"]["pooled"]["flip_rate"], summary["noise"]["drop"]
    print(f"[folds] {rec.key}: OOF acc {summary['oof_accuracy']:.4f} flip {flip} drop {drop:.4f}")


# ----------------------------------------------------------------- OOF metric summaries
def swap_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Pooled and per-swap flip rate (prediction differs from the same model's clean one)."""

    def one(d: pd.DataFrame) -> dict[str, Any]:
        if d.empty:
            return {"n": 0, "flip_rate": None, "accuracy_swapped": None, "accuracy_clean": None}
        return {
            "n": len(d), "n_flips": int((d["swap_pred"] != d["clean_pred"]).sum()),
            "flip_rate": float((d["swap_pred"] != d["clean_pred"]).mean()),
            "accuracy_swapped": float((d["swap_pred"] == d["gold"]).mean()),
            "accuracy_clean": float((d["clean_pred"] == d["gold"]).mean()),
        }  # fmt: skip

    return {"pooled": one(df), "per_swap": {s: one(d) for s, d in df.groupby("swap")}}


def noise_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Accuracy clean vs noisy and the drop (clean - noisy)."""
    c, n = (
        float((df["clean_pred"] == df["gold"]).mean()),
        float((df["noise_pred"] == df["gold"]).mean()),
    )
    return {"n": len(df), "accuracy_clean": c, "accuracy_noisy": n, "drop": c - n,
            "flip_rate": float((df["clean_pred"] != df["noise_pred"]).mean())}  # fmt: skip


def mt_summary(df: pd.DataFrame, langs: Sequence[str]) -> dict[str, Any]:
    """Per language: agreement with the English prediction + accuracy, filtered and unfiltered."""
    out: dict[str, Any] = {}
    for lang in langs:
        d = df[df["lang"] == lang]
        rec: dict[str, Any] = {}
        for tag, sub in (("filtered", d[d["kept"]]), ("unfiltered", d)):
            rec[tag] = {
                "n": len(sub),
                "agreement": float((sub["mt_pred"] == sub["en_pred"]).mean()) if len(sub) else None,
                "accuracy": float((sub["mt_pred"] == sub["gold"]).mean()) if len(sub) else None,
            }
        out[lang] = rec
    for tag in ("filtered", "unfiltered"):
        agr = [out[la][tag]["agreement"] for la in langs if out[la][tag]["agreement"] is not None]
        acc = [out[la][tag]["accuracy"] for la in langs if out[la][tag]["accuracy"] is not None]
        out[f"mean_{tag}"] = {"agreement": float(np.mean(agr)) if agr else None,
                              "accuracy": float(np.mean(acc)) if acc else None}  # fmt: skip
    return out


# ================================================================================== trackb
def heldout_frozen(env: Env, eval_unknown: pd.DataFrame) -> np.ndarray:
    """Frozen-e5 rows of every held-out-class row (read only to drop synthetic training items)."""
    fz = env.ctx.frozen_of()
    return np.stack([fz[i] for i in eval_unknown["id"]])


def run_one_p4c(env: Env, rec: Recipe, run: ti.ImpRun) -> None:
    """trackb_improve.run_one with the recipe's overrides + extras; score CSV written last."""
    ctx = env.ctx
    csv_path = ctx.P.scores / f"{run.run_id}.csv"
    if csv_path.exists():
        print(f"[trackb] {run.run_id}: score CSV exists, skipping", flush=True)
        return
    t0 = time.perf_counter()
    sets = build_holdout_sets(ctx.df_all, list(run.holdout), ctx.smoke)
    tcfg = replace(ti.train_config(ctx, run.cand, run.seed), **recipe_overrides(rec, env.pcfg))
    if tcfg.attn_implementation != "eager" or tcfg.epochs != 20:
        raise AssertionError(
            "Robustness and open-set training fixes trains with the final config "
            "(eager, 20-epoch schedule)"
        )
    held_emb = heldout_frozen(env, sets.eval_unknown) if rec.a2 else None
    forbid = set(sets.eval_unknown["id"].astype(str)) | set(sets.cal["id"].astype(str))
    record: dict[str, Any] = {}

    def build(tr: pd.DataFrame, _ev: pd.DataFrame) -> Extras:
        return build_extras(
            rec, env.inputs if rec.factors else None, env.pcfg, tr, forbid, held_emb
        )

    with patched_training([ti], build, record):
        res, arrays, sha, waited = ti.train_and_extract(ctx, run, sets, tcfg)
    out_dir = ctx.P.outputs / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{run.run_id}.npz", **arrays)
    table, info = ti.score_run(ctx, sets, arrays)
    meta = {
        "run_id": run.run_id, "candidate": run.cand.key, "recipe": list(rec.factors),
        "kind": run.kind, "holdout": list(run.holdout), "seed": run.seed, "labels": sets.labels,
        "call_type": run.call_type, "git_sha": git_sha(), "smoke": ctx.P.smoke,
        "extras": record, "audit": sets.audit | info,
        "training": {
            "config": res.config, "epochs": res.epochs, "wall_clock_s": res.wall_clock_s,
            "load_s": res.load_s, "peak_vram_mb": res.peak_vram_mb,
            "gpu_exclusive": res.gpu_exclusive, "gpu_foreign_seen": res.gpu_foreign_seen,
            "nan_detected": res.nan_detected, "fingerprint": sha, "gpu_lock_waited_s": waited,
            "wandb_url": res.wandb_url, "run_total_s": time.perf_counter() - t0,
        },
    }  # fmt: skip
    write_json(ctx.P.runs / f"{run.run_id}.json", meta)
    ti._save_csv(table, csv_path)  # noqa: SLF001 - CSV existence == run complete (resume rule)
    print(f"[trackb] {run.run_id}: train {res.wall_clock_s:.0f}s total "
          f"{time.perf_counter() - t0:.0f}s extras={record}", flush=True)  # fmt: skip


def stage_trackb(env: Env, rec: Recipe) -> None:
    """
    5 DEV LOCO runs (seed 42) stopped at e*; `current` reuses the Open-set scorer comparison score
    tables.
    """
    if rec.key == "current" and env.reuses_phase3b_for_current():
        cur = env.ctx_current()
        base = ti.base_candidate(cur)
        missing = [
            c for c in env.dev_classes
            if not (cur.P.scores / f"{ti.run_id_for(base, 'loco', 42, c)}.csv").exists()
        ]  # fmt: skip
        if missing:
            raise SystemExit(f"Open-set scorer comparison DEV score tables missing for {missing}")
        print("[trackb] current: reusing the Open-set scorer comparison base DEV score tables")
        return
    cand = candidate_for(env, rec.key, deployed_epoch(env, rec))
    runs = ti.plan(
        env.ctx, cand, env.dev_classes, [], [int(env.ctx.cfg["loco_seed"])], "phase4c_trackb"
    )
    for r in runs:
        run_one_p4c(env, rec, r)


def dev_tables(env: Env, key: str) -> dict[str, pd.DataFrame]:
    """
    DEV class -> score table of a candidate (current: the Open-set scorer comparison base runs).
    """
    if key == "current" and env.reuses_phase3b_for_current():
        ctx, cand = env.ctx_current(), ti.base_candidate(env.ctx_current())
    else:
        ctx, cand = env.ctx, candidate_for(env, key, None)
    seed = int(env.ctx.cfg["loco_seed"])
    return {c: ti.read_scores(ctx, ti.run_id_for(cand, "loco", seed, c)) for c in env.dev_classes}


def table_rejection(table: pd.DataFrame, method: str, retention: float = 0.95) -> float:
    """Strict rejection recall at `retention` calibration-known retention (accept = score >= t)."""
    cal = table[table["set"] == "cal"][method].to_numpy()
    ev = table[table["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool)
    return float(np.mean(ev.loc[unk, method].to_numpy() < threshold_at_retention(cal, retention)))


def dev_summary(tables: Mapping[str, pd.DataFrame], methods: Sequence[str]) -> dict[str, Any]:
    """DEV-mean AUROC and strict rejection@95 per method (+ the per-class values)."""
    out: dict[str, Any] = {}
    for m in methods:
        au = [ti.table_auroc(t, m) for t in tables.values()]
        rj = [table_rejection(t, m) for t in tables.values()]
        out[m] = {"auroc_mean": float(np.mean(au)), "rej95_mean": float(np.mean(rj)),
                  "auroc_per_class": dict(zip(tables, au, strict=True)),
                  "rej95_per_class": dict(zip(tables, rj, strict=True))}  # fmt: skip
    return out


# ================================================================== pure statistics (paired)
def boot_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """[B, n] cluster-resample indices (rows drawn with replacement)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def ratio_samples(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per resample: sum(num[idx]) / sum(den[idx]) (NaN when the denominator is 0)."""
    d = den[idx].sum(axis=1).astype(float)
    return num[idx].sum(axis=1) / np.where(d > 0, d, np.nan)


def gain_record(point: float, samples: np.ndarray, level: float) -> dict[str, Any]:
    """Point estimate + percentile CI of a paired gain; `ci_excludes_zero_positive` = lo > 0."""
    s = samples[np.isfinite(samples)]
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "ci_excludes_zero_positive": bool(lo > 0), "n_resamples": int(len(s))}  # fmt: skip


def a1_gain(cur: pd.DataFrame, cand: pd.DataFrame, idx: np.ndarray | None, n_boot: int,
            seed: int, level: float) -> dict[str, Any]:  # fmt: skip
    """A1: pooled neutral-swap flip-rate reduction (current - candidate), cluster = row id."""
    key = ["id", "swap"]
    m = cur.merge(cand, on=key, suffixes=("_cur", "_cand"), validate="one_to_one")
    if len(m) != len(cur) or len(m) != len(cand):
        raise ValueError("current and candidate swap instances differ")
    m["f_cur"] = (m["swap_pred_cur"] != m["clean_pred_cur"]).astype(float)
    m["f_cand"] = (m["swap_pred_cand"] != m["clean_pred_cand"]).astype(float)
    g = m.groupby("id")[["f_cur", "f_cand"]].sum()
    n = m.groupby("id").size().to_numpy(dtype=float)
    fc, fa = g["f_cur"].to_numpy(), g["f_cand"].to_numpy()
    idx = boot_indices(len(n), n_boot, seed) if idx is None else idx
    point = (fc.sum() - fa.sum()) / n.sum()
    samples = ratio_samples(fc, n, idx) - ratio_samples(fa, n, idx)
    return {**gain_record(point, samples, level), "flip_rate_current": float(fc.sum() / n.sum()),
            "flip_rate_candidate": float(fa.sum() / n.sum()), "n_instances": int(n.sum()),
            "n_clusters": len(n)}  # fmt: skip


def mt_arrays(cur: pd.DataFrame, cand: pd.DataFrame, langs: Sequence[str]) -> dict[str, np.ndarray]:
    """[n_ids, n_langs] agreement of current / candidate and the shared `kept` mask."""
    key = ["id", "lang"]
    m = cur.merge(cand, on=key, suffixes=("_cur", "_cand"), validate="one_to_one")
    if len(m) != len(cur) or len(m) != len(cand):
        raise ValueError("current and candidate translation rows differ")
    if (m["kept_cur"] != m["kept_cand"]).any():
        raise ValueError("current and candidate disagree on the quality-filter mask")
    ids = sorted(m["id"].unique())
    pos = {i: n for n, i in enumerate(ids)}
    shape = (len(ids), len(langs))
    out = {k: np.zeros(shape) for k in ("agree_cur", "agree_cand", "kept")}
    for r in m.itertuples(index=False):
        i, j = pos[r.id], list(langs).index(r.lang)
        out["agree_cur"][i, j] = float(r.mt_pred_cur == r.en_pred_cur)
        out["agree_cand"][i, j] = float(r.mt_pred_cand == r.en_pred_cand)
        out["kept"][i, j] = float(r.kept_cur)
    return out


def a3_agreement_gains(
    cur: pd.DataFrame,
    cand: pd.DataFrame,
    langs: Sequence[str],
    n_boot: int,
    seed: int,
    level: float,
    idx: np.ndarray | None = None,
) -> dict[str, Any]:  # fmt: skip
    """A3: zh and mean-over-languages agreement gains, quality-filtered rows (cluster = id)."""
    a = mt_arrays(cur, cand, langs)
    n_ids = a["kept"].shape[0]
    empty = [la for la, k in zip(langs, a["kept"].sum(axis=0), strict=True) if k == 0]
    if empty:
        raise ValueError(f"no quality-filtered rows left for {empty}: the gain is undefined")
    idx = boot_indices(n_ids, n_boot, seed) if idx is None else idx
    per_lang_point, per_lang_samples = [], []
    for j in range(len(langs)):
        k = a["kept"][:, j]
        den = k
        d_pt = ((a["agree_cand"][:, j] * k).sum() - (a["agree_cur"][:, j] * k).sum()) / den.sum()
        d_s = ratio_samples(a["agree_cand"][:, j] * k, den, idx) - ratio_samples(
            a["agree_cur"][:, j] * k, den, idx
        )
        per_lang_point.append(d_pt)
        per_lang_samples.append(d_s)
    zj = list(langs).index("zh")
    return {
        "zh": gain_record(per_lang_point[zj], per_lang_samples[zj], level),
        "mean": gain_record(
            float(np.mean(per_lang_point)), np.mean(per_lang_samples, axis=0), level
        ),
        "per_language": {la: gain_record(p, s, level) for la, p, s in
                         zip(langs, per_lang_point, per_lang_samples, strict=True)},
        "n_ids": n_ids,
        "n_kept_per_language": dict(zip(langs, a["kept"].sum(axis=0).tolist(), strict=True)),
    }  # fmt: skip


def noise_gain(cur: pd.DataFrame, cand: pd.DataFrame, n_boot: int, seed: int, level: float,
               idx: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    """A3: 10%-noise accuracy-drop reduction (current drop - candidate drop), cluster = row id."""
    m = cur.merge(cand, on="id", suffixes=("_cur", "_cand"), validate="one_to_one")
    if len(m) != len(cur) or len(m) != len(cand):
        raise ValueError("current and candidate noise rows differ")
    d_cur = (m["clean_pred_cur"] == m["gold_cur"]).astype(float) - (
        m["noise_pred_cur"] == m["gold_cur"]
    ).astype(float)
    d_cand = (m["clean_pred_cand"] == m["gold_cand"]).astype(float) - (
        m["noise_pred_cand"] == m["gold_cand"]
    ).astype(float)
    delta = (d_cur - d_cand).to_numpy()
    idx = boot_indices(len(delta), n_boot, seed) if idx is None else idx
    rec = gain_record(float(delta.mean()), delta[idx].mean(axis=1), level)
    return {**rec, "drop_current": float(d_cur.mean()), "drop_candidate": float(d_cand.mean())}


def a2_gains(
    cur_tables: Mapping[str, pd.DataFrame], cand_tables: Mapping[str, pd.DataFrame],
    method: str, cur_method: str, safe_labels: Sequence[str], n_boot: int, seed: int, level: float,
) -> dict[str, Any]:  # fmt: skip
    """DEV AUROC and strict rejection@95 gains of (cand, method) over (current, cur_method).

    Rows are resampled within each holdout, stratified known/unknown, with identical draws for
    both sides (same seed, same unit shapes); the DEV mean is taken per resample.
    """
    if list(cur_tables) != list(cand_tables):
        raise ValueError("current and candidate DEV class sets differ")
    u_cur = [ti.boot_unit(t, cur_method, [0.95], safe_labels) for t in cur_tables.values()]
    u_cand = [ti.boot_unit(t, method, [0.95], safe_labels) for t in cand_tables.values()]
    for a, b in zip(u_cur, u_cand, strict=True):
        if len(a.known) != len(b.known) or len(a.unknown) != len(b.unknown):
            raise ValueError("current and candidate eval sets differ in size (not paired)")
    b_cur = ti.bootstrap_mean_ci(u_cur, n_boot, seed, False, level)
    b_cand = ti.bootstrap_mean_ci(u_cand, n_boot, seed, False, level)
    out: dict[str, Any] = {"method": method, "current_method": cur_method}
    for name, key in (("auroc_gain", "auroc"), ("rej95_gain", "strict_recall_95")):
        point = b_cand["point"][key] - b_cur["point"][key]
        samples = b_cand["samples"][key] - b_cur["samples"][key]
        out[name] = {**gain_record(point, samples, level), "candidate": b_cand["point"][key],
                     "current": b_cur["point"][key]}  # fmt: skip
    return out


# ============================================================================ decision rules
def best_score_by_auroc(dev_auroc: Mapping[str, float], order: Sequence[str]) -> str:
    """A2 candidate score: highest DEV AUROC (ties go to the earlier score in `order`)."""
    return max(order, key=lambda s: (dev_auroc[s], -list(order).index(s)))


def select_shipped_score(
    dev_auroc: Mapping[str, float], dev_rej95: Mapping[str, float], tol: float, order: Sequence[str]
) -> dict[str, Any]:
    """C-comb: among scores within `tol` DEV AUROC of the best, the highest DEV rejection@95."""
    best = max(dev_auroc[s] for s in order)
    eligible = [s for s in order if best - dev_auroc[s] <= tol + 1e-12]
    chosen = max(eligible, key=lambda s: (dev_rej95[s], dev_auroc[s], -list(order).index(s)))
    return {"chosen": chosen, "eligible": eligible, "best_dev_auroc": best, "tolerance": tol}


def _clauses(**kw: bool) -> dict[str, bool]:
    return {k: bool(v) for k, v in kw.items()}


def evaluate_a1(gains: Mapping[str, Any], guard_pass: bool) -> dict[str, Any]:
    """A1: the neutral-swap flip-rate reduction CI excludes 0 (> 0), AND the guard passes."""
    ok = gains["a1_flip_reduction"]["lo"] > 0
    return {"adopted": bool(ok and guard_pass), "guard_pass": bool(guard_pass),
            "clauses": _clauses(flip_reduction_ci_gt_0=ok),
            "adoption_metrics": ["a1_flip_reduction"] if ok else []}  # fmt: skip


def evaluate_a2(gains: Mapping[str, Any], guard_pass: bool, margin: float) -> dict[str, Any]:
    """A2: (DEV AUROC gain > margin OR DEV rej@95 gain CI > 0), AND the guard passes."""
    c_auroc = gains["a2_auroc_gain"]["point"] > margin
    c_rej = gains["a2_rej95_gain"]["lo"] > 0
    metrics = [m for m, ok in (("a2_auroc_gain", c_auroc), ("a2_rej95_gain", c_rej)) if ok]
    return {"adopted": bool((c_auroc or c_rej) and guard_pass), "guard_pass": bool(guard_pass),
            "clauses": _clauses(auroc_gain_gt_margin=c_auroc, rej95_gain_ci_gt_0=c_rej),
            "adoption_metrics": metrics}  # fmt: skip


def evaluate_a3(gains: Mapping[str, Any], guard_pass: bool) -> dict[str, Any]:
    """A3: (zh AND mean agreement-gain CIs > 0) OR (noise drop-reduction CI > 0), AND the guard."""
    c_agree = gains["a3_zh_agreement_gain"]["lo"] > 0 and gains["a3_mean_agreement_gain"]["lo"] > 0
    c_noise = gains["a3_noise_drop_reduction"]["lo"] > 0
    metrics = []
    if c_agree:
        metrics += ["a3_zh_agreement_gain", "a3_mean_agreement_gain"]
    if c_noise:
        metrics.append("a3_noise_drop_reduction")
    return {"adopted": bool((c_agree or c_noise) and guard_pass), "guard_pass": bool(guard_pass),
            "clauses": _clauses(
                agreement_zh_and_mean_ci_gt_0=c_agree, noise_drop_reduction_ci_gt_0=c_noise
            ),
            "adoption_metrics": metrics}  # fmt: skip


def evaluate_comb(
    adopted: Sequence[str], alone: Mapping[str, Mapping[str, Any]],
    comb_gains: Mapping[str, Mapping[str, Any]], guard_pass: bool,
) -> dict[str, Any]:  # fmt: skip
    """C-comb: guard passes AND, for each adopted factor's adoption metric(s), comb's paired gain
    point estimate >= the lower 95% bound of that factor's alone-gain."""
    checks = []
    for f in adopted:
        for m in alone[f]["adoption_metrics"]:
            lo = alone[f]["gains"][m]["lo"]
            pt = comb_gains[m]["point"]
            checks.append({"factor": f, "metric": m, "comb_point": pt, "alone_lo": lo,
                           "passed": bool(pt >= lo)})  # fmt: skip
    ok = bool(guard_pass and checks and all(c["passed"] for c in checks))
    return {"adopted": ok, "guard_pass": bool(guard_pass), "checks": checks}


# ===================================================================== gains + metrics table
def oof_frames(env: Env, key: str) -> dict[str, pd.DataFrame]:
    d = env.P.cand(key)
    return {n: pd.read_csv(d / f"oof_{n}.csv") for n in ("clean", "swap", "noise", "mt")}


def mt_with_bool(df: pd.DataFrame) -> pd.DataFrame:
    if df["kept"].dtype != bool:
        df = df.assign(kept=df["kept"].astype(str).str.lower().eq("true"))
    return df


def candidate_gains(env: Env, key: str, score: str | None) -> dict[str, Any]:
    """Every adoption metric of candidate `key` vs current (paired), with CIs.

    score = the Track B score whose DEV gains are evaluated (None skips the A2 metrics).
    """
    pc = env.pcfg
    bs, level = pc["bootstrap"], float(pc["bootstrap"]["level"])
    seed, n_boot = int(bs["seed"]), env.n_boot
    cur, cand = oof_frames(env, "current"), oof_frames(env, key)
    langs = pc["translation"]["langs"]
    out: dict[str, Any] = {}
    n_ids = cur["clean"]["id"].nunique()
    idx_rows = boot_indices(n_ids, n_boot, seed)  # shared by the noise gain (all 426 row ids)
    out["a1_flip_reduction"] = a1_gain(cur["swap"], cand["swap"], None, n_boot, seed, level)
    a3 = a3_agreement_gains(
        mt_with_bool(cur["mt"]), mt_with_bool(cand["mt"]), langs, n_boot, seed, level
    )
    out["a3_zh_agreement_gain"] = a3["zh"]
    out["a3_mean_agreement_gain"] = a3["mean"]
    out["a3_per_language_agreement_gain"] = a3["per_language"]
    out["a3_noise_drop_reduction"] = noise_gain(
        cur["noise"].sort_values("id").reset_index(drop=True),
        cand["noise"].sort_values("id").reset_index(drop=True),
        n_boot,
        seed,
        level,
        idx_rows,
    )
    if score is not None:
        safe = env.ctx.cfg["ood"]["safe_labels"]
        g = a2_gains(dev_tables(env, "current"), dev_tables(env, key), score,
                     pc["current_score"], safe, n_boot, seed, level)  # fmt: skip
        out["a2_auroc_gain"], out["a2_rej95_gain"] = g["auroc_gain"], g["rej95_gain"]
        out["a2_score"] = score
    return out


def candidate_row(env: Env, key: str, label: str | None = None) -> dict[str, Any]:
    """Every reported metric of one candidate (guard, shortcut, Track B, translation, noise)."""
    pc = env.pcfg
    folds = read_json(env.P.cand(key) / "folds.json")
    guard = read_json(env.P.guard(key))
    row: dict[str, Any] = {"candidate": label or key, "artifact_key": key}
    if guard is not None:
        keys = ("macro_f1_at_argmax", "argmax_epoch", "deployed_epoch", "macro_f1_at_deployed")
        row["guard"] = {k: guard[k] for k in (*keys, "passed", "threshold")}
    if folds is not None:
        row["oof_accuracy"] = folds["oof_accuracy"]
        row["oof_macro_f1"] = folds["oof_macro_f1"]
        row["neutral_swap"] = folds["swap"]
        row["noise_char_mixed_10"] = folds["noise"]
        row["translation"] = folds["translation"]
    with contextlib.suppress(FileNotFoundError):  # Track B not run yet for this candidate
        row["dev"] = dev_summary(dev_tables(env, key), pc["scores"])
    return row


def render_table_md(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[str], langs: Sequence[str]
) -> str:
    """Markdown tables of all candidates x all metrics (numbers only)."""

    def f(v: Any, nd: int = 4) -> str:
        return "-" if v is None else f"{v:.{nd}f}"

    lines = ["## Track A guard, shortcut, noise", "",
             "| candidate | guard F1 | e* | pass | flip | swap acc | noise drop | OOF acc |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        g, sw, nz = r.get("guard", {}), r.get("neutral_swap", {}), r.get("noise_char_mixed_10", {})
        lines.append(
            f"| {r['candidate']} | {f(g.get('macro_f1_at_argmax'))} | "
            f"{g.get('argmax_epoch', '-')} | "
            f"{g.get('passed', '-')} | {f(sw.get('pooled', {}).get('flip_rate'))} | "
            f"{f(sw.get('pooled', {}).get('accuracy_swapped'))} | {f(nz.get('drop'))} | "
            f"{f(r.get('oof_accuracy'))} |"
        )
    lines += ["", "## Track B DEV (mean over DEV classes): AUROC / rejection@95", "",
              "| candidate | " + " | ".join(scores) + " |",
              "|---|" + "---|" * len(scores)]  # fmt: skip
    for r in rows:
        dev = r.get("dev", {})
        cells = [
            f"{f(dev[s]['auroc_mean'])} / {f(dev[s]['rej95_mean'])}" if s in dev else "-"
            for s in scores
        ]
        lines.append(f"| {r['candidate']} | " + " | ".join(cells) + " |")
    for tag in ("filtered", "unfiltered"):
        lines += ["", f"## Eval-MT ({tag}): agreement / accuracy per language", "",
                  "| candidate | " + " | ".join(langs) + " | mean |",
                  "|---|" + "---|" * (len(langs) + 1)]  # fmt: skip
        for r in rows:
            tr = r.get("translation", {})
            cells = [
                f"{f(tr[la][tag]['agreement'])} / {f(tr[la][tag]['accuracy'])} "
                f"(n={tr[la][tag]['n']})" if la in tr else "-" for la in langs
            ]  # fmt: skip
            m = tr.get(f"mean_{tag}", {})
            lines.append(f"| {r['candidate']} | " + " | ".join(cells)
                         + f" | {f(m.get('agreement'))} / {f(m.get('accuracy'))} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def write_metrics_table(env: Env, entries: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """results/phase4c/metrics_table.{json,md} for the available (artifact_key, label) pairs."""
    rows = []
    for key, label in entries:
        if (env.P.cand(key) / "folds.json").exists():
            rows.append(candidate_row(env, key, label))
    write_json(env.P.results / "metrics_table.json", {"rows": rows, "git_sha": git_sha()})
    (env.P.results / "metrics_table.md").write_text(
        render_table_md(rows, env.pcfg["scores"], env.pcfg["translation"]["langs"]),
        encoding="utf-8",
    )
    return rows


# ============================================================================ decide_factors
def stage_decide_factors(env: Env) -> None:
    """Paired bootstraps of A1/A2/A3 vs current + the pre-registered adoption rules."""
    pc = env.pcfg
    out_path = env.P.results / "factor_decisions.json"
    for k in ("current", *FACTORS):
        for need in (env.P.guard(k), env.P.cand(k) / "folds.json"):
            if not need.exists():
                raise SystemExit(f"missing {need}: run --stage cand_{k} (or its guard/folds) first")
    scores = list(pc["scores"])
    dev_cur = dev_summary(dev_tables(env, "current"), scores)
    dev_a2 = dev_summary(dev_tables(env, "a2"), scores)
    a2_score = best_score_by_auroc({s: dev_a2[s]["auroc_mean"] for s in scores}, scores)
    guards = {k: read_json(env.P.guard(k)) for k in ("current", *FACTORS)}
    margin = float(pc["factors"]["a2"]["auroc_margin"])
    decisions: dict[str, Any] = {}
    for f in FACTORS:
        gains = candidate_gains(env, f, a2_score if f == "a2" else None)
        gp = bool(guards[f]["passed"])
        ev = {"a1": lambda g, p: evaluate_a1(g, p), "a2": lambda g, p: evaluate_a2(g, p, margin),
              "a3": lambda g, p: evaluate_a3(g, p)}[f](gains, gp)  # fmt: skip
        used = ev["adoption_metrics"]
        own = {m: gains[m] for m in gains if m in _FACTOR_METRICS[f] or m in used}
        decisions[f] = {**ev, "gains": own, "all_gains": gains}
    adopted = [f for f in FACTORS if decisions[f]["adopted"]]
    out = {
        "label": "measured, paired bootstrap (clusters = row ids for OOF metrics; Track B rows "
        "resampled within each holdout, stratified known/unknown)",
        "bootstrap": pc["bootstrap"] | {"n_resamples_used": env.n_boot},
        "rules": {
            "a1": "neutral-swap flip-rate reduction CI excludes 0 (> 0) AND guard passes",
            "a2": f"(DEV AUROC gain > {margin} OR DEV rej@95 gain CI > 0) AND guard passes",
            "a3": "(zh AND mean-over-languages agreement gain CIs > 0, filtered set) OR "
            "(10%-noise drop-reduction CI > 0), AND guard passes",
        },
        "a2_score_by_dev_auroc": a2_score,
        "dev_current": dev_cur, "dev_a2": dev_a2,
        "guards": {
            k: {x: g[x] for x in ("macro_f1_at_argmax", "argmax_epoch", "passed")}
            for k, g in guards.items()
        },
        "factors": decisions, "adopted": adopted,
        "git_sha": git_sha(), "smoke": env.smoke,
    }  # fmt: skip
    write_json(out_path, out)
    write_metrics_table(env, [(k, k) for k in ("current", *FACTORS)])
    for f in FACTORS:
        d = decisions[f]
        print(
            f"[decide] {f}: adopted={d['adopted']} clauses={d['clauses']} guard={d['guard_pass']}"
        )
    print(f"[decide] adopted factors: {adopted or 'none'} -> {out_path}")


_FACTOR_METRICS = {
    "a1": ("a1_flip_reduction",),
    "a2": ("a2_auroc_gain", "a2_rej95_gain"),
    "a3": ("a3_zh_agreement_gain", "a3_mean_agreement_gain", "a3_noise_drop_reduction"),
}


# ===================================================================================== comb
def adopted_factors(env: Env) -> list[str]:
    """Factors adopted by decide_factors (smoke: the --smoke-adopt override when given)."""
    if env.smoke and env.smoke_adopt is not None:
        return list(env.smoke_adopt)
    dec = read_json(env.P.results / "factor_decisions.json")
    if dec is None:
        raise SystemExit("run --stage decide_factors first")
    return list(dec["adopted"])


def run_candidate_pipeline(env: Env, rec: Recipe) -> None:
    """guard -> folds -> trackb for one recipe (each stage resumable)."""
    guard = run_guard(env, rec)
    if not guard["passed"] and not env.pcfg["run_even_if_guard_failed"]:
        print(f"[cand] {rec.key} FAILED the guard: skipping folds/trackb")
        return
    stage_folds(env, rec)
    stage_trackb(env, rec)


def stage_comb(env: Env) -> None:
    """C-comb: guard + folds + trackb for the adopted factors, then the C-comb rules."""
    adopted = adopted_factors(env)
    out_path = env.P.results / "comb_decision.json"
    if not adopted:
        write_json(out_path, {"status": NO_FACTORS_MSG, "adopted_factors": [], "adopted": False})
        print(f"[comb] {NO_FACTORS_MSG}")
        return
    rec = comb_recipe(adopted)
    run_candidate_pipeline(env, rec)
    pc = env.pcfg
    scores = list(pc["scores"])
    dev = dev_summary(dev_tables(env, rec.key), scores)
    sel = select_shipped_score(
        {s: dev[s]["auroc_mean"] for s in scores}, {s: dev[s]["rej95_mean"] for s in scores},
        float(pc["shipped_score_auroc_tolerance"]), scores,
    )  # fmt: skip
    gains = candidate_gains(env, rec.key, sel["chosen"])
    dec = read_json(env.P.results / "factor_decisions.json") or {}
    alone = {f: dec["factors"][f] for f in adopted}
    guard = read_json(env.P.guard(rec.key)) or {}
    verdict = evaluate_comb(adopted, alone, gains, bool(guard.get("passed")))
    out = {
        "status": "adopted" if verdict["adopted"] else "rejected",
        "adopted_factors": adopted, "artifact_key": rec.key,
        "deployed_epoch": deployed_epoch(env, rec),
        "single_factor_alias": rec.key != "comb",
        "shipped_score": sel, "dev_comb": dev, "gains_vs_current": gains, "rules": verdict,
        "guard": {k: guard.get(k) for k in ("macro_f1_at_argmax", "argmax_epoch", "passed")},
        "git_sha": git_sha(), "smoke": env.smoke,
    }  # fmt: skip
    write_json(out_path, out)
    write_metrics_table(env, [(k, k) for k in ("current", *FACTORS)] + [(rec.key, "comb")])
    print(f"[comb] {out['status']}: shipped score {sel['chosen']}; checks "
          f"{[(c['factor'], c['metric'], c['passed']) for c in verdict['checks']]}")  # fmt: skip


# ============================================================================== confirm_comb
def adopted_comb(env: Env, why_stop: str) -> tuple[Recipe, dict[str, Any]]:
    """(comb recipe, comb_decision.json); stops unless comb was adopted (smoke: always proceeds,
    with a stand-in shipped score if no decision file exists)."""
    dec = read_json(env.P.results / "comb_decision.json")
    if env.smoke and env.smoke_adopt:
        dec = dec or {"shipped_score": {"chosen": env.pcfg["current_score"]}}
        dec["adopted_factors"] = list(env.smoke_adopt)
        return comb_recipe(dec["adopted_factors"]), dec
    if dec is None or dec.get("status") != "adopted":
        raise SystemExit(f"comb is not adopted (see comb_decision.json): {why_stop}")
    return comb_recipe(dec["adopted_factors"]), dec


def stage_confirm_comb(env: Env) -> None:
    """CONFIRM classes + headline seeds for comb (shipped score) vs current maha_ft."""
    rec, dec = adopted_comb(env, "nothing to confirm")
    cfg = env.ctx.cfg
    # Confirm runs are comb-specific even for a single-factor comb (alone only trained DEV classes).
    cand = candidate_for(env, "comb", deployed_epoch(env, rec))
    plan = ti.plan(env.ctx, cand, cfg["confirm_classes"] if not env.smoke else env.dev_classes,
                   cfg["headline_seeds"] if not env.smoke else [], [int(cfg["loco_seed"])],
                   "phase4c_confirm")  # fmt: skip
    for r in plan:
        run_one_p4c(env, rec, r)
    if env.smoke:
        print("[confirm_comb] smoke: runs only (no CONFIRM tables)")
        return
    method = dec["shipped_score"]["chosen"]
    cur_ctx = env.ctx_current()
    e_cur = ti.confirm_entry(cur_ctx, ti.base_candidate(cur_ctx), env.pcfg["current_score"])
    e_new = ti.confirm_entry(env.ctx, cand, method)
    level = float(env.pcfg["bootstrap"]["level"])
    deltas = {part: ti.paired_delta_ci(e_new[part]["_samples"], e_cur[part]["_samples"], level)
              for part in ("confirm", "headline")}  # fmt: skip
    for e in (e_cur, e_new):
        for part in ("confirm", "headline"):
            e[part].pop("_samples")
    write_json(env.P.results / "confirm_comb.json", {
        "label": "measured, CONFIRM LOCO classes (5, seed 42) and headline holdout (seeds "
        f"{cfg['headline_seeds']}); thresholds from calibration-known rows",
        "comb": {"factors": dec["adopted_factors"], "score": method, **e_new},
        "current": {"score": env.pcfg["current_score"], **e_cur},
        "paired_delta_comb_minus_current": deltas,
        "ci_note": "AUPR has a point estimate only; resampled: AUROC, FPR@95TPR, recall, retention",
        "git_sha": git_sha(),
    })  # fmt: skip
    print("[confirm_comb] wrote confirm_comb.json")


# ============================================================================== final_retrain
def derived_final_cfg(env: Env, rec: Recipe, e_star: int, smoke: bool) -> dict[str, Any]:
    """final.yaml with comb's training recipe at e* and v2 paths (the model-defining fields)."""
    cfg = json.loads(json.dumps(env.fcfg))  # deep copy of plain yaml data
    v2 = env.pcfg["final_v2"]
    cfg["train"] = cfg["train"] | {"stop_epoch": e_star, **recipe_overrides(rec, env.pcfg)}
    cfg["model_dir"] = v2["model_dir"]
    cfg["determinism"]["sdpa_timing_run"] = bool(v2["sdpa_timing_run"])
    cfg["wandb"]["group"] = "phase4c-final-v2"
    if not smoke:
        sel_dir = Path(v2["results_dir"])
        cfg["selection_path"] = str(sel_dir / "selection_v2.json")
        cfg["analysis"]["oof_dir"] = str(sel_dir / "oof")
        cfg["analysis"]["oof_config_id"] = f"comb_{'_'.join(rec.factors)}"
        cfg["analysis"]["oof_predictions_per_id"] = 1
    return cfg


def write_v2_cv_stand_ins(env: Env, rec: Recipe, cfg: Mapping[str, Any], e_star: int) -> None:
    """Selection json + OOF csv of the v2 config's fold models (the shape stage_evaluate reads)."""
    cid = cfg["analysis"]["oof_config_id"]
    guard = read_json(env.P.guard(rec.key))
    assert guard is not None
    f1 = [
        read_json(env.P.results / "guard" / f"{rec.key}_f{k}.json")["macro_f1"][e_star - 1]
        for k in env.folds
    ]  # type: ignore[index]
    ac = [
        read_json(env.P.results / "guard" / f"{rec.key}_f{k}.json")["accuracy"][e_star - 1]
        for k in env.folds
    ]  # type: ignore[index]
    sel = {"candidates": {cid: {
        "chosen_epoch": e_star, "n_runs": len(f1), "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1, ddof=1)), "accuracy_mean": float(np.mean(ac)),
        "accuracy_std": float(np.std(ac, ddof=1)),
        "note": "v2 config: 5 folds, fold-seed s0, model seed 0 (v1 used 9 fold-runs per id)",
    }}}  # fmt: skip
    write_json(Path(cfg["selection_path"]), sel)
    oof_dir = Path(cfg["analysis"]["oof_dir"])
    oof_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(env.P.cand(rec.key) / "oof_clean.csv", oof_dir / f"{cid}.csv")


def archive_v1(env: Env) -> None:
    """Copy (never move) v1's results/outputs; the shared append-only test log stays in place."""
    v2 = env.pcfg["final_v2"]
    src_r, dst_r = Path(env.fcfg["results_dir"]), Path(v2["archive_results"])
    if dst_r.exists():
        print(f"[final] {dst_r} exists: v1 already archived")
    else:
        shutil.copytree(src_r, dst_r, ignore=shutil.ignore_patterns("test_eval_log.jsonl"))
        fig_dst = dst_r / "figures"
        fig_dst.mkdir(exist_ok=True)
        for p in Path(env.fcfg["figures_dir"]).glob("final_*.png"):
            shutil.copyfile(p, fig_dst / p.name)
    src_o, dst_o = Path(env.fcfg["outputs_dir"]), Path(v2["archive_outputs"])
    if not dst_o.exists():
        shutil.copytree(src_o, dst_o)


def shipped_ood_threshold(cfg: Mapping[str, Any], npz_path: Path, method: str, retention: float,
                          fusion_names: Sequence[str]) -> dict[str, Any]:  # fmt: skip
    """95%-val-retention threshold of the shipped score, from saved train/val/test arrays only."""
    from intent_router import data as data_mod
    from intent_router.ood import fit_gaussian_lw, score_mahalanobis, score_neg_energy

    z = np.load(npz_path)
    labels = list(data_mod.LABELS)
    full = data_mod.load_data(cfg["data_path"]).set_index("id")["label"]
    l2i = {lab: i for i, lab in enumerate(labels)}
    y_tr = np.array([l2i[full[i]] for i in z["train_ids"]])
    g = fit_gaussian_lw(z["train_features"], y_tr, len(labels))

    def comps(lg: np.ndarray, ft: np.ndarray) -> dict[str, np.ndarray]:
        return {"maha_ft": score_mahalanobis(ft, g), "neg_energy": score_neg_energy(lg)}

    val_c, test_c = (
        comps(z["val_logits"], z["val_features"]),
        comps(z["test_logits"], z["test_features"]),
    )
    if method == "fuse_maha_energy":
        fusion = ti.fit_rank_fusion([val_c[n] for n in fusion_names])
        val_s, test_s = fusion.transform([val_c[n] for n in fusion_names]), fusion.transform(
            [test_c[n] for n in fusion_names])  # fmt: skip
    else:
        val_s, test_s = val_c[method], test_c[method]
    thr = threshold_at_retention(val_s, retention)
    return {"method": method, "threshold": float(thr), "retention_target": retention,
            "val_retention_achieved": float(np.mean(val_s >= thr)),
            "test_coverage_at_threshold": float(np.mean(test_s >= thr)), "n_val": len(val_s),
            "n_test": len(test_s),
            "note": "12 classes known: fitted on val, applied to the saved test arrays "
            "(no new test inference)"}  # fmt: skip


def stage_final_retrain(env: Env) -> None:
    """Retrain the final model with comb's recipe, ONE logged test evaluation, artifacts v2."""
    rec, dec = adopted_comb(env, "the final model stays")
    if not env.smoke and read_json(env.P.results / "confirm_comb.json") is None:
        raise SystemExit("run --stage confirm_comb first (the shipped score comes from CONFIRM)")
    e_star = deployed_epoch(env, rec)
    cfg = derived_final_cfg(env, rec, e_star, env.smoke)
    inp = env.inputs if rec.factors else None
    method = dec["shipped_score"]["chosen"]
    if env.smoke:
        root = env.P.results.parent / "final_v2"
        P = final_mod.Paths(root / "results", root / "figures", root / "outputs", root / "model",
                            root / "test_eval_log.jsonl", True)  # fmt: skip
        use_wandb = False
    else:
        archive_v1(env)
        write_v2_cv_stand_ins(env, rec, cfg, e_star)
        P = final_mod.make_paths(cfg, False)
        use_wandb = bool(cfg["wandb"]["enabled"]) and bool(env.ctx.use_wandb)
        v2dir = Path(env.pcfg["final_v2"]["results_dir"])
        v2dir.mkdir(parents=True, exist_ok=True)
        (v2dir / "final_v2_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    record: dict[str, Any] = {}

    def build(tr: pd.DataFrame, _ev: pd.DataFrame) -> Extras:
        # No holdout in the 12-class run: the synthetic pool is unfiltered; A3 = train ids only.
        return build_extras(rec, inp, env.pcfg, tr, set())

    with patched_training([final_mod], build, record):
        final_mod.stage_train(cfg, P, env.smoke)
    final_mod.stage_evaluate(cfg, P, env.smoke, use_wandb)
    tsum = read_json(P.results / "train_summary.json") or {}
    ood = shipped_ood_threshold(
        cfg, P.outputs / "features_logits.npz", method, float(env.ctx.tb["ood"]["retention"]),
        env.ctx.cfg["c1"]["fusion_components"],
    )  # fmt: skip
    write_json(P.results / "ood_shipped.json", ood)
    write_json(P.results / "model_version.json", {
        "version": "v2", "factors": list(rec.factors), "deployed_epoch": e_star,
        "train_config": cfg["train"], "extras": record,
        "model_fingerprint": tsum.get("model_fingerprint"), "model_dir": str(P.model_dir),
        "shipped_ood_score": method, "archived_v1": str(env.pcfg["final_v2"]["archive_results"]),
        "git_sha": git_sha(), "smoke": env.smoke,
    })  # fmt: skip
    if env.smoke:
        print("[final] smoke: trained + evaluated on val stand-in; analysis/robustness skipped")
        return
    cfg_path = Path(env.pcfg["final_v2"]["results_dir"]) / "final_v2_config.yaml"
    subprocess.run(  # noqa: S603 - analysis refuses to run in a process that loaded ood/baselines
        [sys.executable, "-m", "intent_router.analysis", "--config", str(cfg_path)],
        check=True, env={**os.environ, "PYTHONPATH": "src"},
    )  # fmt: skip
    write_side_by_side(env)
    robustness_v2(env, rec, cfg_path)


def write_side_by_side(env: Env) -> None:
    """Old (results/final_v1) vs new (results/final) headline test numbers."""
    old = read_json(Path(env.pcfg["final_v2"]["archive_results"]) / "track_a.json")
    new = read_json(Path(env.fcfg["results_dir"]) / "track_a.json")
    if old is None or new is None:
        print("[final] side-by-side skipped: track_a.json missing")
        return

    def pick(d: Mapping[str, Any]) -> dict[str, Any]:
        t = d["test"]
        return {"model_fingerprint": d["model_fingerprint"], "macro_f1": t["macro_f1"],
                "accuracy": t["accuracy"], "temperature": d["calibration"]["temperature"],
                "ece_test_after": d["calibration"]["test"]["ece_after"],
                "selective_at_threshold": t["selective"]["at_threshold"]}  # fmt: skip

    label = "test split, one logged evaluation each"
    write_json(env.P.results / "final_v1_vs_v2.json",
               {"v1": pick(old), "v2": pick(new), "label": label})  # fmt: skip
    print(f"[final] v1 test macro-F1 {old['test']['macro_f1']['point']:.4f} -> "
          f"v2 {new['test']['macro_f1']['point']:.4f}")  # fmt: skip


def robustness_v2(env: Env, rec: Recipe, final_cfg_path: Path) -> None:
    """ID-swap / noise / translation on v2's test rows (logged `robustness_inference` calls)
    and on the OOF rows with the comb fold models; the robustness evaluation's v1 results stay
    untouched."""
    v2 = env.pcfg["final_v2"]
    rcfg = yaml.safe_load(Path(env.pcfg["robustness_config"]).read_text(encoding="utf-8"))
    out = env.P.outputs / rec.key
    res_dir = Path(v2["robustness_results"])
    rcfg |= {"final_config": str(final_cfg_path), "results_dir": str(res_dir),
             "figures_dir": str(res_dir / "figures"), "outputs_dir": str(out)}  # fmt: skip
    cache = out / "translations.csv"
    src = Path("outputs/robustness/translations.csv")  # same NLLB greedy translations as 4a
    if not cache.exists() and src.exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cache)
    r_path = Path(v2["results_dir"]) / "robustness_v2.yaml"
    r_path.write_text(yaml.safe_dump(rcfg, sort_keys=False), encoding="utf-8")
    rcfg, fcfg = rb.load_configs(str(r_path))
    P = rb.Paths(rcfg, False)
    rb.stage_idswap(rcfg, fcfg, P)
    rb.stage_noise(rcfg, fcfg, P)
    rb.stage_translate(rcfg, fcfg, P)
    rb.stage_report(rcfg, P, False)


# ====================================================================================== main
def run_stage(env: Env, stage: str) -> None:
    if stage in ("decide_factors", "comb", "confirm_comb", "final_retrain"):
        fns = {"decide_factors": stage_decide_factors, "comb": stage_comb,
               "confirm_comb": stage_confirm_comb,
               "final_retrain": stage_final_retrain}  # fmt: skip
        fns[stage](env)
        return
    head, cand = stage.split("_", 1)
    rec = comb_recipe(adopted_factors(env)) if cand == "comb" else factor_recipe(cand)
    if head == "guard":
        run_guard(env, rec)
    elif head == "folds":
        stage_folds(env, rec)
    elif head == "trackb":
        stage_trackb(env, rec)
    else:  # cand
        run_candidate_pipeline(env, rec)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Robustness and open-set training fixes")
    ap.add_argument("--config", default="configs/phase4c.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: fold 0, 1st DEV class, 1 epoch, stand-in inputs")  # fmt: skip
    ap.add_argument("--smoke-adopt", default=None,
                    help="smoke only: factors treated as adopted, e.g. a1,a2")  # fmt: skip
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)
    env = make_env(args)
    run_stage(env, args.stage)


if __name__ == "__main__":
    main()
