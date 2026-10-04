"""Open-set improvement round selection: the five axes with paired bootstrap CIs and the
pre-registered rule.

PURE over saved DEV / CV outputs (results/phase6a/<weights>/s*/oof_*.csv and the view tables
results/phase6a/<candidate>/s*/views/<mode>/*.csv) plus the combos choice file. It never imports or
reads the D1 oracle diagnostics, the CONFIRM / headline outputs or any model; a test enforces it
(static scan of this file's imports and string constants, and a run in a directory without them).

Axes (candidate vs ref, paired, 10,000 resamples, seed 42, percentile 95% CI; pre-registered):
a CV macro-F1, b Track B DEV AUROC + strict rejection@95 in BOTH raw and ID-neutral mode,
c neutral-swap flip rate, d translation agreement, e OOF ECE after temperature scaling.
Candidates whose weights are the ref's share axes a, c, d, e exactly (delta 0, never improved).
CV macro-F1 of every candidate is read from its OOF clean predictions (the e*-stopped fold models),
uniformly, instead of from training curves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intent_router import phase4c as p4
from intent_router import phase4e as p5
from intent_router import trackb_improve as ti
from intent_router.evaluate import git_sha
from intent_router.phase6a_views import MODES, view_dir
from intent_router.stats import macro_f1

OOF_NAMES = p5.OOF_NAMES
AXES = ("a", "b", "c", "d", "e")
REF = "ref"
# Retention guard (added after the first selection picked a collapsed operating point):
# DEV-mean retention of known eval rows at the candidate's own threshold
# must stay within this of ref's, in BOTH modes (a strict-rejection gain from a collapsed
# operating point is not an open-set improvement).
RETENTION_MAX_DROP = 0.02
CHOICE_FILE = "combos_choice.json"


# ================================================================================== context
@dataclass
class SelCtx:
    """Everything the selection needs (so tests can build one without the training stack)."""

    results: Path
    pcfg: Mapping[str, Any]
    dev_classes: list[str]
    seeds: list[int]
    folds: list[int]
    safe_labels: list[str]
    n_boot: int
    smoke: bool = False


def resolve_candidates(
    pcfg: Mapping[str, Any], choice: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Candidate specs from the config, with combo placeholders (i3x / i1x / i6x) resolved from the
    combos choice. A combo that could not be formed gets `formed: False` (reported, not scored)."""
    out = []
    for c in pcfg["candidates"]:
        spec = dict(c)
        if spec.get("combo"):
            ch = (choice or {}).get("combos", {}).get(spec["key"])
            if ch is None:
                spec |= {"formed": False, "reason": "combos stage has not run"}
            elif not ch["formed"]:
                spec |= {"formed": False, "reason": ch["reason"]}
            else:
                spec |= {
                    "formed": True,
                    "weights": ch["weights"],
                    "scorer": ch["scorer"],
                    "thr": ch["thr"],
                }
        else:
            spec["formed"] = True
        out.append(spec)
    return out


# ============================================================================== data loading
@dataclass
class CVData:
    """CV-side arrays of one weight set (axes a, c, d, e)."""

    key: str
    f1: np.ndarray  # [S, F] held-out macro-F1 per (seed, fold) from the OOF clean predictions
    flips: np.ndarray  # [S, n_clusters]
    inst: np.ndarray  # [n_clusters]
    agree: np.ndarray  # [S, n_ids, L]
    kept: np.ndarray  # [n_ids, L]
    probs: np.ndarray  # [S, n_rows, K]
    gold: np.ndarray
    keys: dict[str, Any] = field(default_factory=dict)


def seed_dir(sc: SelCtx, name: str, seed: int) -> Path:
    return sc.results / name / f"s{seed}"


def load_cv(sc: SelCtx, key: str) -> CVData:
    """Read one weight set's saved OOF CSVs (the e*-stopped fold models' predictions)."""
    seeds, langs = sc.seeds, list(sc.pcfg["translation"]["langs"])
    oof = {
        s: {n: pd.read_csv(seed_dir(sc, key, s) / f"oof_{n}.csv") for n in OOF_NAMES} for s in seeds
    }
    f1 = np.array(
        [
            [
                macro_f1(g["gold"].to_numpy(), g["pred"].to_numpy())
                for g in (oof[s]["clean"][oof[s]["clean"]["fold"] == k] for k in sc.folds)
            ]
            for s in seeds
        ]
    )
    pcols = sorted(
        (c for c in oof[seeds[0]]["clean"].columns if c.startswith("prob_")),
        key=lambda c: int(c.split("_")[1]),
    )
    clean0 = oof[seeds[0]]["clean"].sort_values("id").reset_index(drop=True)
    probs, gold = [], clean0["gold"].to_numpy()
    for s in seeds:
        c = oof[s]["clean"].sort_values("id").reset_index(drop=True)
        if not (c["id"].equals(clean0["id"]) and (c["gold"].to_numpy() == gold).all()):
            raise ValueError(f"{key}: OOF clean rows differ between seeds")
        probs.append(c[pcols].to_numpy(dtype=np.float64))
    clusters = sorted(set(oof[seeds[0]]["swap"]["id"]))
    flips, inst = [], None
    for s in seeds:
        sw = oof[s]["swap"].assign(flip=lambda d: (d["swap_pred"] != d["clean_pred"]).astype(float))
        if sorted(set(sw["id"])) != clusters:
            raise ValueError(f"{key}: swap id clusters differ between seeds")
        g = sw.groupby("id").agg(flips=("flip", "sum"), n=("flip", "size")).reindex(clusters)
        n = g["n"].to_numpy(dtype=float)
        if inst is not None and not np.array_equal(inst, n):
            raise ValueError(f"{key}: swap instance counts differ between seeds")
        inst = n
        flips.append(g["flips"].to_numpy(dtype=float))
    mt_ids = sorted(set(oof[seeds[0]]["mt"]["id"]))
    agree, kept = [], None
    pos = {i: n for n, i in enumerate(mt_ids)}
    for s in seeds:
        mt = p4.mt_with_bool(oof[s]["mt"])
        a, k = np.zeros((len(mt_ids), len(langs))), np.zeros((len(mt_ids), len(langs)))
        for r in mt.itertuples(index=False):
            a[pos[r.id], langs.index(r.lang)] = float(r.mt_pred == r.en_pred)
            k[pos[r.id], langs.index(r.lang)] = float(r.kept)
        if kept is not None and not np.array_equal(kept, k):
            raise ValueError(f"{key}: LaBSE-kept mask differs between seeds")
        kept = k
        agree.append(a)
    keys = {"clean": tuple(clean0["id"]), "swap": tuple(clusters), "mt": tuple(mt_ids)}
    assert inst is not None and kept is not None
    return CVData(
        key, f1, np.array(flips), inst, np.array(agree), kept, np.array(probs), gold, keys
    )


def unit_triplet(table: pd.DataFrame, safe_labels: Sequence[str]) -> dict[str, ti.BootUnit]:
    """BootUnits of one view table: `score` (for AUROC) and the margin units at 95% / 90%
    retention (accept <=> margin >= 0, so the threshold handed to the bootstrap is 0)."""
    ev = table[table["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool)
    safe = ev.loc[unk, "pred"].isin(list(safe_labels)).to_numpy()
    out = {
        "score": ti.BootUnit(
            ev.loc[~unk, "score"].to_numpy(np.float64),
            ev.loc[unk, "score"].to_numpy(np.float64),
            safe,
            {0.95: 0.0},
        )
    }
    for ret, col in ((0.95, "margin95"), (0.90, "margin90")):
        out[f"m{round(ret * 100)}"] = ti.BootUnit(
            ev.loc[~unk, col].to_numpy(np.float64),
            ev.loc[unk, col].to_numpy(np.float64),
            safe,
            {ret: 0.0},
        )
    return out


def known_retention(table: pd.DataFrame) -> float:
    """Fraction of known evaluation rows accepted at the table's own threshold (margin95 >= 0)."""
    ev = table[table["set"] == "eval"]
    known = ev.loc[~ev["is_unknown"].astype(bool), "margin95"].to_numpy(np.float64)
    return float((known >= 0).mean()) if len(known) else float("nan")


@dataclass
class DevView:
    """Saved DEV view tables of one candidate: mode -> [class][seed] -> unit triplet."""

    name: str
    units: dict[str, list[list[dict[str, ti.BootUnit]]]]
    fingerprint: dict[str, list[str]]  # mode -> per-class row-identity hash (pairing check)
    # mode -> DEV-mean (over classes, seeds) fraction of known eval rows accepted (margin95 >= 0)
    retention: dict[str, float] = field(default_factory=dict)


def load_dev_view(sc: SelCtx, name: str) -> DevView:
    """Read candidate `name`'s view tables (every DEV class x seed x mode; missing => error)."""
    units: dict[str, list[list[dict[str, ti.BootUnit]]]] = {}
    fps: dict[str, list[str]] = {}
    ret: dict[str, float] = {}
    for mode in MODES:
        per_class: list[list[dict[str, ti.BootUnit]]] = []
        rets: list[float] = []
        fp_class: list[str] = []
        for cls in sc.dev_classes:
            per_seed, fp = [], None
            for s in sc.seeds:
                path = view_dir(sc.results, name, s, mode) / f"{name}_loco_{cls}_s{s}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"missing view table {path}")
                t = pd.read_csv(path)
                this = p5.hash_ids(t[["id", "set"]].astype(str).agg("|".join, axis=1).tolist())
                if fp is not None and fp != this:
                    raise ValueError(f"{name}: rows of {cls} differ between seeds ({mode})")
                fp = this
                per_seed.append(unit_triplet(t, sc.safe_labels))
                rets.append(known_retention(t))
            per_class.append(per_seed)
            fp_class.append(str(fp))
        units[mode], fps[mode] = per_class, fp_class
        ret[mode] = float(np.mean(rets))
    return DevView(name, units, fps, ret)


# ================================================================================= statistics
def b_stats(
    view: DevView, mode: str, n_boot: int, seeds: Sequence[int], level: float
) -> dict[str, tuple[float, np.ndarray]]:
    """DEV-mean AUROC and strict rejection@95: (point, per-resample samples). Rows are resampled
    within each class stratified known / unknown (draws shared by the seeds of a class), seeds are
    averaged inside each resample, then the DEV mean is taken. AUROC comes from the raw score unit,
    rejection from the margin unit (identical draws: same seed and same row counts)."""
    cls_units = view.units[mode]
    auroc = [
        ti.bootstrap_mean_ci([u["score"] for u in per_seed], n_boot, s, True, level)
        for per_seed, s in zip(cls_units, seeds, strict=True)
    ]
    rej = [
        ti.bootstrap_mean_ci([u["m95"] for u in per_seed], n_boot, s, True, level)
        for per_seed, s in zip(cls_units, seeds, strict=True)
    ]
    return {
        "auroc": (
            float(np.mean([r["point"]["auroc"] for r in auroc])),
            np.mean([r["samples"]["auroc"] for r in auroc], axis=0),
        ),
        "rej95": (
            float(np.mean([r["point"]["strict_recall_95"] for r in rej])),
            np.mean([r["samples"]["strict_recall_95"] for r in rej], axis=0),
        ),
    }


def cv_stats(
    cv: CVData, idx_c: np.ndarray, idx_d: np.ndarray, idx_e: np.ndarray, n_bins: int
) -> dict[str, Any]:
    """Axis c / d / e (point, samples) of one weight set; T per seed."""
    cal = [p5.fit_calibration(cv.probs[s], cv.gold, n_bins) for s in range(cv.probs.shape[0])]
    return {
        "c": (
            p5.flip_rate_point(cv.flips, cv.inst),
            p5.flip_rate_samples(cv.flips, cv.inst, idx_c),
        ),
        "d": (
            p5.agreement_point(cv.agree, cv.kept),
            p5.agreement_samples(cv.agree, cv.kept, idx_d),
        ),
        "e": (
            float(np.mean([c["ece"] for c in cal])),
            np.mean([p5.ece_samples(c["conf"], c["correct"], idx_e, n_bins) for c in cal], axis=0),
        ),
        "T": [c["T"] for c in cal],
    }


def check_aligned(ref: CVData, cd: CVData) -> None:
    """Paired axes need identical rows: refuse otherwise (never silently mis-pair)."""
    if cd.keys != ref.keys:
        raise ValueError(f"{cd.key} and {ref.key} are not row-aligned")
    if cd.f1.shape != ref.f1.shape or cd.flips.shape != ref.flips.shape:
        raise ValueError(f"{cd.key} and {ref.key} differ in (seed, fold) / cluster shape")


def b_axis(
    cand: Mapping[str, Mapping[str, tuple[float, np.ndarray]]],
    ref: Mapping[str, Mapping[str, tuple[float, np.ndarray]]],
    th: Mapping[str, float],
    level: float,
    retention: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Axis (b): per mode paired records; eligible iff BOTH modes pass both conditions AND the
    retention guard (`retention` = mode -> (candidate, ref); None skips it, tests
    only); improved iff any of the four (AUROC / rej@95 x raw / neutral) CIs excludes 0 in the
    good direction."""
    out: dict[str, Any] = {"modes": {}}
    eligible, improved = True, False
    guard: bool | None = None if retention is None else True
    for mode in MODES:
        au = p5.paired_record(cand[mode]["auroc"], ref[mode]["auroc"], True, level)
        rj = p5.paired_record(cand[mode]["rej95"], ref[mode]["rej95"], True, level)
        ok = p5.eligible_drop(au["delta"], th["b_max_auroc_drop"]) and p5.eligible_drop(
            rj["delta"], th["b_max_rej95_drop"]
        )
        out["modes"][mode] = {
            "auroc": au,
            "rej95": rj,
            "eligible": bool(ok),
            "improved": bool(au["improved"] or rj["improved"]),
        }
        if retention is not None:
            c_ret, r_ret = retention[mode]
            g_ok = p5.eligible_drop(c_ret - r_ret, RETENTION_MAX_DROP)
            out["modes"][mode]["retention"] = {
                "candidate": c_ret,
                "ref": r_ret,
                "delta": c_ret - r_ret,
                "guard_ok": bool(g_ok),
            }
            guard = bool(guard and g_ok)
            ok = ok and g_ok
            out["modes"][mode]["eligible"] = bool(ok)
        eligible &= ok
        improved |= au["improved"] or rj["improved"]
    out |= {
        "eligible": bool(eligible),
        "retention_guard": guard,
        "improved": bool(improved),
        "rule": f"(dAUROC >= -{th['b_max_auroc_drop']} and dRej95 >= "
        f"-{th['b_max_rej95_drop']}) in raw AND neutral; retention guard: known-row retention "
        f">= ref - {RETENTION_MAX_DROP} in raw AND neutral",
        "neutral_rej95": out["modes"]["neutral"]["rej95"]["candidate"],
    }
    return out


def candidate_axes(
    cv: CVData,
    ref_cv: CVData,
    cv_st: Mapping[str, Any],
    ref_cv_st: Mapping[str, Any],
    b_cand: Mapping[str, Any],
    b_ref: Mapping[str, Any],
    idx_a: np.ndarray,
    th: Mapping[str, float],
    level: float,
    retention: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """All five axes of one candidate vs ref."""
    check_aligned(ref_cv, cv)
    ax: dict[str, Any] = {"a": p5.axis_a(ref_cv.f1, cv, idx_a, th, level)}  # type: ignore[arg-type]
    ax["b"] = b_axis(b_cand, b_ref, th, level, retention)
    for name, rise in (("c", th["c_max_flip_increase"]), ("e", th["e_max_ece_increase"])):
        rec = p5.paired_record(cv_st[name], ref_cv_st[name], False, level)
        rec |= {
            "eligible": p5.eligible_rise(rec["candidate"], rec["v1"], rise),
            "rule": f"candidate <= ref + {rise}",
        }
        ax[name] = rec
    rec = p5.paired_record(cv_st["d"], ref_cv_st["d"], True, level)
    rec |= {
        "eligible": p5.eligible_drop(rec["delta"], th["d_max_agreement_drop"]),
        "rule": f"delta >= -{th['d_max_agreement_drop']}",
    }
    ax["d"] = rec
    return ax


# ============================================================================ selection rule
@dataclass(frozen=True)
class SelRow:
    """One candidate as the selection rule sees it."""

    key: str
    eligible: bool
    improved: tuple[str, ...]  # improved axes (axis b counted once)
    n_components: int
    neutral_rej95: float  # ID-neutral DEV strict rejection@95 point estimate (tie-break 1)


def select_candidate(rows: Sequence[SelRow]) -> dict[str, Any]:
    """Pre-registered "Selection" rule: among eligible non-ref candidates with >= 1 improved axis,
    the most improved axes; tie-break 1: higher ID-neutral DEV strict rejection@95; tie-break 2:
    fewer components; a remaining exact tie goes to the first candidate in input order and is
    flagged. Nobody qualifies -> ref. Pure; `path` records every narrowing step."""
    pool = [r for r in rows if r.key != REF and r.eligible and r.improved]
    path: list[dict[str, Any]] = [
        {"step": "eligible_with_>=1_improved_axis", "kept": [r.key for r in pool]}
    ]
    if not pool:
        return {
            "chosen": REF,
            "path": path,
            "unresolved_tie": False,
            "reason": "no eligible candidate with an improved axis; ref (v1) stays",
        }
    nan_last = lambda v: v if np.isfinite(v) else -np.inf  # noqa: E731 - fail closed on NaN
    for step, key_fn, best in (
        ("most_improved_axes", lambda r: len(r.improved), max),
        ("higher_neutral_rej95", lambda r: nan_last(r.neutral_rej95), max),
        ("fewer_components", lambda r: r.n_components, min),
    ):
        target = best(key_fn(r) for r in pool)
        pool = [r for r in pool if key_fn(r) == target]
        path.append({"step": step, "value": target, "kept": [r.key for r in pool]})
    tie = len(pool) > 1
    return {
        "chosen": pool[0].key,
        "path": path,
        "unresolved_tie": tie,
        "reason": "unresolved exact tie; first in candidate order" if tie else "selected",
    }


# ============================================================================== orchestration
def read_choice(results: Path) -> dict[str, Any] | None:
    p = results / CHOICE_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def compute_all(sc: SelCtx, specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """axes.json content: ref absolutes + every candidate's full axis table (eligible or not)."""
    pc = sc.pcfg
    th, level = pc["thresholds"], float(pc["bootstrap"]["level"])
    n_boot, seed, n_bins = (
        sc.n_boot,
        int(pc["bootstrap"]["seed"]),
        int(pc["calibration"]["ece_bins"]),
    )
    ref_cv = load_cv(sc, REF)
    idx_a = p4.boot_indices(ref_cv.f1.size, n_boot, seed)
    idx_c = p4.boot_indices(len(ref_cv.inst), n_boot, seed)
    idx_d = p4.boot_indices(ref_cv.kept.shape[0], n_boot, seed)
    idx_e = p4.boot_indices(len(ref_cv.gold), n_boot, seed)
    cls_seeds = p5.class_seeds(len(sc.dev_classes), seed)

    cv_cache: dict[str, tuple[CVData, dict[str, Any]]] = {
        REF: (ref_cv, cv_stats(ref_cv, idx_c, idx_d, idx_e, n_bins))
    }

    def cv_of(key: str) -> tuple[CVData, dict[str, Any]]:
        if key not in cv_cache:
            cd = load_cv(sc, key)
            cv_cache[key] = (cd, cv_stats(cd, idx_c, idx_d, idx_e, n_bins))
        return cv_cache[key]

    def b_of(view: DevView) -> dict[str, Any]:
        return {m: b_stats(view, m, n_boot, cls_seeds, level) for m in MODES}

    ref_view = load_dev_view(sc, REF)
    b_ref = b_of(ref_view)
    ref_st = cv_cache[REF][1]
    out: dict[str, Any] = {
        "ref": {
            "key": REF,
            "f1": float(ref_cv.f1.mean()),
            "auroc": {m: b_ref[m]["auroc"][0] for m in MODES},
            "rej95": {m: b_ref[m]["rej95"][0] for m in MODES},
            "retention": dict(ref_view.retention),
            "flip_rate": ref_st["c"][0],
            "agreement": ref_st["d"][0],
            "ece": ref_st["e"][0],
            "temperature_per_seed": ref_st["T"],
        },
        "candidates": {},
    }
    for spec in specs:
        key = spec["key"]
        base = {k: spec.get(k) for k in ("weights", "scorer", "thr", "components")}
        if not spec["formed"]:
            out["candidates"][key] = {"status": "not_formed", "reason": spec["reason"], **base}
            continue
        view = load_dev_view(sc, key)
        if view.fingerprint != ref_view.fingerprint:
            raise ValueError(f"{key}: DEV rows are not aligned with ref")
        cd, st = cv_of(spec["weights"])
        retention = {m: (view.retention[m], ref_view.retention[m]) for m in MODES}
        ax = candidate_axes(cd, ref_cv, st, ref_st, b_of(view), b_ref, idx_a, th, level, retention)
        improved = [a for a in AXES if ax[a]["improved"]]
        out["candidates"][key] = {
            "status": "scored",
            **base,
            "axes": ax,
            "eligible": all(ax[a]["eligible"] for a in AXES),
            "ineligible_axes": [a for a in AXES if not ax[a]["eligible"]],
            "improved_axes": improved,
            "n_improved": len(improved),
            "neutral_rej95": ax["b"]["neutral_rej95"],
            "retention_guard": ax["b"]["retention_guard"],
            "temperature_per_seed": st["T"],
        }
    out |= {
        "bootstrap": dict(pc["bootstrap"]) | {"n_resamples_used": n_boot},
        "seeds": sc.seeds,
        "folds": sc.folds,
        "dev_classes": sc.dev_classes,
        "thresholds": dict(th),
        "git_sha": git_sha(),
        "smoke": sc.smoke,
        "label": "measured; paired bootstrap, seeds averaged inside each resample; "
        "CV macro-F1 from OOF clean predictions of the e*-stopped fold models",
    }
    return out


def sel_rows(axes: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> list[SelRow]:
    """SelRows (spec order) of the scored candidates in a saved axes.json."""
    rows = []
    for spec in specs:
        c = axes["candidates"][spec["key"]]
        if c["status"] != "scored":
            continue
        rows.append(
            SelRow(
                spec["key"],
                bool(c["eligible"]),
                tuple(c["improved_axes"]),
                int(spec["components"]),
                float(c["neutral_rej95"]),
            )
        )
    return rows


def render_axes_table(axes: Mapping[str, Any]) -> str:
    """Markdown table of EVERY candidate x axis (numbers only), eligible or not."""

    def f(v: Any, nd: int = 4) -> str:
        return "-" if v is None else f"{v:+.{nd}f}"

    def cell(a: Mapping[str, Any], elig: bool | None = None) -> str:
        elig = a["eligible"] if elig is None else elig
        mark = ("ok" if elig else "INELIGIBLE") + (", improved" if a["improved"] else "")
        return f"{f(a['delta'])} [{f(a['lo'])}, {f(a['hi'])}] {mark}"

    def ret_cell(m: Mapping[str, Any]) -> str:
        g = m.get("retention")
        return "-" if g is None else f"{g['candidate']:.4f} / {f(g['delta'])}"

    r = axes["ref"]
    lines = [
        "## Open-set improvement round axes: candidate - ref "
        "(CI of the improvement; reductions for flip rate / ECE)",
        "",
        f"ref: F1 {r['f1']:.4f}, AUROC raw/neutral "
        f"{r['auroc']['raw']:.4f}/{r['auroc']['neutral']:.4f}, "
        f"rej@95 raw/neutral {r['rej95']['raw']:.4f}/{r['rej95']['neutral']:.4f}, "
        f"retention raw/neutral {r['retention']['raw']:.4f}/{r['retention']['neutral']:.4f}, flip "
        f"{r['flip_rate']:.4f}, agreement {r['agreement']:.4f}, ECE {r['ece']:.4f}",
        "",
        "| candidate | (a) CV F1 | (b) raw dAUROC | (b) raw dRej@95 | (b) neutral dAUROC | "
        "(b) neutral dRej@95 | (b) retention raw (cand / delta) | "
        "(b) retention neutral (cand / delta) | retention guard | "
        "(c) flip | (d) agreement | (e) ECE | eligible | improved axes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in axes["candidates"].items():
        if c["status"] != "scored":
            lines.append(f"| {key} | not formed: {c['reason']} |" + " |" * 12)
            continue
        a = c["axes"]
        raw, neu = a["b"]["modes"]["raw"], a["b"]["modes"]["neutral"]
        be = a["b"]["eligible"]
        cells = [
            cell(a["a"]),
            cell(raw["auroc"], be),
            cell(raw["rej95"], be),
            cell(neu["auroc"], be),
            cell(neu["rej95"], be),
            ret_cell(raw),
            ret_cell(neu),
            str(a["b"]["retention_guard"]),
            cell(a["c"]),
            cell(a["d"]),
            cell(a["e"]),
        ]
        lines.append(
            f"| {key} | {' | '.join(cells)} | {c['eligible']} | "
            f"{','.join(c['improved_axes']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def stage_select(sc: SelCtx, out_dir: Path) -> dict[str, Any]:
    """axes.json + selection.json + axes_table.md from the saved DEV / CV outputs only."""
    specs = resolve_candidates(sc.pcfg, read_choice(sc.results))
    axes = compute_all(sc, specs)
    p5.write_json(out_dir / "axes.json", axes)
    (out_dir / "axes_table.md").write_text(render_axes_table(axes), encoding="utf-8")
    main = select_candidate(sel_rows(axes, specs))
    rows = sel_rows(axes, specs)
    out = {
        "chosen": main["chosen"],
        "path": main["path"],
        "unresolved_tie": main["unresolved_tie"],
        "reason": main["reason"],
        "eligible": [r.key for r in rows if r.eligible],
        "ineligible": {
            k: c["ineligible_axes"]
            for k, c in axes["candidates"].items()
            if c["status"] == "scored" and not c["eligible"]
        },
        "not_formed": {
            k: c["reason"] for k, c in axes["candidates"].items() if c["status"] != "scored"
        },
        "improved_axes": {r.key: list(r.improved) for r in rows},
        "n_improved_axes": {r.key: len(r.improved) for r in rows},
        "chosen_spec": next((s for s in specs if s["key"] == main["chosen"]), None),
        "axes": axes,
        "git_sha": git_sha(),
        "smoke": sc.smoke,
    }
    p5.write_json(out_dir / "selection.json", out)
    print(
        f"[select] chosen: {out['chosen']} (eligible {out['eligible']}, "
        f"improved axes {out['n_improved_axes']}, not formed {list(out['not_formed'])})"
    )
    return out
