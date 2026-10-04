"""Loader layer for the report: reads results JSON/CSV/JSONL and exposes one flat metric dict.

Every number the report prints is registered here with a key, a format kind, the results file it
came from and the JSON path inside it. The template never contains a numeric literal: it asks for
`n("key")`, and the renderer emits `<span data-k="key">formatted value</span>` plus a footnote id
into the provenance table. No dataset text is read here (the optional `--include-text` examples
live in `load_examples`, which `build.py` calls only when that flag is given).
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- formatting

Number = float | int


def _p3(v: float) -> str:
    """p-values: 3 decimals, scientific below 0.001 (so tiny p-values are not shown as 0.000)."""
    return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"


def _ci(d: dict[str, float], signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['point'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


FORMATS: dict[str, Any] = {
    "f1": lambda v: f"{v:.1f}",
    "f2": lambda v: f"{v:.2f}",
    "f3": lambda v: f"{v:.3f}",
    "f4": lambda v: f"{v:.4f}",
    "d3": lambda v: f"{v:+.3f}",
    "d4": lambda v: f"{v:+.4f}",
    "int": lambda v: f"{int(v):d}",
    "pct": lambda v: f"{100 * v:.1f}%",  # fraction -> percent
    "pct0": lambda v: f"{100 * v:.0f}%",
    "pctpts": lambda v: f"{v:.1f}%",  # value already in percent units
    "mb": lambda v: f"{v / 1e6:.0f}",
    "gb": lambda v: f"{v / 1e9:.2f}",
    "mparams": lambda v: f"{v / 1e6:.0f}",
    "ms_from_s": lambda v: f"{1000 * v:.0f}",
    "ms": lambda v: f"{v:.0f}",
    "s1": lambda v: f"{v:.1f}",
    "sci": lambda v: f"{v:.1e}",
    "pval": _p3,
    "usd": lambda v: f"${v:.4f}",
    "usd2": lambda v: f"${v:.2f}",
    "usd5": lambda v: f"${v:.5f}",  # CPU cost per 1k messages is a fraction of a cent
    "ci": _ci,
    "dci": lambda d: _ci(d, signed=True),
    "pm3": lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}",
    "per1000": lambda v: f"{v:.1f}",
    "text": lambda v: str(v),
    "short": lambda v: str(v)[:8],
    "bool": lambda v: "yes" if v else "no",
}


def fmt(kind: str, value: Any) -> str:
    """Format a metric value by kind (the single formatting authority; tests re-use it)."""
    return str(FORMATS[kind](value))


@dataclass(frozen=True)
class Metric:
    """One rendered value with its provenance."""

    key: str
    value: Any
    kind: str
    file: str
    path: str
    fam: str
    computed: bool = False

    @property
    def text(self) -> str:
        return fmt(self.kind, self.value)


def _segments(path: str | tuple[Any, ...] | list[Any]) -> list[Any]:
    """'a.b[2].c' -> ['a', 'b', 2, 'c']; a tuple/list is used verbatim (keys may contain dots)."""
    if not isinstance(path, str):
        return list(path)
    out: list[Any] = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[\d+\])*)", part)
        if m is None:
            raise ValueError(f"bad path segment {part!r}")
        if m.group(1):
            out.append(m.group(1))
        out.extend(int(i) for i in re.findall(r"\[(\d+)\]", m.group(2)))
    return out


def _path_str(segs: list[Any]) -> str:
    s = "$"
    for g in segs:
        if isinstance(g, int):
            s += f"[{g}]"
        elif re.fullmatch(r"\w+", g):
            s += f".{g}"
        else:
            s += f"['{g}']"
    return s


def _resolve(obj: Any, segs: list[Any]) -> Any:
    for s in segs:
        obj = obj[s]
    return obj


def git_head(root: Path = ROOT) -> dict[str, Any]:
    """Build-time HEAD sha and whether the working tree has uncommitted changes."""

    def run(*a: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *a], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    full = run("rev-parse", "HEAD")
    return {"full": full, "short": full[:7], "dirty": bool(run("status", "--porcelain"))}


# --------------------------------------------------------------------------- loader


@dataclass
class Report:
    """Everything the template needs: metrics (flat), structured table data and figure sources."""

    metrics: dict[str, Metric] = field(default_factory=dict)
    ctx: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, list[str]] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)
    file_sha: dict[str, tuple[str, str]] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        """key -> raw value (the flat dict the tests and the provenance JSON are built from)."""
        return {k: m.value for k, m in self.metrics.items()}

    def provenance(self) -> list[dict[str, str]]:
        """One row per metric key: results file, JSON path, git sha of the results."""
        rows = []
        for m in self.metrics.values():
            sha, src = self.file_sha[m.file]
            rows.append(
                {
                    "key": m.key,
                    "family": m.fam,
                    "file": m.file,
                    "path": m.path,
                    "sha": sha,
                    "sha_source": src,
                    "kind": m.kind,
                    "computed": str(m.computed).lower(),
                    "text": m.text,
                }
            )
        return rows


class Loader:
    """Reads results files once and registers metrics."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.rep = Report()
        self.rep.head = git_head(root)
        self._cache: dict[str, Any] = {}

    # -- file access
    def j(self, rel: str) -> Any:
        if rel not in self._cache:
            self._cache[rel] = json.loads((self.root / rel).read_text(encoding="utf-8"))
        return self._cache[rel]

    def _sha(self, rel: str) -> tuple[str, str]:
        if rel in self.rep.file_sha:
            return self.rep.file_sha[rel]
        sha = None
        if rel.endswith(".json"):
            d = self.j(rel)
            if isinstance(d, dict):
                sha = d.get("git_sha") or (d.get("provenance") or {}).get("git_sha")
        if sha:
            res = (str(sha), "recorded in results file")
        else:
            h = self.rep.head
            res = (h["short"] + ("+dirty" if h["dirty"] else ""), "build HEAD (none recorded)")
        self.rep.file_sha[rel] = res
        return res

    # -- registration
    def _store(self, m: Metric) -> None:
        if m.key in self.rep.metrics:
            raise KeyError(f"duplicate metric key {m.key}")
        try:
            fmt(m.kind, m.value)  # fail at registration, naming the key, not at render time
        except (TypeError, ValueError, KeyError) as e:
            raise TypeError(
                f"metric {m.key!r}: value {m.value!r} does not fit kind {m.kind!r}"
            ) from e
        self._sha(m.file)
        self.rep.metrics[m.key] = m

    def add(
        self,
        key: str,
        rel: str,
        path: str | tuple[Any, ...],
        kind: str = "f3",
        fam: str | None = None,
        point_key: str = "point",
        optional: bool = False,
    ) -> Any:
        segs = _segments(path)
        try:
            v = _resolve(self.j(rel), segs)
        except (KeyError, IndexError, TypeError):
            if optional:
                return None
            raise
        if kind in ("ci", "dci") and isinstance(v, dict):
            v = {"point": v[point_key], "lo": v["lo"], "hi": v["hi"]}
        if kind == "pm3" and isinstance(v, dict):
            v = {"mean": v["mean"], "std": v["std"]}
        self._store(Metric(key, v, kind, rel, _path_str(segs), fam or key))
        return v

    def add_pair(
        self,
        key: str,
        rel: str,
        point_path: str | tuple[Any, ...],
        ci_path: str | tuple[Any, ...],
        kind: str = "dci",
        fam: str | None = None,
    ) -> None:
        """Point estimate and a [lo, hi] list stored elsewhere in the file."""
        ps, cs = _segments(point_path), _segments(ci_path)
        pt = _resolve(self.j(rel), ps)
        lo, hi = _resolve(self.j(rel), cs)
        d = {"point": pt, "lo": lo, "hi": hi}
        self._store(Metric(key, d, kind, rel, f"{_path_str(ps)} + {_path_str(cs)}", fam or key))

    def put(
        self, key: str, value: Any, kind: str, rel: str, desc: str, fam: str | None = None
    ) -> Any:
        """A value computed from a results file by code (counts, maxima, ratios)."""
        self._store(Metric(key, value, kind, rel, f"computed: {desc}", fam or key, computed=True))
        return value

    def exists(self, rel: str) -> bool:
        """
        Whether a results file exists (Open-set improvement round sections are driven by this).
        """
        return (self.root / rel).exists()

    def val(self, rel: str, path: str | tuple[Any, ...]) -> Any:
        return _resolve(self.j(rel), _segments(path))

    def val_opt(self, rel: str, path: str | tuple[Any, ...]) -> Any:
        """Like val(), but None when the key is absent (a clean latency run omits some blocks)."""
        try:
            return _resolve(self.j(rel), _segments(path))
        except (KeyError, IndexError, TypeError):
            return None

    def fig(self, name: str, sources: list[str]) -> None:
        self.rep.figures[name] = sources
        for s in sources:
            self._sha(s)


# --------------------------------------------------------------------------- constants

DISPLAY_METHOD = {
    "msp": "MSP (softmax)",
    "msp_temp": "MSP, temperature-scaled",
    "max_logit": "max logit",
    "neg_energy": "negative energy",
    "maha_ft": "Mahalanobis, fine-tuned features (shipped)",
    "knn1_ft": "kNN k=1, fine-tuned features",
    "knn5_ft": "kNN k=5, fine-tuned features",
    "maha_frozen": "Mahalanobis, frozen e5 features",
    "knn1_frozen": "kNN k=1, frozen e5 features",
    "knn5_frozen": "kNN k=5, frozen e5 features",
}
SLICES = [
    ("all", "all rows"),
    ("english", "English"),
    ("non_english_defA", "non-English (primary language)"),
    ("lang_es", "Spanish"),
    ("lang_fr", "French"),
    ("lang_de", "German"),
    ("lang_zh", "Chinese"),
    ("undetermined_lang", "language undetermined"),
    ("code_mixed", "code-mixed"),
    ("short_lt30_chars", "short texts"),
    ("noisy", "noisy"),
    ("clean", "clean"),
    ("shipment_family", "shipment_information.* family"),
]
ABLATIONS = [
    ("class_weight", "class-weighted CE"),
    ("label_smoothing", "label smoothing"),
    ("lld", "layer-wise LR decay"),
    ("entity_mask", "entity masking"),
    ("hier_aux", "hierarchical auxiliary loss"),
]
PERTURBATIONS = [
    ("char_swap_5", "character swaps, low rate"),
    ("char_mixed_5", "mixed character noise, low rate"),
    ("char_swap_10", "character swaps, high rate"),
    ("char_mixed_10", "mixed character noise, high rate"),
    ("lowercase", "lower-casing"),
    ("abbrev", "abbreviation substitution"),
]
SWAPS = ["PO->LD", "PO->REF", "LD->PO", "LD->REF", "other->REF"]
LANGS = [("es", "Spanish"), ("fr", "French"), ("de", "German"), ("zh", "Chinese")]
LLM_CONFIGS = [
    ("qwen3|zero", "qwen3:8b, zero-shot"),
    ("qwen3|five", "qwen3:8b, few-shot"),
    ("llama3|zero", "llama3.1:8b, zero-shot"),
    ("llama3|five", "llama3.1:8b, few-shot"),
]
LOG_PURPOSE = {
    "evaluation": "metric-producing evaluation",
    "robustness_inference": "robustness (ID swap, noise, translation), no Track A metric",
    "baseline_inference": "LLM baseline inference",
    "quantization_inference": "int8 / ONNX quantization comparison",
    "latency_inference": "latency measurement",
    "determinism_inference": "bit-for-bit determinism re-run",
    "hub_roundtrip_inference": "Hub round-trip reproduction check",
}

F = "results/final/"
TA = F + "track_a.json"
TB = "results/trackb/"
TBI = "results/trackb_improve/"
P4E = "results/phase4e/"
P4C = "results/phase4c/"
SRV = "results/serving/"
LLM = "results/llm_baseline/summary.json"
EDA = "results/eda.json"
TRADE = "results/tradeoff_v1_v3.json"
SEL = "results/bakeoff/selection.json"
BASE = "results/bakeoff/baselines.json"
ABL = "results/bakeoff/ablations.json"
BSUM = "results/bakeoff/summary.json"
ERR = F + "error_analysis.json"
HEAD = TB + "headline.json"
LOCO = TB + "loco.json"
CONF4E = P4E + "confirm_a1a3.json"
AX = P4E + "axes.json"
LOG_FINAL = F + "test_eval_log.jsonl"
OTHER_LOGS = [
    "results/trackb/test_inference_log.jsonl",
    "results/trackb_improve/test_inference_log.jsonl",
    "results/phase4c/test_inference_log.jsonl",
    "results/phase4e/test_inference_log.jsonl",
    "results/phase4e/confirm_test_inference_log.jsonl",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- sections


OOF_CSV = "results/oof/e5_lr3e-05.csv"  # the CV config whose OOF predictions Track A reports


def _oof_per_class(L: Loader, labels: list[str]) -> list[dict[str, str]]:
    """Per-class P/R/F1/support of the probability-averaged OOF predictions (same rows as oof.f1).

    The saved results hold only the aggregate OOF macro-F1, so the per-class table is computed
    here with the definition of `evaluate.per_class_report` (0 where undefined) on the same
    predictions `final.py` scores: probabilities averaged over the 9 OOF predictions per id,
    argmax. The macro-F1 of the result is checked against the stored value (loud on mismatch).
    """
    import numpy as np

    k = len(labels)
    rows = _read_csv_rows(L.root / OOF_CSV)  # dicts: id, gold, prob_0..prob_{k-1}
    by_id: dict[str, list[list[float]]] = {}
    gold: dict[str, int] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append([float(r[f"prob_{i}"]) for i in range(k)])
        gold[r["id"]] = int(r["gold"])
    ids = sorted(by_id)
    y = np.array([gold[i] for i in ids])
    pred = np.array([int(np.argmax(np.mean(by_id[i], axis=0))) for i in ids])
    conf = np.zeros((k, k), dtype=float)
    for g, p in zip(y, pred, strict=True):
        conf[g, p] += 1
    tp, pred_n, true_n = np.diag(conf), conf.sum(axis=0), conf.sum(axis=1)
    prec = np.divide(tp, pred_n, out=np.zeros(k), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros(k), where=true_n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    stored = L.val(TA, "cv.oof_probability_averaged_426_rows.macro_f1.point")
    if abs(float(f1.mean()) - float(stored)) > 1e-9:
        raise ValueError(f"OOF per-class macro-F1 {f1.mean()} != stored {stored}")
    out = []
    for i, lab in enumerate(labels):
        r = {}
        for met, val, kd in (
            ("precision", prec[i], "f3"),
            ("recall", rec[i], "f3"),
            ("f1", f1[i], "f3"),
            ("support", int(true_n[i]), "int"),
        ):
            key = f"oofpc.{lab}.{met}"
            desc = f"per-class {met} of the probability-averaged OOF predictions ({OOF_CSV})"
            L.put(key, float(val) if kd == "f3" else val, kd, OOF_CSV, desc, fam=f"oofpc.*.{met}")
            r[met] = key
        out.append(r)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _track_a(L: Loader) -> None:
    c = L.rep.ctx
    L.add("a.n", TA, "test.n", "int")
    L.add("a.f1", TA, "test.macro_f1", "ci")
    L.add("a.acc", TA, "test.accuracy", "ci")
    L.add("a.boot.n", TA, "test.bootstrap.n_resamples", "int")
    L.add("a.boot.seed", TA, "test.bootstrap.seed", "int")
    L.add("a.boot.level", TA, "test.bootstrap.level", "pct0")
    L.add("a.parent_acc", TA, "test.hierarchy.parent_accuracy")
    L.add("a.n_errors", ERR, "test.n_errors", "int")
    L.add("a.val_f1", TA, "val_sanity.macro_f1")
    L.add("a.fp", TA, "model_fingerprint", "short")
    L.add("a.model_seed", F + "model_version.json", "train_config.model_seed", "int")
    L.add("a.epoch", F + "model_version.json", "deployed_epoch", "int")
    L.add("a.epochs_sched", F + "model_version.json", "train_config.epochs", "int")
    L.add("a.lr", F + "model_version.json", "train_config.lr", "sci")
    L.add("a.batch", F + "model_version.json", "train_config.batch_size", "int")
    L.add("a.maxlen", F + "model_version.json", "train_config.max_len", "int")
    L.add("a.n_train", F + "baselines_fit.json", "B0.n_train", "int")
    for b in ("B0", "B1"):
        L.add(f"a.{b}.f1", TA, f"test.baselines.{b}.macro_f1", "ci")
        L.add(f"a.{b}.acc", TA, f"test.baselines.{b}.accuracy", "ci")
        L.add(
            f"a.{b}.delta",
            TA,
            f"test.baselines.{b}.delta_macro_f1_final_minus_baseline",
            "dci",
            point_key="delta",
        )
        mc = f"test.baselines.{b}.mcnemar_exact_on_correctness"
        L.add(f"a.{b}.mc_a", TA, f"{mc}.a_only_correct", "int")
        L.add(f"a.{b}.mc_b", TA, f"{mc}.b_only_correct", "int")
        L.add(f"a.{b}.mc_p", TA, f"{mc}.p_value", "pval")
    # per-class table
    pcs = L.val(TA, "test.per_class")
    rows = []
    for i, pc in enumerate(pcs):
        r = {"label": pc["label"]}
        for m, kd in (("precision", "f3"), ("recall", "f3"), ("f1", "f3"), ("support", "int")):
            k = f"a.pc.{pc['label']}.{m}"
            L.add(k, TA, f"test.per_class[{i}].{m}", kd, fam=f"a.pc.*.{m}")
            r[m] = k
        rows.append(r)
    oof_pc = _oof_per_class(L, [x["label"] for x in pcs])
    for r, o in zip(rows, oof_pc, strict=True):
        r["oof"] = o
    c["perclass"] = rows
    sup = [x["support"] for x in pcs]
    L.put("a.sup_min", min(sup), "int", TA, "min over test.per_class[*].support")
    L.put("a.sup_max", max(sup), "int", TA, "max over test.per_class[*].support")
    worst = min(pcs, key=lambda x: (x["f1"], x["label"]))
    L.put("a.worst.f1", worst["f1"], "f3", TA, "min over test.per_class[*].f1")
    L.put("a.worst.support", worst["support"], "int", TA, "support of the min-F1 class")
    L.put("a.worst.label", worst["label"], "text", TA, "label of the min-F1 class")
    # calibration
    L.add("cal.T", TA, "calibration.temperature", "f3")
    for s in ("test", "val"):
        for w in ("before", "after"):
            L.add(f"cal.{s}.ece_{w}", TA, f"calibration.{s}.ece_{w}", "f3")
    sel = "test.selective"
    L.add("sel.threshold", TA, f"{sel}.at_threshold.threshold", "f3")
    L.add("sel.coverage", TA, f"{sel}.at_threshold.coverage", "f3")
    L.add("sel.n_accepted", TA, f"{sel}.at_threshold.n_accepted", "int")
    L.add("sel.acc", TA, f"{sel}.at_threshold.accuracy", "f3")
    L.add("sel.aurc", TA, f"{sel}.aurc_temp_scaled_msp", "f4")
    # CV / OOF (post-selection, optimistic)
    cv = "cv"
    L.add("cv.f1_mean", TA, f"{cv}.macro_f1_mean")
    L.add("cv.f1_std", TA, f"{cv}.macro_f1_std")
    L.add("cv.acc_mean", TA, f"{cv}.accuracy_mean")
    L.add("cv.acc_std", TA, f"{cv}.accuracy_std")
    L.add("cv.n_runs", TA, f"{cv}.n_fold_runs", "int")
    L.add("cv.epoch", TA, f"{cv}.chosen_epoch", "int")
    L.add("oof.f1", TA, f"{cv}.oof_probability_averaged_426_rows.macro_f1", "ci")
    L.add("oof.acc", TA, f"{cv}.oof_probability_averaged_426_rows.accuracy", "ci")
    L.add("oof.n", ERR, "oof.hypothesis_errors_concentrate_in_shipment_information.n_rows", "int")
    # fold structure counted from the per-fold records
    pf = L.val(BSUM, ("configs", "e5_lr3e-05", "per_fold"))
    L.put(
        "cv.n_fold_seeds",
        len({p["fold_seed_idx"] for p in pf}),
        "int",
        BSUM,
        "distinct fold_seed_idx in configs.e5_lr3e-05.per_fold",
    )
    L.put(
        "cv.n_folds",
        len({p["fold"] for p in pf}),
        "int",
        BSUM,
        "distinct fold in configs.e5_lr3e-05.per_fold",
    )
    L.put(
        "cv.n_model_seeds",
        max(p["n_model_seeds"] for p in pf),
        "int",
        BSUM,
        "max n_model_seeds in configs.e5_lr3e-05.per_fold",
    )
    # slices (OOF = 426 rows post-selection; test = n 74)
    srows = []
    for sid, name in SLICES:
        r: dict[str, Any] = {"name": name}
        for src in ("oof", "test"):
            d = L.val(F + "slices.json", ("sources", src, sid)) if _has(L, src, sid) else None
            if d is None:
                r[src] = None
                continue
            base = ("sources", src, sid)
            k = f"sl.{src}.{sid}"
            L.add(f"{k}.n", F + "slices.json", base + ("n",), "int", fam=f"sl.{src}.*.n")
            e: dict[str, Any] = {
                "n": f"{k}.n",
                "flag": bool(d["indicative_only"]),
                "f1": None,
                "acc": None,
            }
            if isinstance(d.get("macro_f1"), dict):
                mf = d["macro_f1"]
                # a resample that drops a class biases the interval low; where the point then
                # lies outside its own interval, show the point alone (marked in the table)
                e["f1_no_ci"] = not (mf["lo"] <= mf["point"] <= mf["hi"])
                L.add(
                    f"{k}.f1",
                    F + "slices.json",
                    base + (("macro_f1", "point") if e["f1_no_ci"] else ("macro_f1",)),
                    "f3" if e["f1_no_ci"] else "ci",
                    fam=f"sl.{src}.*.f1",
                )
                e["f1"] = f"{k}.f1"
            if isinstance(d.get("accuracy"), dict):
                L.add(
                    f"{k}.acc", F + "slices.json", base + ("accuracy",), "ci", fam=f"sl.{src}.*.acc"
                )
                e["acc"] = f"{k}.acc"
            r[src] = e
        srows.append(r)
    c["slices"] = srows
    L.add("sl.short_def", F + "slices.json", "short", "text")
    L.add("sl.indic_rule", F + "slices.json", "indicative_only_rule", "text")


def _has(L: Loader, src: str, sid: str) -> bool:
    return sid in L.val(F + "slices.json", ("sources", src))


def _bakeoff(L: Loader) -> None:
    c = L.rep.ctx
    rows: list[dict[str, Any]] = []
    for b, name in (("B0", "B0: char TF-IDF + LR"), ("B1", "B1: frozen e5 + LR")):
        r = {"name": name, "lr": None, "epoch": None, "params": None, "wall": None}
        L.add(f"bk.{b}.f1", BASE, f"{b}.macro_f1_mean", "f3")
        L.add(f"bk.{b}.f1s", BASE, f"{b}.macro_f1_std", "f3")
        L.add(f"bk.{b}.acc", BASE, f"{b}.accuracy_mean", "f3")
        L.add(f"bk.{b}.n", BASE, f"{b}.n_fold_runs", "int")
        L.add(f"bk.{b}.wall", BASE, f"{b}.wall_clock_s", "s1")
        r.update(
            f1=f"bk.{b}.f1",
            f1s=f"bk.{b}.f1s",
            acc=f"bk.{b}.acc",
            n=f"bk.{b}.n",
            wall=f"bk.{b}.wall",
        )
        rows.append(r)
    names = {
        "e5_lr3e-05": "F1: multilingual-e5-base (winner)",
        "xlmr_lr5e-05": "F2: xlm-roberta-base",
        "mdeberta_lr5e-05": "F3: mdeberta-v3-base",
    }
    for cid, name in names.items():
        k = f"bk.{cid}"
        base = ("candidates", cid)
        L.add(f"{k}.f1", SEL, base + ("macro_f1_mean",), "f3")
        L.add(f"{k}.f1s", SEL, base + ("macro_f1_std",), "f3")
        L.add(f"{k}.acc", SEL, base + ("accuracy_mean",), "f3")
        L.add(f"{k}.epoch", SEL, base + ("chosen_epoch",), "int")
        L.add(f"{k}.n", SEL, base + ("n_runs",), "int")
        L.add(f"{k}.params", SEL, base + ("params",), "mparams")
        L.add(f"{k}.wall", SEL, base + ("mean_wall_clock_s",), "s1")
        L.put(f"{k}.lr", cid.split("_lr")[1], "text", SEL, f"learning rate in the config id {cid}")
        rows.append(
            {
                "name": name,
                "lr": f"{k}.lr",
                "f1": f"{k}.f1",
                "f1s": f"{k}.f1s",
                "acc": f"{k}.acc",
                "epoch": f"{k}.epoch",
                "n": f"{k}.n",
                "params": f"{k}.params",
                "wall": f"{k}.wall",
            }
        )
    c["bakeoff"] = rows
    L.add("bk.win_threshold", SEL, "rule1_winner_std_threshold", "f3")
    L.add("bk.r2_gap", SEL, "rule2[0].gap_to_winner", "f3")
    L.add("bk.r2_within", SEL, "rule2[0].within_1_std", "bool")
    L.add("bk.nb_p", SEL, "top2_ttest.p", "pval")
    L.add("bk.nb_t", SEL, "top2_ttest.t", "f2")
    L.add("bk.nb_J", SEL, "top2_ttest.J", "int")
    L.add("bk.nb_p_B1", SEL, "baseline_context_ttests.B1.p", "pval")
    L.add("bk.nb_p_B0", SEL, "baseline_context_ttests.B0.p", "pval")
    # all bake-off configs (appendix)
    allrows = []
    for cid in L.val(BSUM, "configs"):
        k = f"bkall.{cid}"
        base = ("configs", cid)
        L.add(f"{k}.f1", BSUM, base + ("macro_f1_mean",), "f3", fam="bkall.*.f1")
        L.add(f"{k}.f1s", BSUM, base + ("macro_f1_std",), "f3", fam="bkall.*.f1s")
        L.add(f"{k}.epoch", BSUM, base + ("chosen_epoch",), "int", fam="bkall.*.epoch")
        L.add(f"{k}.n", BSUM, base + ("n_runs",), "int", fam="bkall.*.n")
        L.add(f"{k}.vram", BSUM, base + ("max_peak_vram_mb",), "ms", fam="bkall.*.vram")
        allrows.append(
            {
                "cid": cid,
                "f1": f"{k}.f1",
                "f1s": f"{k}.f1s",
                "epoch": f"{k}.epoch",
                "n": f"{k}.n",
                "vram": f"{k}.vram",
            }
        )
    c["bk_all"] = allrows
    # ablations
    arows = []
    for aid, name in ABLATIONS:
        k = f"abl.{aid}"
        base = ("ablations", aid)
        L.add(f"{k}.delta", ABL, base + ("delta_macro_f1",), "d4", fam="abl.*.delta")
        L.add(f"{k}.p", ABL, base + ("ttest", "p"), "pval", fam="abl.*.p")
        L.add(f"{k}.epoch", ABL, base + ("ablation_epoch",), "int", fam="abl.*.epoch")
        L.add(f"{k}.adopt", ABL, base + ("decision", "adopt"), "bool", fam="abl.*.adopt")
        arows.append(
            {
                "name": name,
                "delta": f"{k}.delta",
                "p": f"{k}.p",
                "epoch": f"{k}.epoch",
                "adopt": f"{k}.adopt",
            }
        )
    c["ablations"] = arows
    L.put("abl.n", len(ABLATIONS), "int", ABL, "number of ablations tabulated", fam="abl.n")
    L.add("abl.noise_band", ABL, "noise_band.value", "f4")
    L.add("abl.n_runs", ABL, "n_baseline_runs", "int")
    # post-hoc robustness check on the loser's grids
    sc = L.val("results/selection_check/summary.json", "configs")
    best = max(sc, key=lambda k: sc[k]["macro_f1_mean"])
    f = "results/selection_check/summary.json"
    L.put("sc.best.f1", sc[best]["macro_f1_mean"], "f3", f, "max over configs[*].macro_f1_mean")
    L.put("sc.best.name", best, "text", f, "argmax config name")
    L.put("sc.best.epoch", sc[best]["chosen_epoch"], "int", f, "chosen_epoch of the best config")
    L.add("sc.comparator", f, "comparator.exact_score", "f3")
    L.put("sc.n_configs", len(sc), "int", f, "len(configs)")
    L.put("sc.n_confirm", len(L.val(f, "decision.confirm")), "int", f, "len(decision.confirm)")


def _data_quirks(L: Loader) -> None:
    c = L.rep.ctx
    L.add("eda.n", EDA, "n_rows", "int")
    L.add("eda.k", EDA, "n_classes", "int")
    L.add("eda.imb", EDA, "imbalance_ratio_max_over_min", "f2")
    cc = L.val(EDA, "class_counts")
    L.put("eda.cc_min", min(cc.values()), "int", EDA, "min(class_counts)")
    L.put("eda.cc_max", max(cc.values()), "int", EDA, "max(class_counts)")
    for lab in ("chitchat", "other"):
        L.put(f"eda.cc.{lab}", cc[lab], "int", EDA, f"class_counts[{lab}]")
    L.add("a.wd", F + "model_version.json", "train_config.weight_decay", "f2")
    lg = "language"
    L.add("eda.nonen", EDA, f"{lg}.pct_non_english_primary_definition_A", "pctpts")
    L.add("eda.nonen_B", EDA, f"{lg}.pct_any_non_english_segment_definition_B", "pctpts")
    L.add("eda.claim", EDA, f"{lg}.pdf_claim_pct_non_english", "pctpts")
    L.add("eda.mixed", EDA, f"{lg}.pct_code_mixed", "pctpts")
    L.add("eda.undet", EDA, f"{lg}.undetermined_n", "int")
    for code in ("es", "fr", "de", "zh"):
        L.add(f"eda.lang.{code}", EDA, f"{lg}.primary_language_counts.{code}", "int")
    L.add("eda.short", EDA, "length.n_texts_under_15_chars", "int")
    L.add(
        "eda.p99_tok",
        EDA,
        ("length", "tokens_by_tokenizer", "intfloat/multilingual-e5-base", "p99"),
        "f1",
    )
    L.add("eda.maxlen", EDA, "length.max_len_decision.max_len", "int")
    L.add("eda.noisy", EDA, "noise.pct_is_noisy", "pctpts")
    L.add("eda.lower", EDA, "noise.pct_all_lowercase_among_rows_with_letters", "pctpts")
    L.add("eda.slang", EDA, "noise.pct_slang_rows_strict", "pctpts")
    nd = "near_duplicates"
    L.add("eda.dup_pairs", EDA, f"{nd}.n_pairs", "int")
    L.add("eda.dup_thr", EDA, f"{nd}.threshold", "f1")
    L.add("eda.dup_cross", EDA, f"{nd}.n_cross_label_pairs_above_cross_label_threshold", "int")
    L.add("eda.dup_cross_thr", EDA, f"{nd}.cross_label_threshold", "f1")
    L.add("eda.sib_1nn", EDA, "sibling_overlap.overall_1nn_accuracy", "f3")
    for tag, key in (("po", "PO-\\d+"), ("ld", "LD-\\d+")):
        a = ("shortcut_audit", key)
        L.add(f"eda.sc.{tag}.n", EDA, a + ("n_rows",), "int")
        L.add(f"eda.sc.{tag}.share", EDA, a + ("max_label_share",), "pct")
        L.add(f"eda.sc.{tag}.top", EDA, a + ("top_label",), "text")
        L.add(f"eda.sc.{tag}.risk", EDA, a + ("shortcut_risk",), "bool")
    c["quirk_langs"] = ["es", "fr", "de", "zh"]


def _protocol(L: Loader) -> None:
    c = L.rep.ctx
    # leakage audit of the Track B runs
    au = TB + "audit.json"
    L.add("aud.n_runs", au, "n_runs", "int")
    L.add("aud.ok", au, "all_assertions_passed", "bool")
    L.add("aud.once", au, "all_test_logs_exactly_one", "bool")
    # test-exposure statement, derived from the log files themselves
    final_log = _read_jsonl(L.root / LOG_FINAL)
    by_type = Counter(r["call_type"] for r in final_log)
    L._sha(LOG_FINAL)
    L.put("log.total", len(final_log), "int", LOG_FINAL, "number of log lines")
    fps = {}
    for ver, rel in (("v1", F), ("v2", "results/final_v2/"), ("v3", "results/final_v3/")):
        mv = L.root / rel / "model_version.json"
        if mv.exists():
            fps[json.loads(mv.read_text(encoding="utf-8"))["model_fingerprint"]] = ver
    evals = []
    for r in final_log:
        if r["call_type"] != "evaluation":
            continue
        fp = r["model_fingerprint"]
        ver = fps.get(fp)
        evals.append(
            {
                "ver": ver,
                "fp": fp,
                "role": r.get("role"),
                "ts": r["timestamp"][:10],
                "sha": str(r.get("git_sha", ""))[:7],
            }
        )
    finals = [e for e in evals if e["ver"] in ("v1", "v2", "v3")]
    base = [e for e in evals if e["ver"] is None]
    L.put(
        "log.eval.final_models",
        len(finals),
        "int",
        LOG_FINAL,
        "evaluation lines whose fingerprint equals a final model's model_fingerprint",
    )
    L.put(
        "log.eval.distinct_final_fps",
        len({e["fp"] for e in finals}),
        "int",
        LOG_FINAL,
        "distinct fingerprints among those",
    )
    L.put("log.eval.baselines", len(base), "int", LOG_FINAL, "evaluation lines for B0/B1 baselines")
    erows = []
    for i, e in enumerate(sorted(evals, key=lambda x: (x["ver"] or "z", x["role"] or ""))):
        k = f"log.ev{i}"
        name = e["ver"] or str(e["role"]).replace("baseline_", "")
        L.put(
            f"{k}.name", name, "text", LOG_FINAL, "evaluation line: model version or baseline role"
        )
        L.put(f"{k}.fp", e["fp"], "short", LOG_FINAL, "evaluation line: model_fingerprint")
        L.put(f"{k}.ts", e["ts"], "text", LOG_FINAL, "evaluation line: timestamp date")
        L.put(f"{k}.sha", e["sha"], "text", LOG_FINAL, "evaluation line: git_sha at call time")
        erows.append({"name": f"{k}.name", "fp": f"{k}.fp", "ts": f"{k}.ts", "sha": f"{k}.sha"})
    c["eval_rows"] = erows
    infer = []
    for ct, cnt in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0])):
        if ct == "evaluation":
            continue
        k = f"log.ct.{ct}"
        L.put(k, cnt, "int", LOG_FINAL, f"count of lines with call_type == '{ct}'")
        infer.append({"label": LOG_PURPOSE.get(ct, ct), "key": k})
    c["infer_rows"] = infer
    L.put(
        "log.infer_total",
        sum(v for t, v in by_type.items() if t != "evaluation"),
        "int",
        LOG_FINAL,
        "lines with call_type != 'evaluation'",
    )
    # other logs: LOCO / Track B selection models reading known-class test rows
    orows = []
    tot = 0
    for rel in OTHER_LOGS:
        rows = _read_jsonl(L.root / rel)
        L._sha(rel)
        tot += len(rows)
        cts = Counter(r["call_type"] for r in rows)
        folder = rel.split("/")[1]
        k = f"log.other.{folder}.{rel.split('/')[-1].split('_')[0]}"
        L.put(k, len(rows), "int", rel, "number of log lines")
        orows.append(
            {
                "folder": rel.replace("results/", "").replace("/", " / "),
                "key": k,
                "types": ", ".join(f"{t} ({n})" for t, n in sorted(cts.items())),
            }
        )
    L.put(
        "log.other.n_files",
        len(OTHER_LOGS),
        "int",
        OTHER_LOGS[0],
        "number of Track B log files read",
    )
    L.put("log.other.total", tot, "int", OTHER_LOGS[0], "sum of lines over the five Track B logs")
    c["other_logs"] = orows


def _op90(L: Loader, h: dict[str, Any]) -> None:
    """MSP and shipped-scorer strict rejection and known retention at the 90% operating point.

    headline.json stores the 95% point only; the 90% point is recomputed from the saved per-seed
    score tables with the same rule as `evaluate.threshold_at_retention` (keep ceil(r * n) known
    calibration rows, threshold = the smallest kept score, accept = score >= threshold). The same
    code is run at 95% and must reproduce the stored 3-seed values, otherwise the build fails.
    """
    import numpy as np

    def point(rows: list[dict[str, str]], m: str, r: float) -> tuple[float, float]:
        cal = np.sort([float(x[m]) for x in rows if x["set"] == "cal"])[::-1]
        thr = cal[max(1, math.ceil(r * len(cal) - 1e-9)) - 1]
        ev = [x for x in rows if x["set"] == "eval"]
        known = np.array([float(x[m]) for x in ev if x["is_unknown"] != "True"])
        unk = np.array([float(x[m]) for x in ev if x["is_unknown"] == "True"])
        return float(np.mean(unk < thr)), float(np.mean(known >= thr))

    rels = [f"{TB}scores/headline_s{s}.csv" for s in h["seeds"]]
    tables = [_read_csv_rows(L.root / rel) for rel in rels]
    for m in ("msp", "maha_ft"):
        for r, tag in ((0.95, "95"), (0.90, "90")):
            pts = [point(t, m, r) for t in tables]
            vals = {"rej": [p[0] for p in pts], "ret": [p[1] for p in pts]}
            if tag == "95":  # self-check against the stored 3-seed means
                for met, stored in (
                    ("rej", "strict_rejection_recall"),
                    ("ret", "retention_known"),
                ):
                    got, want = float(np.mean(vals[met])), h["methods"][m][stored]["mean"]
                    if abs(got - want) > 1e-9:
                        raise ValueError(f"op90 recomputation drifted at 95%: {m} {met}")
                continue
            for met, label in (("rej", "strict rejection"), ("ret", "known retention")):
                v = np.array(vals[met])
                L.put(
                    f"tb.op90.{m}.{met}",
                    {"mean": float(v.mean()), "std": float(v.std(ddof=1))},
                    "pm3",
                    rels[0],
                    f"{label} at 90% retention, {m}, mean/std over seeds, from the score tables",
                    fam=f"tb.op90.*.{met}",
                )


def _track_b(L: Loader) -> None:
    c = L.rep.ctx
    h = L.j(HEAD)
    L.put("tb.holdout", " + ".join(h["holdout"]), "text", HEAD, "join(holdout)")
    L.add("tb.n_seeds", HEAD, "n_runs", "int")
    L.put("tb.seeds", ", ".join(str(s) for s in h["seeds"]), "text", HEAD, "join(seeds)")
    L.add("tb.n_known", HEAD, "n_eval_known", "int")
    L.add("tb.n_unknown", HEAD, "n_eval_unknown", "int")
    tpr_match = re.search(r"(\d+)tpr", "fpr_at_95tpr")  # the metric name carries its TPR
    assert tpr_match is not None
    L.put(
        "tb.fpr_tpr",
        int(tpr_match.group(1)) / 100,
        "pct0",
        HEAD,
        "TPR parsed from the metric name fpr_at_95tpr",
    )
    L.add("tb.closed_f1", HEAD, "closed_set_known.macro_f1", "pm3")
    mrows = []
    loco_sel = LOCO
    for m, name in DISPLAY_METHOD.items():
        k = f"tb.m.{m}"
        mb = ("methods", m)
        for met, kind in (
            ("auroc", "pm3"),
            ("strict_rejection_recall", "pm3"),
            ("lenient_rejection_recall", "pm3"),
            ("retention_known", "pm3"),
            ("fpr_at_95tpr", "pm3"),
        ):
            L.add(f"{k}.{met}", HEAD, mb + (met,), kind, fam=f"tb.m.*.{met}")
        L.add(
            f"{k}.loco_auroc",
            loco_sel,
            ("selection", "mean_auroc_by_method", m),
            "f3",
            fam="tb.m.*.loco_auroc",
        )
        L.add(
            f"{k}.loco_rej",
            loco_sel,
            ("mean_across_classes", m, "strict_rejection_recall", "mean"),
            "f3",
            fam="tb.m.*.loco_rej",
        )
        mrows.append(
            {
                "name": name,
                "shipped": m == "maha_ft",
                "auroc": f"{k}.auroc",
                "rej": f"{k}.strict_rejection_recall",
                "len": f"{k}.lenient_rejection_recall",
                "ret": f"{k}.retention_known",
                "fpr": f"{k}.fpr_at_95tpr",
                "loco_auroc": f"{k}.loco_auroc",
                "loco_rej": f"{k}.loco_rej",
            }
        )
    c["tb_methods"] = mrows
    _op90(L, h)
    L.add("tb.loco_n", LOCO, "n_runs", "int")
    L.add("tb.loco_rule", LOCO, "selection.rule", "text")
    L.add("tb.loco_std", LOCO, ("mean_across_classes", "maha_ft", "auroc", "std"), "f3")
    L.add(
        "tb.loco_rej_std",
        LOCO,
        ("mean_across_classes", "maha_ft", "strict_rejection_recall", "std"),
        "f3",
    )
    # strict vs lenient, computed from the shipped method's means
    s = h["methods"]["maha_ft"]["strict_rejection_recall"]["mean"]
    ln = h["methods"]["maha_ft"]["lenient_rejection_recall"]["mean"]
    L.put(
        "tb.lenient_equals_strict",
        abs(s - ln) < 1e-12,
        "bool",
        HEAD,
        "strict mean == lenient mean for maha_ft",
    )
    L.put("tb.not_rej", 1 - s, "pct", HEAD, "1 - mean strict rejection recall (maha_ft, 3 seeds)")
    # where un-rejected unknowns land
    nr = h["methods"]["maha_ft"]["unknown_not_rejected_by_pred_label_summed_over_seeds"]
    total = sum(nr.values())
    L.put("tb.nr_total", total, "int", HEAD, "sum of unknown_not_rejected_by_pred_label (3 seeds)")
    top = sorted(nr.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    c["tb_land"] = []
    for i, (lab, cnt) in enumerate(top):
        L.put(f"tb.land{i}.label", lab, "text", HEAD, f"rank {i} label of unknown_not_rejected")
        L.put(f"tb.land{i}.n", cnt, "int", HEAD, f"rank {i} count of unknown_not_rejected")
        L.put(f"tb.land{i}.share", cnt / total, "pct", HEAD, f"rank {i} count / total")
        c["tb_land"].append(
            {"label": f"tb.land{i}.label", "n": f"tb.land{i}.n", "share": f"tb.land{i}.share"}
        )
    other = nr.get("other", 0) + nr.get("chitchat", 0)
    L.put("tb.nr_safe", other, "int", HEAD, "unknown_not_rejected landing in other + chitchat")
    # paired CIs (v1) from the multi-axis selection confirmation file
    for part in ("headline", "confirm"):
        for met in (
            "auroc",
            "strict_recall_95",
            "retention_95",
            "strict_recall_90",
            "retention_90",
            "fpr_at_95tpr",
        ):
            L.add(f"tb.{part}.{met}", CONF4E, ("v1", part, "ci95", met), "ci", fam=f"tb.{part}.*")
    L.add("tb.boot.n", CONF4E, "bootstrap.n_resamples", "int")
    L.add("tb.boot.seed", CONF4E, "bootstrap.seed", "int")
    L.add("tb.dev_auroc", TBI + "select.json", "current_dev_mean", "f3")
    L.add("tb.dev_rej", P4C + "factor_decisions.json", "dev_current.maha_ft.rej95_mean", "f3")
    L.put(
        "tb.confirm_n",
        len(L.val(TBI + "confirm.json", "confirm_classes")),
        "int",
        TBI + "confirm.json",
        "len(confirm_classes)",
    )
    L.put(
        "tb.dev_n",
        len(L.val(TBI + "select.json", "dev_classes")),
        "int",
        TBI + "select.json",
        "len(dev_classes)",
    )
    # label-smoothing effect on the headline holdout
    ls = TB + "ls_ablation.json"
    L.add("tb.ls.auroc", ls, "methods.maha_ft.auroc.delta", "pm3", fam="tb.ls")
    L.add("tb.ls.rej", ls, "methods.maha_ft.strict_rejection_recall.delta", "pm3", fam="tb.ls")
    # business framing (ESTIMATE): rendered from results/trackb_improve/business.json
    bz = TBI + "business.json"
    rows = []
    for i in range(len(L.val(bz, "rows"))):
        k = f"biz.r{i}"
        for met, kd in (
            ("calibration_retention_target", "pct0"),
            ("prevalence", "pct0"),
            ("retention_measured", "pct"),
            ("known_wrongly_abstained", "per1000"),
            ("unknowns_caught_strict", "per1000"),
            ("unknowns_misrouted_strict", "per1000"),
        ):
            L.add(f"{k}.{met}", bz, f"rows[{i}].{met}", kd, fam=f"biz.*.{met}")
        rows.append(
            {
                m: f"{k}.{m}"
                for m in (
                    "calibration_retention_target",
                    "prevalence",
                    "retention_measured",
                    "known_wrongly_abstained",
                    "unknowns_caught_strict",
                    "unknowns_misrouted_strict",
                )
            }
        )
    c["biz_rows"] = rows
    L.add("biz.per", bz, "per_messages", "int")
    L.add("biz.formula_wrong", bz, "formulas.known_wrongly_abstained", "text")
    L.add("biz.formula_caught", bz, "formulas.unknowns_caught", "text")
    L.add("biz.formula_missed", bz, "formulas.unknowns_misrouted", "text")
    L.add("biz.rej95", bz, "measured_confirm_mean.op95.strict_rejection_recall", "f3")
    L.add("biz.rej90", bz, "measured_confirm_mean.op90.strict_rejection_recall", "f3")


def _errors(L: Loader) -> None:
    c = L.rep.ctx
    oa = "oof_all_fold_runs.hypothesis_errors_concentrate_in_shipment_information"
    o = "oof.hypothesis_errors_concentrate_in_shipment_information"
    t = "test.hypothesis_errors_concentrate_in_shipment_information"
    for tag, base in (("oof45", oa), ("oof", o), ("test", t)):
        L.add(f"err.{tag}.rows", ERR, f"{base}.n_rows", "int")
        L.add(f"err.{tag}.errors", ERR, f"{base}.n_errors", "int")
        L.add(f"err.{tag}.fam_errors", ERR, f"{base}.family_errors", "int")
        L.add(f"err.{tag}.fam_share_err", ERR, f"{base}.family_share_of_errors", "pct")
        L.add(f"err.{tag}.fam_share_rows", ERR, f"{base}.family_share_of_rows", "pct")
    L.add("err.oof45.within", ERR, f"{oa}.family_errors_within_family", "int")
    L.add("err.oof45.within_share", ERR, f"{oa}.share_family_errors_within_family", "pct")
    L.add("err.oof45.into", ERR, f"{oa}.errors_from_outside_into_family", "int")
    L.add_pair(
        "err.oof45.diff_ci",
        ERR,
        f"{oa}.family_share_of_errors",
        f"{oa}.cluster_bootstrap.family_share_of_errors_ci95",
        "ci",
    )
    L.add("err.oof45.boot_n", ERR, f"{oa}.cluster_bootstrap.n_resamples", "int")
    L.add("err.oof45.per_id", ERR, f"{oa}.n_predictions_per_id", "int")
    L.add("err.oof45.n_runs", ERR, "oof_all_fold_runs.n_fold_runs", "int")
    L.add(
        "err.oof45.rows_ci_lo",
        ERR,
        f"{oa}.cluster_bootstrap.share_errors_minus_share_rows_ci95[0]",
        "d3",
    )
    L.add(
        "err.oof45.rows_ci_hi",
        ERR,
        f"{oa}.cluster_bootstrap.share_errors_minus_share_rows_ci95[1]",
        "d3",
    )
    # top confusions (45 fold-runs, counts) and the test confusions
    rows = []
    for i in range(len(L.val(ERR, f"{oa}.top_confusions_by_count"))):
        k = f"err.top{i}"
        L.add(
            f"{k}.gold", ERR, f"{oa}.top_confusions_by_count[{i}].gold", "text", fam="err.top.gold"
        )
        L.add(
            f"{k}.pred", ERR, f"{oa}.top_confusions_by_count[{i}].pred", "text", fam="err.top.pred"
        )
        L.add(f"{k}.n", ERR, f"{oa}.top_confusions_by_count[{i}].count", "int", fam="err.top.n")
        rows.append({"gold": f"{k}.gold", "pred": f"{k}.pred", "n": f"{k}.n"})
    c["top_conf"] = rows
    trows = []
    for i in range(len(L.val(ERR, "test.top_confusions"))):
        k = f"err.ttop{i}"
        L.add(f"{k}.gold", ERR, f"test.top_confusions[{i}].gold", "text", fam="err.ttop.gold")
        L.add(f"{k}.pred", ERR, f"test.top_confusions[{i}].pred", "text", fam="err.ttop.pred")
        L.add(f"{k}.n", ERR, f"test.top_confusions[{i}].count", "int", fam="err.ttop.n")
        trows.append({"gold": f"{k}.gold", "pred": f"{k}.pred", "n": f"{k}.n"})
    c["test_conf"] = trows
    # realtime <-> disruptions over the 45 OOF fold-runs, from the OOF CSV
    classes = [x["label"] for x in L.val(TA, "test.classes")]
    rt, di = (
        classes.index("shipment_information.realtime_query"),
        classes.index("shipment_information.disruptions"),
    )
    oof_csv = "results/oof/e5_lr3e-05.csv"
    L._sha(oof_csv)
    rt_di = di_rt = rt_n = di_n = tot_err = n = 0
    with (L.root / oof_csv).open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            n += 1
            g, p = int(r["gold"]), int(r["pred"])
            tot_err += g != p
            rt_n += g == rt
            di_n += g == di
            rt_di += g == rt and p == di
            di_rt += g == di and p == rt
    d = "gold/pred columns of results/oof/e5_lr3e-05.csv (45 fold-runs)"
    L.put("err.rtdi.rt_to_di", rt_di, "int", oof_csv, f"count gold=realtime, pred=disruptions; {d}")
    L.put("err.rtdi.di_to_rt", di_rt, "int", oof_csv, f"count gold=disruptions, pred=realtime; {d}")
    L.put(
        "err.rtdi.share",
        (rt_di + di_rt) / tot_err,
        "pct",
        oof_csv,
        "(realtime->disruptions + disruptions->realtime) / all errors",
    )
    L.put("err.rtdi.rt_n", rt_n, "int", oof_csv, "rows with gold=realtime")
    L.put("err.rtdi.di_n", di_n, "int", oof_csv, "rows with gold=disruptions")
    L.put(
        "err.rtdi.rate_rt", rt_di / rt_n, "pct", oof_csv, "realtime->disruptions / rows(realtime)"
    )
    L.put(
        "err.rtdi.rate_di",
        di_rt / di_n,
        "pct",
        oof_csv,
        "disruptions->realtime / rows(disruptions)",
    )
    # same pair on the single test split (confusion matrix)
    cm = L.val(TA, "test.confusion_counts")
    L.put(
        "err.test_rtdi",
        cm[rt][di] + cm[di][rt],
        "int",
        TA,
        "confusion_counts[realtime][disruptions] + confusion_counts[disruptions][realtime]",
    )
    # ID-prefix counterfactuals (OOF rows, 5 swaps)
    idx = {
        (r["source"], r["swap"]): i
        for i, r in enumerate(L.val("results/robustness/idswap.json", "results"))
    }
    srows = []
    f = "results/robustness/idswap.json"
    for s in SWAPS:
        i = idx[("oof", s)]
        k = f"swap.{s}"
        L.add(f"{k}.n", f, f"results[{i}].n", "int", fam="swap.*.n")
        L.add(f"{k}.flips", f, f"results[{i}].n_flips", "int", fam="swap.*.flips")
        L.add(f"{k}.rate", f, f"results[{i}].flip_rate", "pct", fam="swap.*.rate")
        L.add(f"{k}.acc_clean", f, f"results[{i}].accuracy_clean", "f3", fam="swap.*.acc_clean")
        L.add(f"{k}.acc_swap", f, f"results[{i}].accuracy_swapped", "f3", fam="swap.*.acc_swap")
        srows.append(
            {
                "swap": s,
                "n": f"{k}.n",
                "flips": f"{k}.flips",
                "rate": f"{k}.rate",
                "ac": f"{k}.acc_clean",
                "as": f"{k}.acc_swap",
            }
        )
    c["swaps"] = srows


def _robustness(L: Loader) -> None:
    c = L.rep.ctx
    f = "results/robustness/translate.json"
    res = L.val(f, "results")
    L.add("tr.translator", f, "translator", "text")
    L.add("tr.label", f, "data_label", "text")
    L.add("tr.beams", f, "decoding.num_beams", "int")
    rows = []
    for code, name in LANGS:
        r: dict[str, Any] = {"name": name}
        for src in ("oof", "test"):
            i = next(j for j, x in enumerate(res) if x["source"] == src and x["lang"] == code)
            k = f"tr.{src}.{code}"
            for met, kd in (
                ("n", "int"),
                ("agreement_with_english_pred", "f3"),
                ("accuracy", "f3"),
                ("english_accuracy_same_rows", "f3"),
            ):
                L.add(f"{k}.{met}", f, f"results[{i}].{met}", kd, fam=f"tr.{src}.*.{met}")
            r[src] = {
                m: f"{k}.{m}"
                for m in (
                    "n",
                    "agreement_with_english_pred",
                    "accuracy",
                    "english_accuracy_same_rows",
                )
            }
        rows.append(r)
    c["translate"] = rows
    nf = "results/robustness/noise.json"
    nres = L.val(nf, "results")
    nrows = []
    for pid, name in PERTURBATIONS:
        i = next(j for j, x in enumerate(nres) if x["source"] == "oof" and x["perturbation"] == pid)
        k = f"nz.{pid}"
        L.add(f"{k}.acc", nf, f"results[{i}].accuracy_perturbed", "f3", fam="nz.*.acc")
        L.add(f"{k}.drop", nf, f"results[{i}].accuracy_drop", "ci", fam="nz.*.drop")
        L.add(f"{k}.flip", nf, f"results[{i}].flip_rate", "pct", fam="nz.*.flip")
        L.add(f"{k}.n", nf, f"results[{i}].n", "int", fam="nz.*.n")
        nrows.append({"name": name, "acc": f"{k}.acc", "drop": f"{k}.drop", "flip": f"{k}.flip"})
    c["noise"] = nrows
    L.add("nz.clean_acc", nf, "results[0].accuracy_clean", "f3")  # test clean
    j0 = next(j for j, x in enumerate(nres) if x["source"] == "oof")
    L.add("nz.oof_clean_acc", nf, f"results[{j0}].accuracy_clean", "f3")
    L.add("nz.oof_n", nf, f"results[{j0}].n", "int")
    L.add("nz.boot_n", nf, "bootstrap.n_resamples", "int")
    L.add("nz.seed", nf, "seed", "int")


def _llm(L: Loader) -> None:
    c = L.rep.ctx
    rows = []
    for cid, name in LLM_CONFIGS:
        k = f"llm.{cid.replace('|', '_')}"
        ta = ("track_a", "llm", cid)
        tb = ("track_b", "llm", cid)
        L.add(f"{k}.f1", LLM, ta + ("macro_f1",), "ci", fam="llm.*.f1")
        L.add(f"{k}.acc", LLM, ta + ("accuracy", "point"), "f3", fam="llm.*.acc")
        L.add(f"{k}.unk", LLM, ta + ("unknown_rate",), "pct", fam="llm.*.unk")
        L.add(f"{k}.pf", LLM, ta + ("parse_failure_rate",), "pct", fam="llm.*.pf")
        L.add(
            f"{k}.delta",
            LLM,
            ta + ("delta_macro_f1_llm_minus_final",),
            "dci",
            point_key="delta",
            fam="llm.*.delta",
        )
        L.add(f"{k}.rej", LLM, tb + ("strict_rejection_recall",), "f3", fam="llm.*.rej")
        L.add(f"{k}.ret", LLM, tb + ("retention_known",), "f3", fam="llm.*.ret")
        lat = ("latency", "llm", "configs", cid)
        L.add(f"{k}.p50", LLM, lat + ("p50_s",), "f3", fam="llm.*.p50")
        L.add(f"{k}.p95", LLM, lat + ("p95_s",), "f3", fam="llm.*.p95")
        cost = ("cost", "rows", f"llm|{cid}")
        L.add(f"{k}.usd", LLM, cost + ("usd_per_1k_messages",), "usd", fam="llm.*.usd")
        L.add(f"{k}.basis", LLM, cost + ("usd_per_hr_basis",), "text", fam="llm.*.basis")
        rows.append(
            {
                "name": name,
                **{
                    m: f"{k}.{m}"
                    for m in (
                        "f1",
                        "acc",
                        "unk",
                        "pf",
                        "delta",
                        "rej",
                        "ret",
                        "p50",
                        "p95",
                        "usd",
                        "basis",
                    )
                },
            }
        )
    c["llm_rows"] = rows
    c["llm_models"] = "qwen3:8b and llama3.1:8b"  # model tags from LLM_CONFIGS (names, not numbers)
    ftb = []
    # GPU batch-1 only: the local CPU timings were taken on a shared machine and are not reported;
    # CPU latency comes from the Colab run (results/serving/latency_colab.json, see _serving).
    for tag, name, cost in (("gpu1", "fine-tuned v1, GPU batch 1", "ft|gpu_batch1"),):
        k = f"ft.{tag}"
        L.add(f"{k}.p50", LLM, ("cost", "rows", cost, "latency_p50_s"), "f3", fam="ft.*.p50")
        L.add(f"{k}.usd", LLM, ("cost", "rows", cost, "usd_per_1k_messages"), "usd", fam="ft.*.usd")
        L.add(
            f"{k}.basis", LLM, ("cost", "rows", cost, "usd_per_hr_basis"), "text", fam="ft.*.basis"
        )
        ftb.append({"name": name, "p50": f"{k}.p50", "usd": f"{k}.usd", "basis": f"{k}.basis"})
    L.add("ft.gpu1.p95", LLM, "latency.finetuned.gpu_batch1.p95_s", "f3")
    # device of the committed results and of the LLM / encoder GPU latency rows
    L.add("llm.gpu_name", "results/llm_baseline/ft_latency.json", "gpu_name", "text")
    ftb[0]["p95"] = "ft.gpu1.p95"
    c["ft_rows"] = ftb
    L.add("llm.ft_f1", LLM, "track_a.final_model.macro_f1", "ci")
    L.add("llm.ft_acc", LLM, "track_a.final_model.accuracy.point", "f3")
    L.add("llm.ft_rej", LLM, "track_b.shipped_maha_ft.strict_rejection_recall", "pm3")
    L.add("llm.ft_ret", LLM, "track_b.shipped_maha_ft.retention_known", "pm3")
    L.add("llm.gpu_hr", LLM, "cost.assumptions.gpu_usd_per_hr", "usd2")
    L.put(
        "llm.n_shots",
        len(L.val(LLM, "track_a.fewshot_ids.five")),
        "int",
        LLM,
        "len(track_a.fewshot_ids.five)",
    )
    L.add("llm.n_known", LLM, "track_b.n_known", "int")
    L.add("llm.n_unknown", LLM, "track_b.n_unknown", "int")
    L.add("llm.boot_n", LLM, ("track_a", "llm", "qwen3|zero", "bootstrap", "n_resamples"), "int")
    _llm_vs_encoder(L)
    # label audit (LLM consensus)
    au = "results/llm_audit/summary.json"
    L.add("aud2.kappa", au, "fleiss_kappa.value", "f3")
    L.add("aud2.gate", au, "calibration_gate", "f2")
    L.add("aud2.reliable", au, "audit_reliable", "bool")
    L.add("aud2.n", au, "fleiss_kappa.n_items_all_valid", "int")
    L.put("aud2.n_judges", len(L.val(au, "judges")), "int", au, "len(judges)")
    for j in ("gemma3", "llama3", "qwen3"):
        L.add(f"aud2.{j}", au, ("per_judge", j, "calibration_accuracy"), "f2")
    cv = ("consensus_vs_model", "error")
    L.add("aud2.err_n", au, cv + ("n",), "int")
    L.add("aud2.err_model", au, cv + ("agrees_with_model",), "int")
    L.add("aud2.err_gold", au, cv + ("agrees_with_gold",), "int")
    L.add("aud2.err_other", au, cv + ("other_label",), "int")
    L.add("aud2.err_none", au, cv + ("no_consensus",), "int")


def _llm_vs_encoder(L: Loader) -> None:
    """Computed comparison keys (section 2): the four LLM configs against the shipped encoder."""
    ids = [cid.replace("|", "_") for cid, _ in LLM_CONFIGS]
    m = L.rep.metrics
    f1p = {i: m[f"llm.{i}.f1"].value["point"] for i in ids}
    best = max(ids, key=lambda i: (f1p[i], i))
    names = {cid.replace("|", "_"): nm for cid, nm in LLM_CONFIGS}
    L.rep.ctx["llm_best"] = {
        "name": names[best],
        **{x: f"llm.{best}.{x}" for x in ("f1", "delta", "rej", "ret", "unk", "p50", "usd")},
    }
    d = "over the four LLM configs in results/llm_baseline/summary.json"
    p50 = {i: m[f"llm.{i}.p50"].value for i in ids}
    usd = {i: m[f"llm.{i}.usd"].value for i in ids}
    rej = {i: m[f"llm.{i}.rej"].value for i in ids}
    L.put("llm.f1_min", min(f1p.values()), "f3", LLM, f"min test macro-F1 {d}", fam="llm.cmp")
    L.put("llm.f1_max", max(f1p.values()), "f3", LLM, f"max test macro-F1 {d}", fam="llm.cmp")
    L.put("llm.p50_min", min(p50.values()), "ms_from_s", LLM, f"min p50 latency {d}", fam="llm.cmp")
    L.put("llm.p50_max", max(p50.values()), "ms_from_s", LLM, f"max p50 latency {d}", fam="llm.cmp")
    L.put(
        "llm.usd_min", min(usd.values()), "usd", LLM, f"min USD per 1k messages {d}", fam="llm.cmp"
    )
    L.put(
        "llm.usd_max", max(usd.values()), "usd", LLM, f"max USD per 1k messages {d}", fam="llm.cmp"
    )
    L.put("llm.rej_max", max(rej.values()), "f3", LLM, f"max strict rejection {d}", fam="llm.cmp")
    ft_gpu, ft_gpu_usd = m["ft.gpu1.p50"].value, m["ft.gpu1.usd"].value
    L.put("ft.gpu1.p50_ms", ft_gpu, "ms_from_s", LLM, "ft gpu batch-1 p50 latency", fam="llm.cmp")
    L.put(
        "ratio.lat_gpu",
        min(p50.values()) / ft_gpu,
        "f1",
        LLM,
        "min LLM p50 / fine-tuned GPU batch-1 p50 (same local GPU)",
        fam="llm.cmp",
    )
    L.put(
        "ratio.usd",
        min(usd.values()) / ft_gpu_usd,
        "f1",
        LLM,
        "min LLM USD per 1k / fine-tuned GPU batch-1 USD per 1k (same T4-class $/hr basis)",
        fam="llm.cmp",
    )


def _selection_history(L: Loader) -> None:
    c = L.rep.ctx
    v2 = P4C + "final_v1_vs_v2.json"
    v3 = P4E + "final_v1_vs_v3.json"
    for ver, rel in (("v1", v2), ("v2", v2), ("v3", v3)):
        L.add(f"hist.{ver}.f1", rel, f"{ver}.macro_f1", "ci")
        L.add(f"hist.{ver}.T", rel, f"{ver}.temperature", "f3")
        L.add(f"hist.{ver}.ece", rel, f"{ver}.ece_test_after", "f3")
        L.add(f"hist.{ver}.fp", rel, f"{ver}.model_fingerprint", "short")
    cc = P4C + "confirm_comb.json"
    for met, tag in (("strict_recall_95", "rej95"), ("auroc", "auroc")):
        L.add(
            f"hist.v2.head.{tag}",
            cc,
            f"paired_delta_comb_minus_current.headline.{met}",
            "dci",
            point_key="mean_delta",
        )
    fd = P4C + "factor_decisions.json"
    f0 = ("factors", "a3", "all_gains", "a1_flip_reduction")
    L.add("hist.v2.flip_cur", fd, f0 + ("flip_rate_current",), "f3")
    L.add("hist.v2.flip_cand", fd, f0 + ("flip_rate_candidate",), "f3")
    L.add("hist.v2.zh", fd, ("factors", "a3", "gains", "a3_zh_agreement_gain"), "dci")
    L.add("hist.v2.agree", fd, ("factors", "a3", "gains", "a3_mean_agreement_gain"), "dci")
    for fct in ("a1", "a2", "a3"):
        L.add(f"hist.guard.{fct}", fd, ("guards", fct, "macro_f1_at_argmax"), "f4")
        L.add(f"hist.guard.{fct}.ok", fd, ("guards", fct, "passed"), "bool")
        L.add(f"hist.adopt.{fct}", fd, ("factors", fct, "adopted"), "bool")
    L.add("hist.guard.thr", "results/trackb_improve/guard/c2_l0.1.json", "threshold", "f4")
    L.add("hist.a1.flip", fd, ("factors", "a1", "gains", "a1_flip_reduction"), "dci")
    L.add("hist.a2.auroc", fd, ("factors", "a2", "gains", "a2_auroc_gain"), "dci")
    L.add("hist.a2.rej", fd, ("factors", "a2", "gains", "a2_rej95_gain"), "dci")
    L.add("hist.sens.v1_argmax", AX, "v1_epoch.own_argmax", "int")
    L.add("hist.sens.v1_e", AX, "v1_epoch.e_star", "int")
    L.add("hist.sens.v1_f1_e", AX, "v1_epoch.f1_at_e_star", "f4")
    L.add("hist.sens.v1_f1_arg", AX, "v1_epoch.f1_at_own_argmax", "f4")
    L.add("hist.v3.epoch", AX, ("candidates", "a1a3", "e_star"), "int")
    L.add(
        "hist.sens.same",
        P4E + "selection.json",
        "sensitivity_v1_at_own_argmax.same_outcome",
        "bool",
    )
    # multi-axis selection axes table
    arows = []
    for cid, name in (
        ("a3", "A3 (MT + noise copies)"),
        ("a1", "A1 (ID-prefix randomisation)"),
        ("a1a3", "A1 + A3 (v3)"),
    ):
        k = f"ax.{cid}"
        base = ("candidates", cid)
        L.add(f"{k}.estar", AX, base + ("e_star",), "int", fam="ax.*.estar")
        row: dict[str, Any] = {"name": name, "estar": f"{k}.estar"}
        for ax, path in (
            ("a", ("axes", "a")),
            ("bA", ("axes", "b", "auroc")),
            ("bR", ("axes", "b", "rej95")),
            ("c", ("axes", "c")),
            ("d", ("axes", "d")),
            ("e", ("axes", "e")),
        ):
            d = L.val(AX, base + path)
            v = {"point": d["improvement"], "lo": d["lo"], "hi": d["hi"]}
            L._store(
                Metric(
                    f"{k}.{ax}",
                    v,
                    "dci",
                    AX,
                    _path_str(_segments(base + path)) + ".{improvement,lo,hi}",
                    f"ax.*.{ax}",
                )
            )
            row[ax] = f"{k}.{ax}"
        L.add(f"{k}.elig", AX, base + ("eligible",), "bool", fam="ax.*.elig")
        L.add(f"{k}.nimp", AX, base + ("n_improved",), "int", fam="ax.*.nimp")
        row["elig"], row["nimp"] = f"{k}.elig", f"{k}.nimp"
        arows.append(row)
    c["axes_rows"] = arows
    for ax, key in (
        ("f1", "f1"),
        ("auroc", "auroc"),
        ("rej95", "rej95"),
        ("flip", "flip_rate"),
        ("agree", "agreement"),
        ("ece", "ece"),
    ):
        L.add(f"ax.v1.{ax}", AX, ("v1", key), "f3")
    # tradeoff table (v1 vs v3), straight from results/tradeoff_v1_v3.json
    t = TRADE
    trows = []

    def ci_row(label: str, base: tuple[Any, ...], lower_better: bool = False) -> None:
        k = f"trd.{len(trows)}"
        L.add(f"{k}.v1", t, base + ("v1",), "ci", fam="trd.*.v1")
        L.add(f"{k}.v3", t, base + ("v3",), "ci", fam="trd.*.v3")
        L.add_pair(
            f"{k}.d", t, base + ("delta_v3_minus_v1",), base + ("ci95",), "dci", fam="trd.*.d"
        )
        trows.append({"name": label, "v1": f"{k}.v1", "v3": f"{k}.v3", "d": f"{k}.d"})

    def sc_row(label: str, base: tuple[Any, ...]) -> None:
        k = f"trd.{len(trows)}"
        L.add(f"{k}.v1", t, base + ("v1",), "f3", fam="trd.*.v1s")
        L.add(f"{k}.v3", t, base + ("v3",), "f3", fam="trd.*.v3s")
        L.add_pair(
            f"{k}.d", t, base + ("delta_v3_minus_v1",), base + ("ci95",), "dci", fam="trd.*.d"
        )
        trows.append({"name": label, "v1": f"{k}.v1", "v3": f"{k}.v3", "d": f"{k}.d"})

    sc_row("CV macro-F1, v1 at shipped epoch", ("cv", "v3_vs_v1_at_e_star"))
    sc_row("CV macro-F1, v1 at its own best epoch", ("cv", "v3_vs_v1_at_own_argmax_sensitivity"))
    sc_row("Track B DEV AUROC", ("trackb_dev", "auroc"))
    sc_row("Track B DEV strict rejection @95", ("trackb_dev", "rej95"))
    for part, nm in (("headline", "HEADLINE"), ("confirm", "CONFIRM")):
        ci_row(f"Track B {nm} AUROC", ("trackb_holdout", part, "auroc"))
        ci_row(f"Track B {nm} strict rejection @95", ("trackb_holdout", part, "strict_recall_95"))
        ci_row(f"Track B {nm} strict rejection @90", ("trackb_holdout", part, "strict_recall_90"))
    sc_row(
        "neutral-swap flip rate (lower is better)",
        ("robustness_cv", "neutral_swap_flip_rate (axis c, lower is better)"),
    )
    sc_row(
        "translation agreement",
        ("robustness_cv", "translation_agreement (axis d, higher is better)"),
    )
    sc_row("OOF ECE (lower is better)", ("robustness_cv", "oof_ece (axis e, lower is better)"))
    c["tradeoff_rows"] = trows
    L.add_pair(
        "trd.head.rej95",
        t,
        ("trackb_holdout", "headline", "strict_recall_95", "delta_v3_minus_v1"),
        ("trackb_holdout", "headline", "strict_recall_95", "ci95"),
        "dci",
    )
    L.add("trd.v1.T", t, "track_a_test.v1.temperature.value", "f3")
    L.add("trd.v3.T", t, "track_a_test.v3.temperature.value", "f3")
    L.add("trd.v1.cov", t, "serving.v1.test_coverage_at_threshold.value", "f3")
    L.add("trd.v3.cov", t, "serving.v3.test_coverage_at_threshold.value", "f3")
    L.add("trd.v1.aurc", t, "track_a_test.v1.selective_prediction.aurc_temp_scaled_msp.value", "f4")
    L.add("trd.v3.aurc", t, "track_a_test.v3.selective_prediction.aurc_temp_scaled_msp.value", "f4")
    L.add("trd.guidance1", t, "deployment_guidance[0]", "text")
    L.add("trd.guidance2", t, "deployment_guidance[1]", "text")


# ESTIMATE assumption, not a measurement: price of one (shared) vCPU per hour for the CPU cost line.
CPU_USD_PER_VCPU_HR = 0.05


def _serving_cpu_cost(L: Loader, lat: str, c: dict[str, Any]) -> None:
    """CPU cost per 1k messages (ESTIMATE) and the thread-scaling facts the deploy text uses."""
    for be in ("torch", "onnx_fp32"):
        L.add(f"srv.{be}.t1.mean", lat, ("latency", be, "threads_1", "mean_ms"), "ms")
    L.put("srv.cpu_usd_hr", CPU_USD_PER_VCPU_HR, "usd2", lat, "assumed price of one vCPU per hour")
    for be in ("torch", "onnx_fp32"):
        for stat in ("p50", "mean"):
            ms = L.val(lat, ("latency", be, "threads_1", f"{stat}_ms"))
            # sequential batch-1: 1000 calls x ms each / 1000 ms per s = `ms` seconds on one vCPU
            usd = ms / 3600 * CPU_USD_PER_VCPU_HR
            L.put(
                f"srv.{be}.usd1k_{stat}",
                usd,
                "usd5",
                lat,
                f"{stat}_ms x 1000 calls / 1000 / 3600 x vCPU price",
            )
    # thread scaling, read from the data (the template says "slower" only if it is)
    multi = c["srv_multi"]
    t1 = {be: L.val(lat, ("latency", be, "threads_1")) for be in ("torch", "onnx_fp32")}
    tn = (
        {
            be: L.val(lat, ("latency", be, f"threads_{L.val(lat, 'settings.threads')[-1]}"))
            for be in ("torch", "onnx_fp32")
        }
        if multi
        else {}
    )
    slower = {
        be: tn[be]["p50_ms"] > t1[be]["p50_ms"] and tn[be]["p95_ms"] > t1[be]["p95_ms"] for be in tn
    }
    c["srv_onnx_t1_best"] = bool(multi and slower["onnx_fp32"])
    c["srv_torch_t1_best"] = bool(multi and slower["torch"])
    c["srv_onnx_beats_torch_t1"] = t1["onnx_fp32"]["p50_ms"] < t1["torch"]["p50_ms"]


def _serving(L: Loader) -> None:
    c = L.rep.ctx
    # CPU batch-1 latency: only from the Colab canonical run. The local latency_v1.json (taken on a
    # shared machine) is kept in results/ as history and is deliberately not read here.
    lat = SRV + "latency_colab.json"
    c["srv_colab"] = L.exists(lat)
    c["srv_rows"] = []
    if c["srv_colab"]:
        L.add("srv.split", lat, "split", "text")
        L.add("srv.n_rows", lat, "n_rows", "int")
        L.add("srv.repeats", lat, "settings.repeats", "int")
        L.add("srv.warmup", lat, "settings.warmup", "int")
        L.add("srv.cpu_cores", lat, "environment.os_cpu_count", "int")
        L.add("srv.cpu_model", lat, "environment.cpu_model", "text")
        L.add("srv.runtime", lat, "environment.runtime_type", "text")
        L.add("srv.torch_ver", lat, "environment.torch", "text")
        L.add("srv.ort_ver", lat, "environment.onnxruntime", "text")
        threads = list(L.val(lat, "settings.threads"))
        if not threads or threads[0] != 1:
            raise ValueError(f"{lat}: settings.threads must start with 1, got {threads}")
        n_thr = threads[-1]  # == 1 on a single-vCPU runtime: one column pair only
        c["srv_multi"] = n_thr != 1
        L.put("srv.threads_n", n_thr, "int", lat, "last entry of settings.threads (os.cpu_count())")
        rows = []
        for be, bname in (("torch", "PyTorch fp32"), ("onnx_fp32", "ONNX fp32 (shipped default)")):
            r: dict[str, Any] = {"name": bname}
            for slot, th in (("t1", 1), ("tn", n_thr)):
                if slot == "tn" and not c["srv_multi"]:
                    continue
                k = f"srv.{be}.{slot}"
                base = ("latency", be, f"threads_{th}")
                L.add(f"{k}.p50", lat, base + ("p50_ms",), "ms", fam=f"srv.*.{slot}.p50")
                L.add(f"{k}.p95", lat, base + ("p95_ms",), "ms", fam=f"srv.*.{slot}.p95")
                r[slot] = {"p50": f"{k}.p50", "p95": f"{k}.p95"}
            rows.append(r)
        c["srv_rows"] = rows
        _serving_cpu_cost(L, lat, c)
    # sizes + quantization decision
    q = SRV + "quantization_v1_test.json"
    qv = SRV + "quantization_v1_val.json"
    sz = "sizes_bytes"
    L.add("q.size.torch", q, f"{sz}.torch_safetensors", "mb")
    L.add("q.size.onnx32", q, f"{sz}.onnx_fp32", "mb")
    L.add("q.size.int8enc", q, f"{sz}.onnx_int8_encoder", "mb")
    L.add("q.size.int8full", q, f"{sz}.onnx_int8", "mb")
    L.add("q.margin", q, "decision.margin", "d3")
    L.add("q.d_val", q, "decision.macro_f1_delta_vs_torch.val", "d3")
    L.add("q.d_test", q, "decision.macro_f1_delta_vs_torch.test", "d3")
    L.add("q.n_rows", q, "n_rows", "int")
    L.add("q.pass_val", q, "decision.passes.val", "bool")
    L.add("q.pass_test", q, "decision.passes.test", "bool")
    L.add("q.decision", q, "decision.decision", "text")
    L.add("q.fp32.agree", q, "quality.onnx_fp32.pred_agreement_vs_torch", "f3")
    L.add("q.fp32.dprob", q, "quality.onnx_fp32.max_abs_diff_prob_vs_torch", "sci")
    L.add("q.fp32.f1", q, "quality.onnx_fp32.macro_f1", "f3")
    L.add("q.int8.agree", q, "quality.onnx_int8_encoder.pred_agreement_vs_torch", "f3")
    L.add("q.int8.abst_agree", q, "quality.onnx_int8_encoder.abstention_agreement_vs_torch", "f3")
    L.add("q.int8.dprob", q, "quality.onnx_int8_encoder.max_abs_diff_prob_vs_torch", "f3")
    L.add("q.int8.f1", q, "quality.onnx_int8_encoder.macro_f1", "f3")
    L.add("q.torch.f1", q, "quality.torch.macro_f1", "f3")
    L.add("q.int8.abst", q, "quality.onnx_int8_encoder.n_abstained", "int")
    L.add("q.torch.abst", q, "quality.torch.n_abstained", "int")
    L.add("qv.int8.f1", qv, "quality.onnx_int8_encoder.macro_f1", "f3")
    L.add("qv.torch.f1", qv, "quality.torch.macro_f1", "f3")
    L.put("q.n_variants", len(L.val(q, "variants")), "int", q, "len(variants)")
    # docker
    dk = SRV + "docker_parity_v1.json"
    L.add("dk.size", dk, "image_size_bytes", "gb")
    L.add("dk.ready", dk, "container_start_to_ready_s", "s1")
    L.add("dk.n", dk, "summary.n", "int")
    L.add("dk.labels_eq", dk, "summary.all_labels_equal", "bool")
    L.add("dk.abst_eq", dk, "summary.all_abstained_equal", "bool")
    L.add("dk.dconf", dk, "summary.max_abs_diff_confidence", "sci")
    L.add("dk.dood", dk, "summary.max_abs_diff_ood_score", "sci")
    L.add("dk.backend", dk, "backend", "text")
    # hub round trip
    hb = "results/hub/roundtrip_v1.json"
    L.add("hub.n", hb, "n", "int")
    L.add("hub.agree", hb, "label_agreement", "f3")
    L.add("hub.dprob", hb, "max_abs_prob_diff", "sci")
    L.add("hub.tol", hb, "tolerances.max_abs_prob_diff", "sci")
    L.add("hub.pass", hb, "passes", "bool")
    # shipped abstention on test
    L.add("srv.thr", F + "ood_shipped.json", "threshold", "f2")
    L.add("srv.retention", F + "ood_shipped.json", "val_retention_achieved", "f3")
    L.add("srv.cov", F + "ood_shipped.json", "test_coverage_at_threshold", "f3")
    L.add("srv.ret_target", F + "ood_shipped.json", "retention_target", "pct")
    L.add("srv.det", F + "determinism.json", "bitwise_identical", "bool")


def _phase3b(L: Loader) -> None:
    c = L.rep.ctx
    s = TBI + "select.json"
    rows = []
    for cid, name in (
        ("base/maha_ft", "current: Mahalanobis (fine-tuned), shipped"),
        ("base/fuse_maha_energy", "C1: rank fusion Mahalanobis + energy"),
        ("base/maha_ml", "C1: multi-layer Mahalanobis"),
        ("c2_l0.1/maha_ft", "C2: CE + SupCon, lambda low"),
        ("c2_l0.5/maha_ft", "C2: CE + SupCon, lambda high"),
        ("c4/ens_maha", "C4: 5-seed ensemble (evidence only)"),
        ("b1/b1_msp", "B1 fairness variant: frozen e5 + LR, MSP"),
    ):
        k = "p3b." + cid.replace("/", "_")
        L.add(k, s, ("dev_mean_auroc", cid), "f3", fam="p3b.*")
        rows.append({"name": name, "key": k})
    c["p3b_rows"] = rows
    L.add("p3b.margin", s, "margin_std_ddof1_over_sqrt_n", "f4")
    L.add("p3b.ship", s, "ship", "bool")
    L.add("p3b.win_minus", s, "winner_minus_current", "d4")
    cf = TBI + "confirm.json"
    L.add("p3b.c4_confirm", cf, ("report_only", "confirm_mean_auroc", "c4/ens_maha"), "f3")
    L.add("p3b.cur_confirm", cf, ("report_only", "confirm_mean_auroc", "base/maha_ft"), "f3")
    L.add("p3b.b1_confirm", cf, ("report_only", "confirm_mean_auroc", "b1/b1_msp"), "f3")
    for lam in ("0.1", "0.5"):
        g = TBI + f"guard/c2_l{lam}.json"
        L.add(f"p3b.c2.{lam}.f1", g, "macro_f1_at_deployed", "f4")
        L.add(f"p3b.c2.{lam}.ok", g, "passed", "bool")
    L.add("p3b.guard", TBI + "guard/c2_l0.1.json", "threshold", "f4")
    c3 = TBI + "c3_fit_check.json"
    L.add("p3b.c3.mem", c3, "attempts[0].max_memory_allocated_mb", "ms")
    L.add("p3b.c3.vram", c3, "vram_budget_gb", "int")
    L.add("p3b.c3.static", c3, "arithmetic.static_total_mb", "ms")
    L.add("p3b.c3.params", c3, "arithmetic.n_params", "mparams")
    L.put("p3b.c3.n_attempts", len(L.val(c3, "attempts")), "int", c3, "len(attempts)")
    L.add("p3b.c3.aborted", c3, "aborted", "bool")
    # robustness-fix factor table
    fd = P4C + "factor_decisions.json"
    L.add("p4c.a2.dev_cur", fd, "dev_current.maha_ft.auroc_mean", "f3")
    L.add("p4c.a2.dev_a2", fd, "dev_a2.maha_ft.auroc_mean", "f3")


P6A = "results/phase6a/"
P6A_D1 = P6A + "diag/d1_oracle.json"
P6A_D2 = P6A + "diag/d2_id_neutral.json"
P6A_D3 = P6A + "diag/d3_threshold_stability.json"
P6A_D4 = P6A + "diag/d4_learning_curve.json"
P6A_N2 = P6A + "diag/n2_decision.json"
P6A_SEL = P6A + "selection.json"
P6A_SEL_UNG = P6A + "selection_unguarded.json"
P6A_AXES = P6A + "axes.json"
P6A_CONF = P6A + "confirm_report.json"
P6A_REF_EPOCH = P6A + "ref/epoch.json"
# Track B test-row reads made in the open-set improvement round (the classifier test split was not
# evaluated)
P6A_LOGS = (
    P6A + "test_inference_log.jsonl",
    P6A + "diag/test_inference_log.jsonl",
    P6A + "confirm_test_inference_log.jsonl",
)
D1_SETS = [
    ("v1_finetuned", "fine-tuned v1 features"),
    ("frozen_e5_base", "frozen e5-base"),
    ("frozen_e5_large", "frozen e5-large"),
    ("frozen_bge_m3", "frozen bge-m3"),
    ("frozen_labse", "frozen LaBSE"),
]
P6A_CANDS = ("i1a", "i1b", "i6a", "i6b", "i3u", "i3g", "i4a", "i4b")
P6A_CANDS += ("i3i4", "i3i1", "i3i4i1", "i3i4i1i6")


def _p6a_d1(L: Loader) -> dict[str, Any]:
    base = "holdouts.headline.feature_sets."
    rows = []
    for fid, name in D1_SETS:
        k = f"p6a.d1.{fid}"
        p = f"{base}{fid}.all_rows."
        L.add_pair(k, P6A_D1, p + "auroc_mean_of_folds", p + "auroc_mean_of_folds_ci", "ci")
        ku = f"p6a.d1.{fid}.unseen"
        q = f"{base}{fid}.unseen_known_only."
        L.add_pair(ku, P6A_D1, q + "auroc_mean_of_folds", q + "auroc_mean_of_folds_ci", "ci")
        rows.append({"name": name, "key": k, "unseen": ku})
    un = "holdouts.headline.unsupervised_finetuned_mahalanobis.phase3_results.mean_3_seeds"
    maha = L.add("p6a.d1.maha", P6A_D1, un, "f3")  # Y: v1's unsupervised Mahalanobis, 3-seed mean
    # HEADLINE pool: unseen known rows only (known TRAIN rows dropped; the v1 encoder trained on
    # them, so the all-rows number is inflated and stays in the appendix only).
    ft = L.rep.metrics["p6a.d1.v1_finetuned.unseen"].value["point"]
    desc = "oracle fine-tuned (unseen-known pool) - unsupervised"
    L.put("p6a.d1.headroom", ft - maha, "d3", P6A_D1, desc)
    u = base + "v1_finetuned.unseen_known_only."
    L.add("p6a.d1.n_pos", P6A_D1, u + "n_pos", "int")
    L.add("p6a.d1.n_neg", P6A_D1, u + "n_neg", "int")
    L.add("p6a.d1.n_neg_all", P6A_D1, base + "v1_finetuned.all_rows.n_neg", "int")
    return {
        "rows": rows,
        "maha": "p6a.d1.maha",
        "headroom": "p6a.d1.headroom",
        "n_pos": "p6a.d1.n_pos",
        "n_neg": "p6a.d1.n_neg",
        "n_neg_all": "p6a.d1.n_neg_all",
    }


def _p6a_n2(L: Loader) -> dict[str, str]:
    n2k = {
        "base": "values.e5_base_mean_of_folds",
        "large": "values.e5_large_mean_of_folds",
        "diff": "values.diff_mean_of_folds",
        "margin": "margin",
        "run": "run_n2",
    }
    for tag, path in n2k.items():
        kind = {"margin": "f2", "run": "bool", "diff": "d4"}.get(tag, "f4")
        L.add(f"p6a.n2.{tag}", P6A_N2, path, kind)
    # Same two frozen encoders on the unseen-known pool (reported next to the pre-registered
    # all-rows figures, never mixed with them in one sentence).
    base = "holdouts.headline.feature_sets."
    vals = {}
    for tag, fid in (("base", "frozen_e5_base"), ("large", "frozen_e5_large")):
        path = f"{base}{fid}.unseen_known_only.auroc_mean_of_folds"
        vals[tag] = L.add(f"p6a.n2.{tag}_unseen", P6A_D1, path, "f4")
    desc = "frozen e5-large minus e5-base, unseen-known pool, mean of folds"
    L.put("p6a.n2.diff_unseen", vals["large"] - vals["base"], "d4", P6A_D1, desc)
    return {
        **{tag: f"p6a.n2.{tag}" for tag in n2k},
        "base_unseen": "p6a.n2.base_unseen",
        "large_unseen": "p6a.n2.large_unseen",
        "diff_unseen": "p6a.n2.diff_unseen",
    }


def _p6a_d2(L: Loader) -> dict[str, Any]:
    rows = []
    for rec, name in (("v1", "v1 (shipped)"), ("a1a3", "v3"), ("a1", "A1")):
        r: dict[str, Any] = {"name": name}
        for mode, tag in (("raw", "raw"), ("neutral", "neu")):
            for met, mt in (("auroc", "auroc"), ("strict_rej95", "rej")):
                k = f"p6a.d2.{rec}.{mode}.{met}"
                L.add(k, P6A_D2, f"results.{rec}.{mode}.headline.{met}", "ci")
                r[f"{tag}_{mt}"] = k
        rows.append(r)
    d2: dict[str, Any] = {"rows": rows}
    for mode, tag in (("raw", "raw"), ("neutral", "neu")):
        for met, mt in (("auroc", "auroc"), ("strict_rej95", "rej")):
            k = f"p6a.d2.delta.{mode}.{met}"
            L.add(k, P6A_D2, f"v3_minus_v1.{mode}.headline.{met}", "dci", point_key="delta")
            d2[f"{tag}_{mt}_delta"] = k
    L.add("p6a.d2.reversal", P6A_D2, "ranking_reversal", "bool")
    d2["reversal"] = "p6a.d2.reversal"
    for part in ("known_eval", "unknown"):
        for met in ("n_rows", "n_rows_with_id"):
            k = f"p6a.d2.ids.{part}.{met}"
            L.add(k, P6A_D2, f"id_prefix_counts.headline.{part}.{met}", "int")
            d2[f"ids_{part}_{met}"] = k
    return d2


def _p6a_d3(L: Loader) -> dict[str, Any]:
    d3: dict[str, Any] = {}
    for met, tag in (("retention_known", "ret"), ("rejection_unknown", "rej")):
        for stat, st in (("mean", "mean"), ("sd", "sd"), ("p2_5", "lo"), ("p97_5", "hi")):
            k = f"p6a.d3.{tag}.{st}"
            L.add(k, P6A_D3, f"pooled_over_seeds.{met}.{stat}", "f3")
            d3[f"{tag}_{st}"] = k
    per_seed = L.val(P6A_D3, "per_seed")
    for blk, tag in (("calibration_threshold", "cal"), ("crossfit_oof_threshold", "cf")):
        for mt, src in (("ret", "retention_known_eval"), ("rej", "rejection_unknown")):
            vals = [v[blk][src] for v in per_seed.values()]
            k = f"p6a.d3.{tag}.{mt}"
            desc = f"mean over seeds of per_seed.*.{blk}.{src}"
            L.put(k, sum(vals) / len(vals), "f3", P6A_D3, desc)
            d3[f"{tag}_{mt}"] = k
        thr = [v[blk]["threshold"] for v in per_seed.values()]
        k = f"p6a.d3.{tag}.thr"
        desc = f"mean over seeds of per_seed.*.{blk}.threshold"
        L.put(k, sum(thr) / len(thr), "f1", P6A_D3, desc)
        d3[f"{tag}_thr"] = k
    L.put("p6a.d3.n_seeds", len(per_seed), "int", P6A_D3, "len(per_seed)")
    lo_q, hi_q = (float(q[1:].replace("_", ".")) for q in ("p2_5", "p97_5"))
    L.put("p6a.d3.level", (hi_q - lo_q) / 100, "pct0", P6A_D3, "central interval p2_5..p97_5")
    d3["level"] = "p6a.d3.level"
    d3["n_seeds"] = "p6a.d3.n_seeds"
    m = re.match(r"(\d+) resamples", str(L.val(P6A_D3, "design.bootstrap")))
    if m is None:
        raise KeyError("design.bootstrap does not start with '<n> resamples'")
    L.put("p6a.d3.n_boot", int(m.group(1)), "int", P6A_D3, "parsed from design.bootstrap")
    d3["n_boot"] = "p6a.d3.n_boot"
    return d3


def _p6a_d4(L: Loader) -> dict[str, Any]:
    fr = L.val(P6A_D4, "design.fractions")
    ntr = L.val(P6A_D4, "n_train_rows")
    rows = []
    for i, f in enumerate(fr):
        sizes = {v for k, v in ntr.items() if k.split("|")[0] == str(f)}
        if len(sizes) != 1:  # the learning-curve rows must not depend on the seed
            raise KeyError(f"n_train_rows for fraction {f} differ across seeds: {sizes}")
        r = {"frac": f"p6a.d4.row{i}.frac", "n": f"p6a.d4.row{i}.n"}
        L.put(r["frac"], f, "pct0", P6A_D4, f"design.fractions[{i}]")
        L.put(r["n"], sizes.pop(), "int", P6A_D4, f"n_train_rows['{f}|*'] (identical over seeds)")
        for blk, tag in (("auroc", "auroc"), ("strict_rejection_95", "rej")):
            r[tag] = f"p6a.d4.row{i}.{tag}"
            L.add(r[tag], P6A_D4, f"{blk}.mean_by_fraction[{i}]", "f3")
        rows.append(r)
    d4: dict[str, Any] = {"rows": rows, "f_lo": rows[0]["frac"], "f_hi": rows[-1]["frac"]}
    for blk, tag in (("auroc", "auroc"), ("strict_rejection_95", "rej")):
        sl = f"{blk}.slope_vs_log2_fraction"
        L.add_pair(f"p6a.d4.{tag}.slope", P6A_D4, f"{sl}.slope_per_doubling", f"{sl}.ci", "dci")
        d4[f"{tag}_slope"] = f"p6a.d4.{tag}.slope"
        d4[f"{tag}_at_lo"], d4[f"{tag}_at_hi"] = rows[0][tag], rows[-1][tag]
    d4.update(_p6a_d4_extrapolation(L, rows[-1]["n"]))
    return d4


# Extrapolation targets, in multiples of the largest measured training size (2x only: the 4x
# estimate was dropped on review as too far beyond the measured range).
D4_MULTIPLES = (2,)


def _p6a_d4_extrapolation(L: Loader, n_full_key: str) -> dict[str, Any]:
    """ESTIMATES beyond the measured range: AUROC = intercept + slope * log2(rows / n_full).

    The fit's own intercept is the value at the full training size (fraction 1.0); the interval
    uses the slope CI only (the intercept is held fixed), and every value is capped at 1.0 because
    AUROC cannot exceed it. Nothing here is measured: the log-linear assumption is the whole claim.
    """
    sl = "auroc.slope_vs_log2_fraction"
    a = L.add("p6a.d4.auroc.intercept", P6A_D4, f"{sl}.intercept", "f3")
    s = L.rep.metrics["p6a.d4.auroc.slope"].value
    n_full = L.rep.metrics[n_full_key].value
    est = []
    for i, m in enumerate(D4_MULTIPLES):
        d = math.log2(m)
        val = {k: min(1.0, a + s[k] * d) for k in ("point", "lo", "hi")}
        k_n, k_m, k_a = (f"p6a.d4.est{i}.{t}" for t in ("n", "mult", "auroc"))
        how = f"ESTIMATE: min(1, intercept + slope * log2({m})); slope CI only"
        L.put(k_n, m * n_full, "int", P6A_D4, f"{m} * largest measured training size")
        L.put(k_m, m, "int", P6A_D4, f"extrapolation multiple {m}")
        L.put(k_a, val, "ci", P6A_D4, how)
        est.append({"n": k_n, "mult": k_m, "auroc": k_a})
    sat = (1.0 - a) / s["point"]  # doublings from the full size until the line reaches 1.0
    L.put("p6a.d4.sat", sat, "f1", P6A_D4, "ESTIMATE: (1 - intercept) / slope")
    return {"intercept": "p6a.d4.auroc.intercept", "est": est, "sat": "p6a.d4.sat"}


def _p6a_axes(L: Loader) -> dict[str, Any]:
    """Candidates x axes (point deltas vs ref; CIs stay in axes.json) and the not-formed list."""
    for mode, tag in (("raw", "raw"), ("neutral", "neu")):
        L.add(f"p6a.ax.ref.ret_{tag}", P6A_AXES, f"ref.retention.{mode}", "f3")
    rows: list[dict[str, str]] = []
    not_formed: list[dict[str, str]] = []
    for c in P6A_CANDS:
        cp = f"candidates.{c}."
        if L.val(P6A_AXES, cp + "status") != "scored":
            k = f"p6a.ax.{c}.reason"
            L.add(k, P6A_AXES, cp + "reason", "text")
            not_formed.append({"name": c, "reason": k})
            continue
        k = {}
        b = cp + "axes.b.modes."
        spec = (
            ("a", cp + "axes.a.delta", "d3"),
            ("b_raw_auroc", b + "raw.auroc.delta", "d3"),
            ("b_raw_rej", b + "raw.rej95.delta", "d3"),
            ("b_neu_auroc", b + "neutral.auroc.delta", "d3"),
            ("b_neu_rej", b + "neutral.rej95.delta", "d3"),
            ("ret_raw", b + "raw.retention.candidate", "f3"),
            ("ret_neu", b + "neutral.retention.candidate", "f3"),
            ("c", cp + "axes.c.delta", "d3"),
            ("d", cp + "axes.d.delta", "d3"),
            ("e", cp + "axes.e.delta", "d3"),
            ("elig", cp + "eligible", "bool"),
        )
        for tag, path, kind in spec:
            k[tag] = f"p6a.ax.{c}.{tag}"
            L.add(k[tag], P6A_AXES, path, kind)
        for tag, blk in (("imp", "improved_axes"), ("inel", "ineligible_axes")):
            names = L.val(P6A_AXES, cp + blk)
            k[tag] = f"p6a.ax.{c}.{tag}"
            L.put(k[tag], ",".join(names) if names else "none", "text", P6A_AXES, cp + blk)
        rows.append({"name": c, **k})
    # the F1 cost of I4 with its CI, and the neutral rejection gain of I1b with its CI
    for c in ("i4a", "i4b"):
        L.add(f"p6a.ax.{c}.a_ci", P6A_AXES, f"candidates.{c}.axes.a", "dci", point_key="delta")
    nb = "candidates.i1b.axes.b.modes.neutral.rej95"
    L.add("p6a.ax.i1b.neu_rej_ci", P6A_AXES, nb, "dci", point_key="delta")
    rule = str(L.val(P6A_AXES, "candidates.i1b.axes.b.rule"))
    m = re.search(r"retention >= ref - ([0-9.]+)", rule)
    if m is None:
        raise KeyError("axes.b.rule lacks 'retention >= ref - <margin>'")
    L.put("p6a.ax.guard_margin", float(m.group(1)), "f2", P6A_AXES, "parsed from axes.b.rule")
    return {"rows": rows, "not_formed": not_formed}


def _p6a_sel(L: Loader) -> dict[str, Any]:
    L.add("p6a.sel.chosen", P6A_SEL, "chosen", "text")
    L.add("p6a.sel.reason", P6A_SEL, "reason", "text")
    L.add("p6a.sel.unguarded.chosen", P6A_SEL_UNG, "chosen", "text")
    for k, rel, blk in (
        ("n_eligible", P6A_SEL, "eligible"),
        ("n_scored", P6A_SEL, "improved_axes"),
        ("n_not_formed", P6A_SEL, "not_formed"),
        ("unguarded.n_eligible", P6A_SEL_UNG, "eligible"),
    ):
        L.put(f"p6a.sel.{k}", len(L.val(rel, blk)), "int", rel, f"len({blk})")
    # DEV known-row retention at the candidate's own threshold (axes.json, 15 runs, mean)
    for c in ("i6a", "i6b"):
        for mode, tag in (("raw", "raw"), ("neutral", "neu")):
            p = f"candidates.{c}.axes.b.modes.{mode}.retention.candidate"
            L.add(f"p6a.sel.{c}.ret_{tag}", P6A_AXES, p, "f3")
    # the cross-fitted vs calibration threshold i6b used: mean over the raw DEV runs
    files = sorted((L.root / P6A / "i6b").glob("s*/views/raw/*.thr.json"))
    if not files:
        raise FileNotFoundError("no results/phase6a/i6b/s*/views/raw/*.thr.json files")
    cf: list[float] = []
    gl: list[float] = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))["per_retention"]["95"]
        cf.append(d["crossfit_threshold"])
        gl.append(d["global"])
    src = files[0].relative_to(L.root).as_posix()
    where = f"over all {len(files)} results/phase6a/i6b/s*/views/raw/*.thr.json"
    L.put("p6a.sel.i6b.thr_cf", sum(cf) / len(cf), "f1", src, f"mean {where}: crossfit_threshold")
    L.put("p6a.sel.i6b.thr_cal", sum(gl) / len(gl), "f1", src, f"mean {where}: global")
    L.put("p6a.sel.i6b.thr_n", len(files), "int", src, f"count {where}")
    # pre-ship check: i1b minus ref, paired bootstrap (confirm_report.json)
    cells = {}
    pd = "paired_delta_candidate_minus_ref."
    for mode, mt in (("raw", "raw"), ("neutral", "neu")):
        for part in ("headline", "confirm"):
            for met, mm in (("auroc", "auroc"), ("rej95", "rej"), ("retention95", "ret")):
                k = f"p6a.ship.{mt}.{part}.{mm}"
                L.add(k, P6A_CONF, f"{pd}{mode}.{part}.{met}", "dci", point_key="mean_delta")
                cells[f"{mt}_{part}_{mm}"] = k
    for part in ("headline", "confirm"):
        L.add(f"p6a.looks.{part}", P6A_CONF, f"adaptive_reuse_looks.{part}.including_6a", "int")
    # the ref disclosure: ref = v1 recipe stopped at epoch 8; shipped v1 = epoch 9
    L.add("p6a.ref.rej", P6A_CONF, "ref.raw.headline.ci95.rej95", "ci")
    L.add("p6a.ref.epoch", P6A_REF_EPOCH, "e_star", "int")
    return {"cells": cells, "chosen_name": L.val(P6A_SEL, "chosen")}


def _p6a_logs(L: Loader) -> dict[str, Any]:
    """
    Track B test-row reads made in the open-set improvement round, counted by call_type from the
    three log files.
    """
    rows = []
    total = 0
    for rel in P6A_LOGS:
        lines = _read_jsonl(L.root / rel)
        L._sha(rel)
        total += len(lines)
        for ct, cnt in sorted(Counter(r["call_type"] for r in lines).items()):
            k = f"p6a.log.{ct}"
            L.put(k, cnt, "int", rel, f"count of lines with call_type == '{ct}'")
            rows.append({"folder": rel.replace("results/", "").replace("/", " / "), "key": k})
    L.put(
        "p6a.log.total",
        total,
        "int",
        P6A_LOGS[0],
        "sum of lines over the three open-set improvement round logs",
    )
    L.put(
        "p6a.log.n_files",
        len(P6A_LOGS),
        "int",
        P6A_LOGS[0],
        "number of open-set improvement round log files read",
    )
    return {"rows": rows}


def _phase6a(L: Loader) -> None:
    """Open-set improvement round diagnostics D1-D4, the selection outcome and the pre-ship check.

    Open-set improvement round is complete, so every file is REQUIRED: a missing file raises
    FileNotFoundError and
    a file that does not match the schema below raises KeyError naming the path (no stubs).
      diag/d1_oracle.json              holdouts.headline.feature_sets.<set>.{all_rows,
                                       unseen_known_only}.{auroc_mean_of_folds,
                                       auroc_mean_of_folds_ci[lo,hi]};
                                       holdouts.headline.unsupervised_finetuned_mahalanobis.
                                       phase3_results.mean_3_seeds
      diag/n2_decision.json            values.{e5_base,e5_large,diff}_mean_of_folds, margin, run_n2
      diag/d2_id_neutral.json          results.<v1|a1a3|a1>.<raw|neutral>.headline.
                                       <auroc|strict_rej95>.{point,lo,hi}; v3_minus_v1.<mode>.
                                       headline.<metric>.{delta,lo,hi}; ranking_reversal;
                                       id_prefix_counts.headline.<known_eval|unknown>.
                                       {n_rows, n_rows_with_id}
      diag/d3_threshold_stability.json pooled_over_seeds.<retention_known|rejection_unknown>.
                                       {mean,sd,p2_5,p97_5}; per_seed.<seed>.
                                       {calibration_threshold,crossfit_oof_threshold}.
                                       {threshold,retention_known_eval,rejection_unknown};
                                       design.bootstrap ('<n> resamples ...')
      diag/d4_learning_curve.json      design.fractions; n_train_rows['<fraction>|<seed>'];
                                       {auroc|strict_rejection_95}.{mean_by_fraction,
                                       slope_vs_log2_fraction.{slope_per_doubling, ci[lo,hi]}}
      selection.json, selection_unguarded.json
                                       chosen, reason, eligible[], improved_axes{}, not_formed{}
      axes.json                        candidates.<c>.{status, eligible, improved_axes,
                                       ineligible_axes, axes.{a,c,d,e}.delta, axes.b.modes.
                                       <raw|neutral>.{auroc,rej95}.delta, .retention.candidate}
      confirm_report.json              paired_delta_candidate_minus_ref.<mode>.<part>.<metric>.
                                       {mean_delta,lo,hi}; ref.raw.headline.ci95.rej95;
                                       adaptive_reuse_looks.<part>.including_6a
      ref/epoch.json                   e_star
      i6b/s*/views/raw/*.thr.json      per_retention.95.{global, crossfit_threshold}
      *test_inference_log.jsonl        one JSON object per line with call_type
    """
    L.rep.ctx["p6a"] = {
        "d1": _p6a_d1(L),
        "n2": _p6a_n2(L),
        "d2": _p6a_d2(L),
        "d3": _p6a_d3(L),
        "d4": _p6a_d4(L),
        "ax": _p6a_axes(L),
        "sel": _p6a_sel(L),
        "logs": _p6a_logs(L),
    }


NOTEBOOK = "notebooks/intent_router_colab.ipynb"
RERUN = "results_rerun/final_wandb/rerun_summary.json"
DETERM = "results/final/determinism.json"
README = "README.md"
# README Reproduce commands rendered in the report (same text, taken from the README code block)
_REPRO_CMD = re.compile(
    r"^(python -m venv .*|python -m intent_router\.final .*--stage (?:train|evaluate)"
    r"|python -m intent_router\.trackb .*--stage headline)$"
)


def _repro(L: Loader) -> None:
    """Deliverables and reproducibility box: determinism evidence, W&B re-run, README commands."""
    c = L.rep.ctx
    # determinism: two identical trainings of the final model, compared bit for bit
    L.add("det.seed", DETERM, "seed", "int")
    L.add("det.state", DETERM, "state_dict_identical", "bool")
    L.add("det.bitwise", DETERM, "bitwise_identical", "bool")
    L.add("det.epochs", DETERM, "epoch_history_identical", "bool")
    L.add("det.val_logit_diff", DETERM, "val.logits.max_abs_diff", "sci")
    L.add("det.test_logit_diff", DETERM, "test.logits.max_abs_diff", "sci")
    L.add("det.val_prob_diff", DETERM, "val.softmax_float32.max_abs_diff", "sci")
    L.add("det.test_prob_diff", DETERM, "test.softmax_float32.max_abs_diff", "sci")
    fp = L.val(DETERM, "fingerprint_run1")
    L.put("det.fp", fp[:8], "text", DETERM, "first 8 hex chars of fingerprint_run1")
    # W&B live re-run of the train stage against the committed fingerprint
    L.add("rr.match", RERUN, "fingerprint_match", "bool")
    L.add("rr.asserted", RERUN, "fingerprint_asserted", "bool")
    L.add("rr.epochs", RERUN, "epochs_run", "int")
    L.add("rr.wall", RERUN, "wall_clock_s", "s1")
    rr_fp = L.val(RERUN, "fingerprint_got")
    L.put("rr.fp", rr_fp[:8], "text", RERUN, "first 8 hex chars of fingerprint_got")
    c["rr_fp_eq_det"] = rr_fp == fp  # drives a sentence; the keys above carry the numbers
    # notebook header: the T4 runtime ESTIMATE figures, as labelled there
    nb = json.loads((L.root / NOTEBOOK).read_text(encoding="utf-8"))
    head = "".join(nb["cells"][0]["source"])
    pat = r"v1 training ~(\d+) min.*?Track B ~(\d+)-(\d+) min.*?about (\d+)-(\d+) min"
    m = re.search(pat, head, re.S)
    if m is None:
        raise ValueError(f"{NOTEBOOK}: runtime estimate line not found in the header cell")
    names = ("nb.train_min", "nb.tb_lo", "nb.tb_hi", "nb.tot_lo", "nb.tot_hi")
    for key, v in zip(names, m.groups(), strict=True):
        L.put(key, int(v), "int", NOTEBOOK, f"header runtime estimate figure {key}")
    # README Reproduce: the exact local commands for install, train, evaluate, Track B headline
    txt = (L.root / README).read_text(encoding="utf-8")
    block = re.search(r"```bash\n(.*?)```", txt[txt.index("## Reproduce") :], re.S)
    if block is None:
        raise ValueError(f"{README}: no bash block under '## Reproduce'")
    cmds = []
    for ln in block.group(1).splitlines():
        ln = re.sub(r"\s+#.*$", "", ln.strip())  # drop the trailing comment
        if _REPRO_CMD.match(ln):
            cmds.append(ln)
    if len(cmds) != 4:
        raise ValueError(f"{README}: expected 4 Reproduce commands, found {len(cmds)}: {cmds}")
    for i, cmd in enumerate(cmds):
        L.put(f"repro.cmd{i}", cmd, "text", README, "command line copied from the Reproduce block")
    c["repro_cmds"] = [f"repro.cmd{i}" for i in range(len(cmds))]
    _repro_colab(L, c)


RUN1 = "results/colab/reproduction_run1.json"
RUN2 = "results/colab/reproduction_run2.json"  # canonical Colab run
HEADLINE = "results/trackb/headline.json"
_EDGE_ROW = "Track B msp rejection@95"  # prefix of the table row that sits on its tolerance edge


def _repro_colab(L: Loader, c: dict[str, Any]) -> None:
    """Colab reproduction verdict + fresh-vs-committed table + edge disclosure + determinism facts.

    Driven by results/colab/reproduction_run{1,2}.json (a draft without them skips the block).
    The Colab-determinism claim is a flag (`repro_runs_identical`) computed here by comparing the
    two runs programmatically; the template prints the claim only when it is true.
    """
    c["repro_colab"] = L.exists(RUN2)
    c["repro_rows"] = []
    c["repro_edge"] = False
    c["repro_edge_row"] = -1
    c["repro_edge_coincidence"] = False
    c["repro_runs_identical"] = False
    if not c["repro_colab"]:
        return
    if not L.exists(RUN1):
        raise ValueError(f"{RUN2} exists but {RUN1} is missing: both Colab runs are needed")
    L.add("rc.verdict", RUN2, "verdict_recomputed", "text")
    t1, t2 = L.val(RUN1, "table_recomputed"), L.val(RUN2, "table_recomputed")
    rows = []
    for i, row in enumerate(t2):
        k, b = f"rc.row{i}", f"table_recomputed[{i}]"
        L.add(f"{k}.metric", RUN2, f"{b}.metric", "text")
        L.add(f"{k}.fresh", RUN2, f"{b}.fresh", "f4")
        L.add(f"{k}.committed", RUN2, f"{b}.committed", "f4")
        L.add(f"{k}.delta", RUN2, f"{b}.delta", "d4")
        L.put(f"{k}.tol", row["tolerance"].replace("+-", "±"), "text", RUN2, f"{b}.tolerance")
        L.add(f"{k}.result", RUN2, f"{b}.result", "text")
        rows.append(
            {q: f"{k}.{q}" for q in ("metric", "fresh", "committed", "delta", "tol", "result")}
        )
    c["repro_rows"] = rows
    n_pass = sum(r["result"] == "PASS" for r in t2)
    L.put("rc.n_pass", n_pass, "int", RUN2, "rows of table_recomputed with result PASS")
    L.put("rc.n_rows", len(t2), "int", RUN2, "len(table_recomputed)")
    # fingerprints: fresh (both Colab runs) against the committed one
    fp1, fp2 = L.val(RUN1, "fresh_fingerprint"), L.val(RUN2, "fresh_fingerprint")
    fpc = L.val(RUN2, "committed_fingerprint")
    L.put("rc.fp1", fp1[:8], "text", RUN1, "first 8 hex chars of fresh_fingerprint")
    L.put("rc.fp2", fp2[:8], "text", RUN2, "first 8 hex chars of fresh_fingerprint")
    L.put("rc.fp_committed", fpc[:8], "text", RUN2, "first 8 hex chars of committed_fingerprint")
    L.add("rc.accel", RUN2, "runtime.accelerator", "text")
    c["repro_runs_identical"] = (
        fp1 == fp2
        and fp1 != fpc
        and [(r["metric"], r["fresh"]) for r in t1] == [(r["metric"], r["fresh"]) for r in t2]
        and all(r["result"] == "PASS" for r in t1 + t2)
    )
    # edge disclosure: the MSP rejection@95 row whose |delta| equals its tolerance
    for i, row in enumerate(t2):
        tol = float(row["tolerance"].lstrip("+-"))
        if row["metric"].startswith(_EDGE_ROW) and abs(row["delta"]) >= tol - 1e-9:
            c["repro_edge"] = True
            c["repro_edge_row"] = i
    if c["repro_edge"]:
        seeds = L.val(RUN2, "trackb_per_seed_msp_vs_maha_ft")
        msp, maha = seeds["msp"].values(), seeds["maha_ft"].values()
        n_unk = sum(s["n_unknown"] for s in msp)
        n_cal = {s["n_cal_known"] for s in msp}
        if len(n_cal) != 1:
            raise ValueError(f"{RUN2}: calibration rows differ across seeds: {n_cal}")
        rej = sum(s["n_unknown_rejected"] for s in msp)
        row = t2[c["repro_edge_row"]]
        if round(rej / n_unk, 4) != row["fresh"]:
            raise ValueError(f"{RUN2}: per-seed counts {rej}/{n_unk} disagree with {row['fresh']}")
        committed = L.val(HEADLINE, "methods.msp.strict_rejection_recall.values")
        n_eval = L.val(HEADLINE, "n_eval_unknown")
        c_rej = round(sum(committed) * n_eval)
        if round(c_rej / (n_eval * len(committed)), 4) != row["committed"]:
            raise ValueError(f"{HEADLINE}: committed msp counts disagree with {row['committed']}")
        L.put("rc.edge.fresh_n", rej, "int", RUN2, "sum of msp n_unknown_rejected over seeds")
        L.put("rc.edge.unk_n", n_unk, "int", RUN2, "sum of msp n_unknown over seeds")
        L.put(
            "rc.edge.committed_n",
            c_rej,
            "int",
            HEADLINE,
            "sum(msp per-seed rejection) x n_eval_unknown",
        )
        L.put(
            "rc.edge.maha_n",
            sum(s["n_unknown_rejected"] for s in maha),
            "int",
            RUN2,
            "sum of maha_ft n_unknown_rejected over seeds",
        )
        L.put("rc.edge.cal_n", next(iter(n_cal)), "int", RUN2, "n_cal_known per seed")
        # the equal totals are a coincidence only if the per-seed counts differ
        c["repro_edge_coincidence"] = sum(s["n_unknown_rejected"] for s in maha) == rej and [
            s["n_unknown_rejected"] for s in maha
        ] != [s["n_unknown_rejected"] for s in msp]


GAP_BOX_TITLE = "The 3–4-hour path"  # the task's own wording; the template holds no numerals
SS_FORMULA = "n = (z(1-α/2)·√ψ + z(1-β)·√(ψ − δ²))² / δ²"  # shown verbatim in the section cell
SS_ALPHA, SS_POWER = 0.05, 0.80  # two-sided alpha and power of the sample-size ESTIMATE


def mcnemar_required_n(b: int, c: int, n: int, alpha: float, power: float) -> int | None:
    """Rows needed to resolve an observed paired difference with the McNemar test (ESTIMATE).

    n_req = (z_{1-a/2} sqrt(psi) + z_{1-b} sqrt(psi - delta^2))^2 / delta^2, with psi = (b + c) / n
    and delta = (b - c) / n taken from the observed discordant counts (Connor 1987 approximation).
    Returns None when delta is 0 or the square root is undefined (no estimate can be formed).
    """
    psi, delta = (b + c) / n, (b - c) / n
    if delta == 0 or psi - delta**2 <= 0:
        return None
    z = NormalDist()
    root = z.inv_cdf(1 - alpha / 2) * math.sqrt(psi) + z.inv_cdf(power) * math.sqrt(psi - delta**2)
    return math.ceil(root**2 / delta**2)


def _gaps(L: Loader) -> None:
    """Numbers for the gaps table in the Production section (keys `gap.*`).

    Everything else the section prints reuses keys registered above; only values that are new
    (headline/CONFIRM rejection read from the files the task names, interval widths, the CV
    difference vs B1 and the sample-size ESTIMATE) are added here.
    """
    m = L.rep.metrics
    L.add("gap.open.head_rej", HEAD, ("methods", "maha_ft", "strict_rejection_recall", "mean"))
    L.add("gap.open.head_ret", HEAD, ("methods", "maha_ft", "retention_known", "mean"))
    L.add(
        "gap.open.confirm_rej",
        TBI + "confirm.json",
        ("entries", "base/maha_ft", "confirm", "mean", "op95", "strict_rejection_recall"),
    )
    L.add("gap.b1.cv_diff", SEL, "baseline_context_ttests.B1.mean_diff", "d3")
    d = m["a.B1.delta"].value
    L.put("gap.b1.delta_w", d["hi"] - d["lo"], "f3", TA, "hi - lo of the paired macro-F1 delta CI")
    f = m["a.f1"].value
    L.put("gap.f1_w", f["hi"] - f["lo"], "f3", TA, "hi - lo of the test macro-F1 CI")
    n_test = int(m["a.n"].value)
    b, c = int(m["a.B1.mc_a"].value), int(m["a.B1.mc_b"].value)
    mc = "test.baselines.B1.mcnemar_exact_on_correctness"
    if b + c != L.val(TA, f"{mc}.n_discordant"):
        raise ValueError("McNemar a_only + b_only differs from n_discordant")
    need = mcnemar_required_n(b, c, n_test, SS_ALPHA, SS_POWER)
    L.put("gap.ss.alpha", SS_ALPHA, "f2", TA, "assumption: two-sided alpha", fam="gap.ss")
    L.put("gap.ss.power", SS_POWER, "pct0", TA, "assumption: power", fam="gap.ss")
    L.put("gap.ss.psi", (b + c) / n_test, "f3", TA, f"({mc}.a + .b) / test.n", fam="gap.ss")
    L.put("gap.ss.delta", (b - c) / n_test, "f3", TA, f"({mc}.a - .b) / test.n", fam="gap.ss")
    if need is not None:
        desc = "ceil((z(1-a/2) sqrt(psi) + z(power) sqrt(psi - delta^2))^2 / delta^2)"
        L.put("gap.ss.n", need, "int", TA, desc, fam="gap.ss")
        L.put("gap.ss.mult", need / n_test, "f1", TA, "gap.ss.n / test.n", fam="gap.ss")
    L.rep.ctx["gap"] = {
        "box_title": GAP_BOX_TITLE,
        "ss_ok": need is not None,
        "ss_formula": SS_FORMULA,
        "delta_ci_has_zero": bool(d["lo"] < 0 < d["hi"]),
    }


def load(root: Path = ROOT) -> Report:
    """Read every results source and return the Report (flat metrics + table structures)."""
    L = Loader(root)
    for step in (
        _track_a,
        _bakeoff,
        _data_quirks,
        _protocol,
        _track_b,
        _errors,
        _robustness,
        _llm,
        _selection_history,
        _serving,
        _phase3b,
        _phase6a,
        _repro,
        _gaps,
    ):
        step(L)
    figs = {
        "confusion": [TA],
        "reliability": [TA],
        "tb_methods": [HEAD, LOCO],
        "v1_v3": [TRADE],
    }
    for name, srcs in figs.items():
        L.fig(name, srcs)
    L.rep.ctx["classes"] = [x["label"] for x in L.val(TA, "test.classes")]
    return L.rep


# --------------------------------------------------------------------------- section numbering

TEMPLATE = ROOT / "report" / "template.html.j2"
# Headings other text cites by number (main sections) or letter (appendix). Titles, never numbers
# or letters, live here: the number or letter is derived from the template's own heading order.
MAIN_TITLES = {
    "approach": "Approach and Model Selection",
    "quirks": "Data Characteristics and Handling",
    "tracka": "Closed-Set Classification (Track A)",
    "trackb": "Open-Set Abstention (Track B)",
    "errors": "Error Analysis",
    "production": "Deployment and Monitoring",
    "next": "Next Steps",
}
APPENDIX_TITLES = {
    "debrief": "Debrief Answers",
    "trackb_full": "Open-Set Evaluation in Full: Scorers, Leave-One-Class-Out and Business Framing",
    "diag": "Diagnostics (Oracle Probe, ID-Neutral Scoring, Threshold Stability, Learning Curve)",
    "history": "Selection History and the v1/v3 Decision",
    "robust": "Robustness",
    "llm": "LLM Baseline",
    "latency": "Latency, Quantization and Cost",
    "repro": "Reproduction and Determinism",
    "ablations": "Ablations and Bake-Off Detail",
    "exposure": "Test-Split Exposure",
    "more": "Further Detail",
}


def heading_map(template: Path = TEMPLATE) -> dict[int, str]:
    """Numbered main-text headings {1: title, ...} in template order.

    The report numbers every <h2> inside <main> with a CSS counter (report.css `counter(sec)`), so
    the number is the h2's 1-based position there. No h2 inside <main> is conditional.
    """
    src = template.read_text(encoding="utf-8")
    body = src.split("<main>", 1)[1].split("</main>", 1)[0]
    return dict(enumerate(_h2_titles(body), start=1))


def _h2_titles(fragment: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r"<h2[^>]*>(.*?)</h2>", fragment)]


def appendix_map(template: Path = TEMPLATE) -> dict[str, str]:
    """Lettered appendix headings {'A': title, ...} in template order.

    The appendix letters are a CSS counter over the <h2>s of <section class="app"> (report.css
    `counter(app, upper-alpha)`). The private example-text appendix (--include-text) is conditional
    and comes last, so it is cut off here.
    """
    src = template.read_text(encoding="utf-8")
    body = src.split('<section class="app">', 1)[1].split("{% if include_text %}", 1)[0]
    return {chr(ord("A") + i): t for i, t in enumerate(_h2_titles(body))}


def section_numbers(template: Path = TEMPLATE) -> dict[str, int | str]:
    """{'trackb': 4, 'diag': 'C', ...}: the report's own numbers (main) and letters (appendix) for
    the sections other text cites. Raises KeyError when a registered title is missing."""
    main = {t: n for n, t in heading_map(template).items()}
    app = {t: letter for letter, t in appendix_map(template).items()}
    out: dict[str, int | str] = {k: main[t] for k, t in MAIN_TITLES.items()}
    out.update({k: app[t] for k, t in APPENDIX_TITLES.items()})
    return out


# --------------------------------------------------------------------------- opt-in text appendix


def load_examples(root: Path = ROOT) -> list[dict[str, str]]:
    """PRIVATE review only (`--include-text`): ids -> verbatim text for the error examples.

    Reads data/dataset.csv (confidential, gitignored). Never called by the default build.
    """
    err = json.loads((root / ERR).read_text(encoding="utf-8"))
    ids: list[tuple[str, str, str, str]] = []
    for section, tag in (("test", "test error"), ("oof_all_fold_runs", "CV confusion")):
        for tc in err.get(section, {}).get("top_confusions", []) or []:
            for i in tc["example_ids"]:
                ids.append((i, tag, tc["gold"], tc["pred"]))
    for tc in err["oof"]["top_confusions"]:
        for i in tc["example_ids"]:
            ids.append((i, "CV confusion", tc["gold"], tc["pred"]))
    text = {}
    with (root / "data" / "dataset.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            text[r["id"]] = r["text"]
    seen: set[str] = set()
    out = []
    for i, tag, g, p in ids:
        if i in seen or i not in text:
            continue
        seen.add(i)
        out.append({"id": i, "kind": tag, "gold": g, "pred": p, "text": text[i]})
    return out
