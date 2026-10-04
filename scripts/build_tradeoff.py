"""Build the v1 vs v3 deployment tradeoff table from existing results JSON.

Deterministic and inference-free: reads only files under results/, hand-types no numbers, and
attaches a ``source`` (results-relative path + JSON path) to every number it emits. Writes
results/tradeoff_v1_v3.json and results/tradeoff_v1_v3.md.

    python scripts/build_tradeoff.py

Sign convention: every ``delta`` is v3 minus v1. Sources that report a *reduction*
(v1 - candidate, axes c and e) have their interval negated so the CI refers to v3 - v1 as well.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "tradeoff_v1_v3.json"
OUT_MD = RESULTS / "tradeoff_v1_v3.md"

# Fixed guidance strings (a deployment decision, not a measurement).
DEPLOYMENT_GUIDANCE = [
    "v1 = ship default (better open-set rejection on the scored headline holdout)",
    "v3 = choose when ID-prefix shortcut robustness and translation robustness matter more "
    "than open-set rejection",
]

# Track B (open-set) metrics: (key in confirm_a1a3.json, label, higher_is_better).
TRACKB_METRICS = [
    ("auroc", "AUROC", True),
    ("strict_recall_95", "strict rejection recall @95% retention", True),
    ("strict_recall_90", "strict rejection recall @90% retention", True),
    ("retention_95", "known retention @95", True),
    ("retention_90", "known retention @90", True),
    ("fpr_at_95tpr", "FPR @95% TPR", False),
]

_cache: dict[str, Any] = {}


def _load(rel: str) -> Any:
    """Load a results-relative JSON file (cached)."""
    if rel not in _cache:
        _cache[rel] = json.loads((RESULTS / rel).read_text(encoding="utf-8"))
    return _cache[rel]


def _get(rel: str, *path: str | int) -> Any:
    """Walk ``path`` through the JSON document ``rel``; raises KeyError on a missing key."""
    node = _load(rel)
    for p in path:
        node = node[p]
    return node


def _src(rel: str, *path: str | int) -> str:
    """Source string: results-relative file + JSONPath."""
    return f"{rel}:$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path)


def scalar(rel: str, *path: str | int) -> dict[str, Any]:
    """A single number with its source."""
    return {"value": _get(rel, *path), "source": _src(rel, *path)}


def interval(rel: str, *path: str | int) -> dict[str, Any]:
    """A {point, lo, hi} node with its source."""
    n = _get(rel, *path)
    return {"point": n["point"], "lo": n["lo"], "hi": n["hi"], "source": _src(rel, *path)}


def paired(rel: str, *path: str | int, v1_key: str = "v1", v3_key: str = "candidate") -> dict:
    """Axis node with v1 / v3 values and the v3-v1 delta with its CI (from axes.json shape).

    Intervals reported as a reduction (v1 - candidate) are negated into v3 - v1 terms.
    """
    n = _get(rel, *path)
    lo, hi = n["lo"], n["hi"]
    if str(n.get("ci_of", "")).startswith("reduction"):
        lo, hi = -hi, -lo
    return {
        "v1": n[v1_key],
        "v3": n[v3_key],
        "delta_v3_minus_v1": n["delta"],
        "ci95": [lo, hi],
        "ci_negated_from_reduction": str(n.get("ci_of", "")).startswith("reduction"),
        "source": _src(rel, *path),
    }


def _verdict_row(name: str, higher_is_better: bool, node: dict[str, Any]) -> dict[str, Any]:
    """Render one verdict row from a paired node (numbers only; wording is generated)."""
    delta = node["delta_v3_minus_v1"]
    lo, hi = node["ci95"]
    excludes = lo > 0 or hi < 0
    gain = delta if higher_is_better else -delta
    better = "tie" if gain == 0 else ("v3" if gain > 0 else "v1")
    if better == "tie":
        text = f"{name}: tie (delta 0)"
    else:
        ci_txt = "CI excludes zero" if excludes else "CI includes zero (not distinguishable)"
        text = (
            f"{name}: {better} better by point estimate "
            f"(v3-v1 = {delta:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]; {ci_txt})"
        )
    return {
        "axis": name,
        "higher_is_better": higher_is_better,
        "better_by_point": better,
        "ci_excludes_zero": excludes,
        "text": text,
    }


def _unpaired_row(name: str, higher_is_better: bool, v1: dict, v3: dict) -> dict[str, Any]:
    """Verdict row for single-evaluation numbers: point difference + whether CIs overlap."""
    delta = v3["point"] - v1["point"]
    overlap = not (v3["lo"] > v1["hi"] or v1["lo"] > v3["hi"])
    gain = delta if higher_is_better else -delta
    better = "tie" if gain == 0 else ("v3" if gain > 0 else "v1")
    text = (
        f"{name}: {better} better by point estimate (v3-v1 = {delta:+.4f}; "
        f"separate 95% CIs {'overlap' if overlap else 'do not overlap'}; no paired CI)"
    )
    return {
        "axis": name,
        "higher_is_better": higher_is_better,
        "better_by_point": better,
        "ci_excludes_zero": None,
        "separate_cis_overlap": overlap,
        "text": text,
    }


def build_tradeoff() -> dict[str, Any]:
    """Assemble the full tradeoff document from results/ JSON (no inference)."""
    cmp_ = "phase4e/final_v1_vs_v3.json"
    ta = {"v1": "final_v1/track_a.json", "v3": "final_v3/track_a.json"}
    ood = {"v1": "final_v1/ood_shipped.json", "v3": "final_v3/ood_shipped.json"}
    axes = "phase4e/axes.json"
    a1a3 = ("candidates", "a1a3", "axes")
    conf = "phase4e/confirm_a1a3.json"

    test_a: dict[str, Any] = {
        "label": _get(cmp_, "label"),
        "label_source": _src(cmp_, "label"),
    }
    for ver, key in (("v1", "v1"), ("v3", "v3")):
        test_a[ver] = {
            "model_fingerprint": scalar(cmp_, key, "model_fingerprint"),
            "macro_f1": interval(cmp_, key, "macro_f1"),
            "accuracy": interval(cmp_, key, "accuracy"),
            "temperature": scalar(cmp_, key, "temperature"),
            "ece_test_after_scaling": scalar(cmp_, key, "ece_test_after"),
            "parent_accuracy": scalar(ta[ver], "test", "hierarchy", "parent_accuracy"),
            "selective_prediction": {
                "threshold": scalar(cmp_, key, "selective_at_threshold", "threshold"),
                "coverage": scalar(cmp_, key, "selective_at_threshold", "coverage"),
                "accuracy_on_accepted": scalar(cmp_, key, "selective_at_threshold", "accuracy"),
                "macro_f1_present": scalar(cmp_, key, "selective_at_threshold", "macro_f1_present"),
                "n_accepted": scalar(cmp_, key, "selective_at_threshold", "n_accepted"),
                "aurc_temp_scaled_msp": scalar(
                    ta[ver], "test", "selective", "aurc_temp_scaled_msp"
                ),
            },
        }

    cv = {
        "note": "CV axis (a): macro-F1, seeds averaged inside each paired bootstrap resample",
        "v1_epoch_e_star": scalar(axes, "v1_epoch", "e_star"),
        "v1_own_argmax_epoch": scalar(axes, "v1_epoch", "own_argmax"),
        "v3_epoch_e_star": scalar(axes, "candidates", "a1a3", "e_star"),
        "v3_vs_v1_at_e_star": paired(axes, *a1a3, "a"),
        "v3_vs_v1_at_own_argmax_sensitivity": paired(
            axes, "candidates", "a1a3", "sensitivity_a_v1_at_own_argmax"
        ),
    }

    trackb_dev = {
        "note": "Track B DEV (4e axis b), CV out-of-fold open-set scoring (maha_ft)",
        "auroc": paired(axes, *a1a3, "b", "auroc"),
        "rej95": paired(axes, *a1a3, "b", "rej95"),
    }

    trackb_holdout: dict[str, Any] = {}
    for split in ("confirm", "headline"):
        rows = {}
        for key, label, hib in TRACKB_METRICS:
            d = _get(conf, "paired_delta_candidate_minus_v1", split, key)
            src = _src(conf, "paired_delta_candidate_minus_v1", split, key)
            rows[key] = {
                "label": label,
                "higher_is_better": hib,
                "v1": interval(conf, "v1", split, "ci95", key),
                "v3": interval(conf, "candidate", split, "ci95", key),
                "delta_v3_minus_v1": d["mean_delta"],
                "ci95": [d["lo"], d["hi"]],
                "source": src,
            }
        trackb_holdout[split] = rows

    robustness = {
        "neutral_swap_flip_rate (axis c, lower is better)": paired(axes, *a1a3, "c"),
        "translation_agreement (axis d, higher is better)": paired(axes, *a1a3, "d"),
        "oof_ece (axis e, lower is better)": paired(axes, *a1a3, "e"),
    }

    serving = {
        ver: {
            "shipped_ood_method": scalar(ood[ver], "method"),
            "ood_threshold": scalar(ood[ver], "threshold"),
            "val_retention_achieved": scalar(ood[ver], "val_retention_achieved"),
            "test_coverage_at_threshold": scalar(ood[ver], "test_coverage_at_threshold"),
        }
        for ver in ("v1", "v3")
    }

    # Verdict: text rendered from the numbers above.
    rows: list[dict[str, Any]] = []
    t1, t3 = test_a["v1"], test_a["v3"]
    rows.append(_unpaired_row("Track A test macro-F1", True, t1["macro_f1"], t3["macro_f1"]))
    rows.append(_unpaired_row("Track A test accuracy", True, t1["accuracy"], t3["accuracy"]))
    rows.append(
        _verdict_row("CV macro-F1 (v1 at e*=shipped epoch)", True, cv["v3_vs_v1_at_e_star"])
    )
    rows.append(
        _verdict_row(
            "CV macro-F1 (v1 at its own argmax epoch)",
            True,
            cv["v3_vs_v1_at_own_argmax_sensitivity"],
        )
    )
    rows.append(_verdict_row("Track B DEV AUROC", True, trackb_dev["auroc"]))
    rows.append(_verdict_row("Track B DEV rejection@95", True, trackb_dev["rej95"]))
    for split in ("confirm", "headline"):
        for key, label, hib in TRACKB_METRICS:
            r = trackb_holdout[split][key]
            rows.append(
                _verdict_row(
                    f"Track B {split.upper()} {label}",
                    hib,
                    {"delta_v3_minus_v1": r["delta_v3_minus_v1"], "ci95": r["ci95"]},
                )
            )
    names = [
        ("neutral-swap flip rate", False),
        ("translation agreement", True),
        ("OOF ECE", False),
    ]
    for (nm, hib), node in zip(names, robustness.values(), strict=True):
        rows.append(_verdict_row(nm, hib, node))

    return {
        "label": "v1 (shipped) vs v3 (robustness variant); all deltas are v3 minus v1",
        "generated_by": "scripts/build_tradeoff.py (reads results/ JSON only; no inference)",
        "versions": {
            "v1": {"model_fingerprint": t1["model_fingerprint"]},
            "v3": {"model_fingerprint": t3["model_fingerprint"]},
        },
        "track_a_test": test_a,
        "cv": cv,
        "trackb_dev": trackb_dev,
        "trackb_holdout": trackb_holdout,
        "robustness_cv": robustness,
        "serving": serving,
        "verdict": {"rows": rows, "text": [r["text"] for r in rows]},
        "deployment_guidance": DEPLOYMENT_GUIDANCE,
    }


def _f(x: float) -> str:
    return f"{x:.4f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def _iv(n: dict[str, Any]) -> str:
    return f"{_f(n['point'])} {_ci(n['lo'], n['hi'])}"


def render_markdown(doc: dict[str, Any]) -> str:
    """Render the document as markdown tables (every row's source is in the JSON)."""
    L: list[str] = ["# v1 vs v3 deployment tradeoff", "", doc["label"], ""]
    L += [
        "Generated by `scripts/build_tradeoff.py`; per-number sources are in "
        "`results/tradeoff_v1_v3.json`.",
        "",
    ]

    ta = doc["track_a_test"]
    L += [
        "## Track A test (one logged evaluation each, n=74)",
        "",
        "| metric | v1 | v3 |",
        "|---|---|---|",
    ]
    for k, lab in (("macro_f1", "macro-F1 (95% CI)"), ("accuracy", "accuracy (95% CI)")):
        L.append(f"| {lab} | {_iv(ta['v1'][k])} | {_iv(ta['v3'][k])} |")
    for k, lab in (
        ("temperature", "temperature"),
        ("ece_test_after_scaling", "test ECE after scaling"),
        ("parent_accuracy", "parent-level accuracy"),
    ):
        L.append(f"| {lab} | {_f(ta['v1'][k]['value'])} | {_f(ta['v3'][k]['value'])} |")
    for k, lab in (
        ("threshold", "selective threshold"),
        ("coverage", "selective coverage"),
        ("accuracy_on_accepted", "accuracy on accepted"),
        ("macro_f1_present", "macro-F1 on accepted (present classes)"),
        ("aurc_temp_scaled_msp", "AURC (lower is better)"),
    ):
        a, b = ta["v1"]["selective_prediction"][k], ta["v3"]["selective_prediction"][k]
        L.append(f"| {lab} | {_f(a['value'])} | {_f(b['value'])} |")
    L.append("")

    def paired_table(title: str, items: dict[str, dict[str, Any]]) -> None:
        L.extend(
            [
                f"## {title}",
                "",
                "| axis | v1 | v3 | v3 - v1 | 95% CI of delta |",
                "|---|---|---|---|---|",
            ]
        )
        for name, n in items.items():
            L.append(
                f"| {name} | {_f(n['v1'])} | {_f(n['v3'])} | {n['delta_v3_minus_v1']:+.4f} | "
                f"{_ci(*n['ci95'])} |"
            )
        L.append("")

    cv = doc["cv"]
    paired_table(
        "CV macro-F1 (axis a)",
        {
            f"v1 at shipped epoch e*={cv['v1_epoch_e_star']['value']} "
            f"(v3 e*={cv['v3_epoch_e_star']['value']})": cv["v3_vs_v1_at_e_star"],
            f"v1 at own argmax epoch {cv['v1_own_argmax_epoch']['value']} (sensitivity)": cv[
                "v3_vs_v1_at_own_argmax_sensitivity"
            ],
        },
    )
    paired_table(
        "Track B DEV (axis b)",
        {"AUROC": doc["trackb_dev"]["auroc"], "rejection@95": doc["trackb_dev"]["rej95"]},
    )
    for split in ("confirm", "headline"):
        L.extend(
            [
                f"## Track B {split.upper()} holdout",
                "",
                "| metric | v1 (95% CI) | v3 (95% CI) | v3 - v1 | 95% CI of delta |",
                "|---|---|---|---|---|",
            ]
        )
        for r in doc["trackb_holdout"][split].values():
            L.append(
                f"| {r['label']} | {_iv(r['v1'])} | {_iv(r['v3'])} | "
                f"{r['delta_v3_minus_v1']:+.4f} | {_ci(*r['ci95'])} |"
            )
        L.append("")
    paired_table("Robustness and calibration (CV)", doc["robustness_cv"])

    L += ["## Serving (shipped OOD threshold)", "", "| item | v1 | v3 |", "|---|---|---|"]
    for k in ("ood_threshold", "val_retention_achieved", "test_coverage_at_threshold"):
        L.append(
            f"| {k} | {_f(doc['serving']['v1'][k]['value'])} | "
            f"{_f(doc['serving']['v3'][k]['value'])} |"
        )
    L.append("")
    L += ["## Verdict (rendered from the numbers above)", ""]
    L += [f"- {t}" for t in doc["verdict"]["text"]]
    L += ["", "## Deployment guidance", ""]
    L += [f"- {t}" for t in doc["deployment_guidance"]]
    L.append("")
    return "\n".join(L)


def main() -> None:
    """Build and write the JSON and markdown artifacts."""
    doc = build_tradeoff()
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(doc), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
