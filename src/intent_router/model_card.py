"""Render the Hub model card from results JSON ONLY via a Jinja2 template.

    python -m intent_router.model_card --version v1 --out outputs/hub_stage/v1/README.md
    python -m intent_router.model_card --version v3 --out ... --publish   # final (public) card

Every number in the card is read from `results/**.json` (plus the label descriptions that live in
`llm_audit.LABEL_DESCRIPTIONS` and the LLM model tags in `configs/llm_baseline.yaml`); nothing is
typed into the template. Only aggregates are rendered: never dataset text, ids or rows.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import jinja2
import yaml

REPO_ID = "gauravgandhi2411/multilingual-intent-router"
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "model_card.md.j2"

# version -> where its results live; v1 is the shipped model (author decision 2026-10-03).
VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {
        "results": "results/final",
        "robustness": "results/robustness",
        "branch": "main",
        "shipped": True,
    },
    "v3": {
        "results": "results/final_v3",
        "robustness": "results/phase4e/final_v3_robustness",
        "branch": "robust-v3",
        "shipped": False,
    },
}
TRADEOFF_SWAP_KEY = "neutral_swap_flip_rate (axis c, lower is better)"
TRADEOFF_TRANSLATION_KEY = "translation_agreement (axis d, higher is better)"
GITHUB_URL = "https://github.com/gaurav-gandhi-2411/multilingual-intent-router"
COLAB_URL = (
    "https://colab.research.google.com/github/gaurav-gandhi-2411/multilingual-intent-router"
    "/blob/v1.0-submission/notebooks/intent_router_colab.ipynb"
)


# ------------------------------------------------------------------------------ filters
def _num(x: Any, fmt: str) -> str:
    return "n/a" if x is None else format(float(x), fmt)


def f2(x: Any) -> str:
    """Two decimals."""
    return _num(x, ".2f")


def f3(x: Any) -> str:
    """Three decimals."""
    return _num(x, ".3f")


def f4(x: Any) -> str:
    """Four decimals."""
    return _num(x, ".4f")


def pct(x: Any) -> str:
    """Fraction -> percent with one decimal (no % sign)."""
    return "n/a" if x is None else format(float(x) * 100.0, ".1f")


def pct0(x: Any) -> str:
    """Fraction -> percent with no decimals (no % sign)."""
    return "n/a" if x is None else format(float(x) * 100.0, ".0f")


def ci(d: dict[str, Any] | None) -> str:
    """{point, lo, hi} -> 'p [lo, hi]' (three decimals)."""
    if not d or d.get("point") is None:
        return "n/a"
    return f"{f3(d['point'])} [{f3(d['lo'])}, {f3(d['hi'])}]"


def ci_or_f3(x: Any) -> str:
    """A {point, lo, hi} node -> 'p [lo, hi]'; a bare number -> three decimals."""
    return ci(x) if isinstance(x, dict) else f3(x)


def make_env() -> jinja2.Environment:
    """Jinja environment: strict undefined (a missing JSON field is an error, not a blank)."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=jinja2.StrictUndefined,
        autoescape=False,  # noqa: S701 - markdown output, not HTML
        keep_trailing_newline=True,
    )
    env.filters.update(f2=f2, f3=f3, f4=f4, pct=pct, pct0=pct0, ci=ci, ci_or_f3=ci_or_f3)
    return env


# --------------------------------------------------------------------------- context
def _load(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _idswap_test(root: Path, rel_dir: str) -> dict[str, float] | None:
    """Pooled identifier-prefix-swap flip rate over the test (row, swap) pairs, or None."""
    rows = [r for r in _load(root, f"{rel_dir}/idswap.json")["results"] if r["source"] == "test"]
    n = sum(r["n"] for r in rows)
    if n == 0:
        return None
    return {"n": n, "flip_rate": sum(r["n_flips"] for r in rows) / n}


def _trade_row(label: str, node: dict[str, Any], delta: bool = True) -> dict[str, Any]:
    """One v1-vs-v3 table row from a tradeoff node ({v1, v3[, delta_v3_minus_v1, ci95]})."""
    lo_hi = node.get("ci95")
    return {
        "label": label,
        "v1": node["v1"],
        "v3": node["v3"],
        "delta": node.get("delta_v3_minus_v1") if delta else None,
        "lo": lo_hi[0] if lo_hi and delta else None,
        "hi": lo_hi[1] if lo_hi and delta else None,
    }


def build_context(version: str, root: Path = Path()) -> dict[str, Any]:
    """Collect every value the template needs from results JSON (and the label descriptions)."""
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; expected one of {sorted(VERSIONS)}")
    spec = VERSIONS[version]
    peer_v = "v3" if version == "v1" else "v1"
    rd = spec["results"]

    mv = _load(root, f"{rd}/model_version.json")
    ta = _load(root, f"{rd}/track_a.json")
    ood = _load(root, f"{rd}/ood_shipped.json")
    fit = _load(root, f"{rd}/baselines_fit.json")
    eda = _load(root, "results/eda.json")
    trade = _load(root, "results/tradeoff_v1_v3.json")
    hb = _load(root, "results/trackb/headline.json")
    confirm = _load(root, "results/phase4e/confirm_a1a3.json")
    d1 = _load(root, "results/phase6a/diag/d1_oracle.json")
    links = _load(root, "report/links.json")
    thresholds = {
        v: _load(root, f"{VERSIONS[v]['results']}/ood_shipped.json")["threshold"] for v in VERSIONS
    }
    llm_cfg = yaml.safe_load((root / "configs/llm_baseline.yaml").read_text(encoding="utf-8"))

    if mv["model_fingerprint"] != ta["model_fingerprint"]:
        raise ValueError(f"{rd}: model_version and track_a fingerprints differ")
    if mv["version"] != version:
        raise ValueError(f"{rd}/model_version.json is {mv['version']!r}, expected {version!r}")
    tc = mv["train_config"]
    test = copy.deepcopy(ta["test"])
    n = int(test["n"])
    cal = ta["calibration"]
    lang = eda["language"]
    counts = eda["class_counts"]
    swap = trade["robustness_cv"][TRADEOFF_SWAP_KEY]
    transl = trade["robustness_cv"][TRADEOFF_TRANSLATION_KEY]
    holdout = trade["trackb_holdout"]
    tbv = {
        split: {metric: node[version] for metric, node in holdout[split].items()}
        for split in ("headline", "confirm")
    }
    seed_m = re.search(r"seed (\d+)", confirm["label"])
    if seed_m is None:
        raise ValueError("cannot read the CONFIRM seed from confirm_a1a3.json label")
    oracle = d1["holdouts"]["headline"]["feature_sets"]["v1_finetuned"]["unseen_known_only"]
    unsup = d1["holdouts"]["headline"]["unsupervised_finetuned_mahalanobis"]
    tt = trade["track_a_test"]
    ccv = holdout["confirm"]
    hdl = holdout["headline"]
    rows = [
        _trade_row(
            f"test macro-F1 (n = {n})", {"v1": tt["v1"]["macro_f1"], "v3": tt["v3"]["macro_f1"]}
        ),
        _trade_row("headline AUROC", hdl["auroc"]),
        _trade_row("headline strict rejection @95% retention", hdl["strict_recall_95"]),
        _trade_row("CONFIRM strict rejection @90% retention", ccv["strict_recall_90"]),
        _trade_row("neutral-swap flip rate (CV, lower is better)", swap),
        _trade_row("translation agreement (CV, higher is better)", transl),
    ]
    return {
        "repo_id": REPO_ID,
        "thresholds": thresholds,
        "m": {
            "version": version,
            "shipped": spec["shipped"],
            "branch": spec["branch"],
            "factors": list(mv.get("factors", [])),
            "fingerprint": mv["model_fingerprint"],
            "base_model": tc["model_name"],
            "temperature": cal["temperature"],
            "ood_threshold": ood["threshold"],
            "retention": ood["retention_target"],
            "query_prefix": tc["query_prefix"],
            "max_length": tc["max_len"],
            "model_seed": tc["model_seed"],
            "n_train": fit["B0"]["n_train"],
            "n_val": ood["n_val"],
            "test": test,
            "oof": ta["cv"]["oof_probability_averaged_426_rows"],
            "oof_n": _load(root, f"{rd}/slices.json")["sources"]["oof"]["all"]["n"],
            "test_idswap": _idswap_test(root, spec["robustness"]),
        },
        "peer": {"version": peer_v, "branch": VERSIONS[peer_v]["branch"]},
        "labels": [
            (c["index"], c["label"]) for c in sorted(test["classes"], key=lambda c: c["index"])
        ],
        "eda": {
            "n_rows": eda["n_rows"],
            "n_classes": eda["n_classes"],
            "min_class": min(counts.values()),
            "max_class": max(counts.values()),
            "pct_non_english": lang["pct_non_english_primary_definition_A"],
        },
        "tb": {
            "seeds": hb["seeds"],
            "n_known": hb["n_eval_known"],
            "n_unknown": hb["n_eval_unknown"],
            "confirm_classes": len(confirm["v1"]["confirm"]["per_unit"]),
            "confirm_seed": int(seed_m.group(1)),
            "holdout": hb["holdout"],
        },
        "tbv": tbv,
        "oracle": {
            "unseen_known_auroc": oracle["auroc_mean_of_folds"],
            "unsup_auroc": unsup["phase3_results"]["mean_3_seeds"],
        },
        "trade_rows": rows,
        "swap": {"v1": swap["v1"], "v3": swap["v3"], "delta": swap["delta_v3_minus_v1"]},
        "transl": {"v1": transl["v1"], "v3": transl["v3"]},
        "llm_tags": [m_["tag"] for m_ in llm_cfg["models"]],
        "links": {"wandb": links["wandb"]},
        "repo_url": GITHUB_URL,
        "colab_url": COLAB_URL,
    }


INITIALS_POSSESSIVE = re.compile(
    r"\bG" r"G(?:'s)?(?= (?:decision|approval))"
)  # owner initials before "decision" -> "the author's decision"
INITIALS_WORD = re.compile(r"\bG" r"G\b")


def render_card(version: str, root: Path = Path(), publish: bool = False) -> str:
    """The README.md model card for `version`, rendered from results JSON.

    `publish` drops the private-repo token remark from the usage snippet. The internal owner
    shorthand inside recorded result strings is replaced at render time (the results JSON is
    left untouched).
    """
    ctx = build_context(version, root)
    ctx["publish"] = publish
    text = make_env().get_template(TEMPLATE_NAME).render(**ctx)
    return INITIALS_WORD.sub("the author", INITIALS_POSSESSIVE.sub("the author's", text))


def main(argv: list[str] | None = None) -> None:
    """CLI: write the card for one version."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", choices=sorted(VERSIONS), required=True)
    ap.add_argument("--root", type=Path, default=Path())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--publish",
        action="store_true",
        help="final card: drop the private-repo (authorised token) remark from the snippets",
    )
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render_card(a.version, a.root, a.publish), encoding="utf-8")
    print(f"[model_card] wrote {a.out}")


if __name__ == "__main__":
    main()
