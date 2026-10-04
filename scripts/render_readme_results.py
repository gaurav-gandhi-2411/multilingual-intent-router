"""Render the generated blocks of README.md from results JSON (no number is typed by hand).

    python scripts/render_readme_results.py            # rewrite the blocks in README.md
    python scripts/render_readme_results.py --check    # exit 1 if README.md is out of date

Two marked blocks are rewritten in place; everything outside the markers is hand-written prose:

    <!-- LINKS:BEGIN --> ... <!-- LINKS:END -->      badge row; read from report/links.json
                                                     (keys: wandb, hf, repo, colab, report; every
                                                     key optional) and left as plain placeholders
                                                     for absent keys or a missing file.
    <!-- DELIVERABLES:BEGIN --> ... <!-- DELIVERABLES:END -->  the "Deliverables map" table rows;
                                                     links from the same report/links.json
                                                     (`report` may be repo-relative: report.pdf).
    <!-- RESULTS:BEGIN --> ... <!-- RESULTS:END -->  Track A and Track B tables.

Sources (all committed): results/final/track_a.json, results/final/model_version.json,
results/llm_baseline/track_a.json (+ configs/llm_baseline.yaml for the model tags),
results/trackb/headline.json, results/trackb_improve/confirm.json. The shipped model is v1
(results/final); any other shipped version is an error rather than a silent fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LINKS_JSON = ROOT / "report" / "links.json"
TRACK_A = "results/final/track_a.json"
MODEL_VERSION = "results/final/model_version.json"
LLM_TRACK_A = "results/llm_baseline/track_a.json"
LLM_CONFIG = "configs/llm_baseline.yaml"
HEADLINE = "results/trackb/headline.json"
CONFIRM = "results/trackb_improve/confirm.json"
SHIPPED_VERSION = "v1"
SHIPPED_METHOD = "maha_ft"
BASELINE_METHOD = "msp"
COLAB_TAG = "v1.0-submission"
COLAB_PATH = "notebooks/intent_router_colab.ipynb"
BLOCKS = ("RESULTS", "LINKS", "DELIVERABLES")

LINK_PLACEHOLDERS = {
    "report": "report.pdf [placeholder: attached to the submission]",
    "hf": "Hugging Face model [placeholder: filled at publish time]",
    "wandb": "W&B report [placeholder: filled at publish time]",
    "colab": "Open in Colab [placeholder: filled at publish time]",
}


def _load(rel: str, root: Path) -> Any:
    """Read one JSON file under the repo root."""
    return json.loads((root / rel).read_text(encoding="utf-8"))


def f3(x: float) -> str:
    """Three decimals, the project's table convention."""
    return f"{x:.3f}"


def ci(d: dict[str, float]) -> str:
    """`point [lo, hi]` from a {point, lo, hi} dict."""
    return f"{f3(d['point'])} [{f3(d['lo'])}, {f3(d['hi'])}]"


def mean_std(d: dict[str, Any]) -> str:
    """`mean +- std` from a {mean, std, ...} dict."""
    return f"{f3(d['mean'])} ± {f3(d['std'])}"


def llm_tags(root: Path) -> dict[str, str]:
    """Map the short LLM names used as result keys to the Ollama model tags in the config."""
    import yaml

    cfg = yaml.safe_load((root / LLM_CONFIG).read_text(encoding="utf-8"))
    return {m["name"]: m["tag"] for m in cfg["models"]}


def best_llm(metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The local LLM configuration with the highest test macro-F1 (key like `llama3|zero`)."""
    key = max(metrics, key=lambda k: metrics[k]["macro_f1"]["point"])
    return key, metrics[key]


def delta_cell(d: dict[str, float], negate: bool = False) -> str:
    """`delta [lo, hi]` of (shipped model - other); `negate` flips an (other - shipped) delta."""
    delta, lo, hi = d["delta"], d["lo"], d["hi"]
    if negate:
        delta, lo, hi = -delta, -hi, -lo
    return f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def track_a_block(root: Path) -> str:
    """Markdown for the Track A tables (test split, single logged evaluation)."""
    ta = _load(TRACK_A, root)
    test = ta["test"]
    b = test["baselines"]
    llm = _load(LLM_TRACK_A, root)
    key, m = best_llm(llm["metrics"])
    name, mode = key.split("|")
    tag = llm_tags(root)[name]
    mode_txt = {"zero": "zero-shot", "five": "5-shot"}[mode]
    rows = [
        (
            f"**Fine-tuned multilingual-e5-base ({SHIPPED_VERSION}, shipped)**",
            ci(test["macro_f1"]),
            ci(test["accuracy"]),
            "n/a",
        ),
        (
            "B0: TF-IDF + logistic regression",
            ci(b["B0"]["macro_f1"]),
            ci(b["B0"]["accuracy"]),
            delta_cell(b["B0"]["delta_macro_f1_final_minus_baseline"]),
        ),
        (
            "B1: frozen e5-base embeddings + logistic regression",
            ci(b["B1"]["macro_f1"]),
            ci(b["B1"]["accuracy"]),
            delta_cell(b["B1"]["delta_macro_f1_final_minus_baseline"]),
        ),
        (
            f"Best local LLM: {tag}, {mode_txt} (of 2 models x 2 modes)",
            ci(m["macro_f1"]),
            ci(m["accuracy"]),
            delta_cell(m["delta_macro_f1_llm_minus_final"], negate=True),
        ),
    ]
    verdicts = []
    for label, row_ci in (
        ("B0", b["B0"]["delta_macro_f1_final_minus_baseline"]),
        ("B1", b["B1"]["delta_macro_f1_final_minus_baseline"]),
        (
            f"{tag} ({mode_txt})",
            {
                "lo": -m["delta_macro_f1_llm_minus_final"]["hi"],
                "hi": -m["delta_macro_f1_llm_minus_final"]["lo"],
            },
        ),
    ):
        verdicts.append(
            f"{label}: {'excludes' if row_ci['lo'] > 0 or row_ci['hi'] < 0 else 'includes'} 0"
        )
    cv = ta["cv"]
    oof = cv["oof_probability_averaged_426_rows"]
    out = [
        f"**Track A: known classes** (n = {test['n']} test rows, {test['label']}; "
        f"{test['bootstrap']['level']:.0%} CIs: {test['bootstrap']['method']}, "
        f"{test['bootstrap']['n_resamples']} resamples, seed {test['bootstrap']['seed']}).",
        "",
        "| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | Paired delta macro-F1, "
        "shipped minus row [95% CI] |",
        "|---|---|---|---|",
    ]
    out += [f"| {a} | {b_} | {c} | {d} |" for a, b_, c, d in rows]
    out += [
        "",
        "The paired 95% interval of the macro-F1 difference " + "; ".join(verdicts) + ".",
        "",
        f"Cross-validation (**{cv['label']}**: the configuration and epoch were chosen on these "
        f"folds, so treat as optimistic and not comparable to the test row): out-of-fold "
        f"macro-F1 {ci(oof['macro_f1'])}, accuracy {ci(oof['accuracy'])} "
        f"({oof['note']}); mean over {cv['n_fold_runs']} fold-runs: macro-F1 "
        f"{f3(cv['macro_f1_mean'])} ± {f3(cv['macro_f1_std'])}.",
    ]
    return "\n".join(out)


def track_b_block(root: Path) -> str:
    """Markdown for the Track B tables (headline holdout and CONFIRM leave-one-class-out)."""
    hl = _load(HEADLINE, root)
    cf = _load(CONFIRM, root)
    meth = hl["methods"]
    base, ship = meth[BASELINE_METHOD], meth[SHIPPED_METHOD]
    out = [
        f"**Track B: unknown classes** (shipped scorer `{SHIPPED_METHOD}`: Mahalanobis distance on "
        f"the fine-tuned features; threshold set for 95% known-class retention on calibration "
        f"rows; baseline `{BASELINE_METHOD}`: plain (uncalibrated) max-softmax probability).",
        "",
        f"Headline holdout, classes held out: {', '.join(f'`{c}`' for c in hl['holdout'])} "
        f"({hl['n_eval_known']} known and {hl['n_eval_unknown']} unknown evaluation rows; "
        f"mean ± sample std over {hl['n_runs']} model seeds {hl['seeds']}).",
        "",
        "| Scorer | AUROC | Strict rejection recall @95% retention | Known retention |",
        "|---|---|---|---|",
        f"| MSP baseline | {mean_std(base['auroc'])} | {mean_std(base['strict_rejection_recall'])} "
        f"| {mean_std(base['retention_known'])} |",
        f"| **`{SHIPPED_METHOD}` (shipped)** | {mean_std(ship['auroc'])} "
        f"| {mean_std(ship['strict_rejection_recall'])} | {mean_std(ship['retention_known'])} |",
        "",
    ]
    entry = cf["entries"][f"base/{SHIPPED_METHOD}"]["confirm"]
    c95 = entry["ci95"]
    msp_auroc = cf["report_only"]["confirm_mean_auroc"][f"base/{BASELINE_METHOD}"]
    out += [
        f"CONFIRM (leave-one-class-out over {len(cf['confirm_classes'])} held-out classes, one "
        f"model seed per class, row bootstrap CIs; thresholds from calibration rows; "
        f"`{CONFIRM}`):",
        "",
        "| Scorer | AUROC | Strict rejection recall @95% retention | Known retention |",
        "|---|---|---|---|",
        f"| MSP baseline | {f3(msp_auroc)} (mean only) | — | — |",
        f"| **`{SHIPPED_METHOD}` (shipped)** | {ci(c95['auroc'])} | {ci(c95['strict_recall_95'])} "
        f"| {ci(c95['retention_95'])} |",
        "",
        "— only MSP AUROC is recorded on CONFIRM.",
    ]
    return "\n".join(out)


def results_block(root: Path = ROOT) -> str:
    """The full RESULTS block body (without markers)."""
    mv = _load(MODEL_VERSION, root)
    if mv["version"] != SHIPPED_VERSION:
        raise SystemExit(
            f"{MODEL_VERSION} says version {mv['version']!r}, this renderer knows "
            f"{SHIPPED_VERSION!r}: add its sources before rendering"
        )
    src = (
        f"<sub>Rendered by `scripts/render_readme_results.py` from `{TRACK_A}`, "
        f"`{LLM_TRACK_A}`, `{HEADLINE}`, `{CONFIRM}`; do not edit by hand.</sub>"
    )
    return "\n\n".join([track_a_block(root), track_b_block(root), src])


def _badge(label: str, url: str | None, colour: str) -> str:
    """One shields.io badge linking to `url`, or a plain placeholder when there is no url."""
    text = quote(label.replace("-", "--").replace("_", "__").replace(" ", "_"), safe="_")
    if url is None:
        return f"{label}: [placeholder until publish]"
    return f"[![{label}](https://img.shields.io/badge/{text}-{colour})]({url})"


def colab_url(repo: str) -> str:
    """Open-in-Colab URL of the notebook at the submission tag, from a GitHub repo URL."""
    m = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?", repo)
    if not m:
        raise ValueError(f"not a github.com repo URL: {repo!r}")
    return f"https://colab.research.google.com/github/{m[1]}/{m[2]}/blob/{COLAB_TAG}/{COLAB_PATH}"


def links_block(root: Path = ROOT) -> str:
    """The LINKS block body: a badge row; absent links stay visible placeholders."""
    path = root / "report" / "links.json"
    links: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    colab = links.get("colab") or (colab_url(links["repo"]) if links.get("repo") else None)
    items = [
        ("report", "report.pdf", "blue", links.get("report")),
        ("hf", "Hugging Face model", "yellow", links.get("hf")),
        ("wandb", "W&B report", "orange", links.get("wandb")),
        ("colab", "Open in Colab", "green", colab),
    ]
    parts = [
        _badge(label, url, colour) if url else LINK_PLACEHOLDERS[key]
        for key, label, colour, url in items
    ]
    if links.get("repo"):
        parts.append(f"[Code]({links['repo']})")
    return " | ".join(parts)


def _link(label: str, url: str | None) -> str:
    """Markdown link, or the bare label when the url is absent (placeholder stays visible)."""
    return f"[{label}]({url})" if url else f"{label} [placeholder until publish]"


def deliverables_block(root: Path = ROOT) -> str:
    """The Deliverables map table (header + rows) with links from report/links.json."""
    path = root / "report" / "links.json"
    links: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    colab = links.get("colab") or (colab_url(links["repo"]) if links.get("repo") else None)
    rows = [
        (
            "4.1 report.pdf",
            f"{_link('report.pdf', links.get('report'))}, built by `python -m report.build` "
            "(numbers read from `results/`)",
        ),
        (
            "4.2 Hugging Face model with label mapping and model card",
            f"repo {_link('gauravgandhi2411/multilingual-intent-router', links.get('hf'))} "
            "(branch `main` = shipped v1, `robust-v3` = robustness variant); label mapping in "
            "`config.json` (`id2label`), card rendered by `intent_router.model_card` from "
            "`src/intent_router/templates/model_card.md.j2`; staged by `intent_router.hub`",
        ),
        (
            "4.3 Weights & Biases",
            f"{_link('W&B report', links.get('wandb'))}: project "
            "`multilingual-intent-router`; curated run `final-v1-train` (display name "
            "`final-v1`) by `scripts/wandb_final_rerun.py`",
        ),
        (
            "4.4 code",
            f"{_link('this repository', links.get('repo'))}: load and split (`data`, `split`), "
            "fine-tune (`train`, `final`), evaluate (`evaluate`, `final`, `trackb`), save and "
            "push (`package`, `hub`), W&B logging, seeded (42), documented (this README); "
            f"{_link('Colab notebook', colab)} reproduces the results",
        ),
    ]
    return "\n".join(
        ["| Assessment item | Where |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    )


def replace_block(text: str, name: str, body: str) -> str:
    """Replace the content between `<!-- NAME:BEGIN -->` and `<!-- NAME:END -->`."""
    pat = re.compile(rf"(<!-- {name}:BEGIN -->)\n(?:.*?\n)?(<!-- {name}:END -->)", re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"README.md has no <!-- {name}:BEGIN/END --> markers")
    return pat.sub(lambda m: f"{m[1]}\n{body}\n{m[2]}", text, count=1)


def render(text: str, root: Path = ROOT) -> str:
    """Return README text with both generated blocks regenerated."""
    text = replace_block(text, "LINKS", links_block(root))
    text = replace_block(text, "DELIVERABLES", deliverables_block(root))
    return replace_block(text, "RESULTS", results_block(root))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = ap.parse_args(argv)
    old = README.read_text(encoding="utf-8")
    new = render(old)
    if args.check:
        if new != old:
            print("README.md generated blocks are stale: run scripts/render_readme_results.py")
            return 1
        print("README.md generated blocks are up to date")
        return 0
    if new != old:
        README.write_text(new, encoding="utf-8", newline="\n")
        print("README.md updated")
    else:
        print("README.md already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
