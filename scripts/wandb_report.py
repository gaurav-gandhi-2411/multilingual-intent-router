"""Create / update the private W&B Report "Submission summary" in `multilingual-intent-router`.

Dev/publish tool (needs `wandb-workspaces`, see requirements-dev.txt). Sections follow the
assessment's W&B bullets in order (loss + LR curves, val macro-F1 + accuracy, per-class test
metrics + confusion matrix, run config), then Track B, then the supporting runs. All numbers in
the text are rendered from saved results files; panels read the logged runs. Idempotent: the
report URL is kept in report/links.json and an existing report is updated in place. The
project's visibility is only READ elsewhere (never changed here).

Usage:
    python scripts/wandb_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENTITY = "gauravgandhi429-gaurav-gandhi"
PROJECT = "multilingual-intent-router"
TITLE = "Submission summary"
LINKS = Path("report/links.json")
LINK_KEYS = ("wandb", "wandb_run", "wandb_project", "hf", "repo", "colab")
TRAIN_RUN = "final-v1-train"
TRAIN_NAME = "final-v1"  # display name of the live training run (report filters use it)


# ------------------------------------------------------------------ pure helpers
def project_url() -> str:
    """Project page URL."""
    return f"https://wandb.ai/{ENTITY}/{PROJECT}"


def run_url(run_id: str) -> str:
    """Run page URL for a deterministic run id."""
    return f"{project_url()}/runs/{run_id}"


def fmt_ci(m: dict[str, float]) -> str:
    """'0.9465 [0.8651, 0.9892]' from a {point, lo, hi} bootstrap record."""
    return f"{m['point']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}]"


def merge_links(existing: dict[str, Any] | None, updates: dict[str, Any]) -> dict[str, Any]:
    """links.json content: all LINK_KEYS present (null placeholders), existing values kept."""
    out: dict[str, Any] = {k: None for k in LINK_KEYS}
    out.update(existing or {})
    out.update(updates)
    return out


def config_markdown(train_cfg: dict[str, Any], split_sha: str, fingerprint: str) -> str:
    """Markdown bullet list of the run config (what the dashboard's config panel also shows)."""
    sched = (
        f"linear warmup ({train_cfg['warmup_ratio']:.0%}) then linear decay over "
        f"{train_cfg['epochs']} epochs, stopped after epoch {train_cfg['stop_epoch']}"
    )
    items = [
        f"model: `{train_cfg['model_name']}` (query prefix `{train_cfg['query_prefix']}`)",
        f"batch size {train_cfg['batch_size']}, peak LR {train_cfg['lr']}, "
        f"weight decay {train_cfg['weight_decay']}, max length {train_cfg['max_len']}",
        f"schedule: {sched}",
        f"precision {train_cfg['precision']} (autocast), "
        f"attention `{train_cfg['attn_implementation']}`, seed {train_cfg['model_seed']}",
        f"split file sha256 `{split_sha}`",
        f"trained-model fingerprint (sha256 of the state dict) `{fingerprint}`",
    ]
    return "\n".join(f"- {i}" for i in items)


# ------------------------------------------------------------------------------ main
def build_blocks(wr: Any, data: dict[str, Any]) -> list[Any]:
    """The report blocks, in assessment order then Track B."""

    def grid(run_filter: str, name: str, panels: list[Any]) -> Any:
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=run_filter)
        return wr.PanelGrid(runsets=[rs], panels=panels)

    train = lambda panels: grid(f"Name = '{TRAIN_NAME}'", "final-v1 (live training run)", panels)  # noqa: E731
    ev = lambda panels: grid("Name = 'final-v1-eval-saved'", "saved single evaluation", panels)  # noqa: E731
    tb = lambda panels: grid("Name = 'trackb-headline'", "Track B headline", panels)  # noqa: E731
    ta, cfg = data["track_a"], data["train_config"]
    h = data["headline"]["methods"]
    return [
        wr.TableOfContents(),
        wr.H1(text="Overview"),
        wr.MarkdownBlock(
            text=(
                f"Project: [{PROJECT}]({project_url()}). Final model: multilingual-e5-base "
                f"fine-tuned for 12 intents, 9 epochs. Fingerprint "
                f"`{data['fingerprint'][:16]}...`.\n\n"
                f"- Live training run: [{TRAIN_RUN}]({run_url(TRAIN_RUN)}). It re-trains the "
                "shipped configuration with live logging; the retrained weights were verified "
                "bit-identical to the shipped model (state-dict sha256 match, see "
                "`fingerprint_match` in the run summary). No test inference happens in it.\n"
                f"- Test metrics: [final-v1-eval-saved]({run_url('final-v1-eval-saved')}). They "
                "come from the single saved test evaluation of the shipped model and are NOT "
                "recomputed.\n"
                f"- Supporting runs: [bakeoff-summary]({run_url('bakeoff-summary')}), "
                f"[trackb-headline]({run_url('trackb-headline')}), "
                f"[tradeoff-v1-v3]({run_url('tradeoff-v1-v3')}), "
                f"[llm-baseline]({run_url('llm-baseline')})."
            )
        ),
        wr.H1(text="1. Training and validation loss, learning rate"),
        wr.MarkdownBlock(
            text="Per-step training loss and LR, per-epoch train and validation loss "
            "(live-logged, steps on the x axis; 22 steps per epoch)."
        ),
        train(
            [
                wr.LinePlot(title="Train loss per step", x="Step", y=["train/step_loss"]),
                wr.LinePlot(title="Learning rate per step", x="Step", y=["lr"]),
                wr.LinePlot(
                    title="Train / validation loss per epoch",
                    x="Step",
                    y=["train/loss", "eval/loss"],
                ),
            ]
        ),
        wr.H1(text="2. Validation macro-F1 and accuracy per epoch"),
        wr.MarkdownBlock(
            text="Macro-F1 is the selection metric. Validation split only (the held-out test "
            "split is never touched during training)."
        ),
        train(
            [
                wr.LinePlot(title="Validation macro-F1", x="Step", y=["eval/macro_f1"]),
                wr.LinePlot(title="Validation accuracy", x="Step", y=["eval/accuracy"]),
            ]
        ),
        wr.H1(text="3. Held-out test: per-class metrics and confusion matrix"),
        wr.MarkdownBlock(
            text=(
                f"Saved single evaluation (n={ta['test']['n']}), not recomputed. Macro-F1 "
                f"{fmt_ci(ta['test']['macro_f1'])}, accuracy {fmt_ci(ta['test']['accuracy'])} "
                "(95% bootstrap CIs). Calibration: temperature "
                f"{ta['calibration']['temperature']:.4f}, test ECE "
                f"{ta['calibration']['test']['ece_before']:.4f} before and "
                f"{ta['calibration']['test']['ece_after']:.4f} after scaling. The interactive "
                "confusion matrix is on the run page."
            )
        ),
        ev(
            [
                wr.WeavePanelSummaryTable(table_name="test/per_class"),
                wr.MediaBrowser(
                    title="Confusion matrices and reliability",
                    media_keys=[
                        "test/confusion_counts_png",
                        "test/confusion_norm_png",
                        "test/reliability_png",
                    ],
                ),
                wr.WeavePanelSummaryTable(table_name="test/calibration"),
                wr.WeavePanelSummaryTable(table_name="test/misclassifications"),
            ]
        ),
        wr.H1(text="4. Run config and hyper-parameters"),
        wr.MarkdownBlock(text=config_markdown(cfg, data["split_sha"], data["fingerprint"])),
        train([wr.RunComparer(diff_only=None)]),
        wr.H1(text="5. Track B: unseen-intent rejection"),
        wr.MarkdownBlock(
            text=(
                "Headline holdout (2 classes held out, 3 model seeds). Strict rejection recall "
                "at 95% known retention and AUROC, mean over seeds: shipped Mahalanobis on the "
                f"fine-tuned features AUROC {h['maha_ft']['auroc']['mean']:.3f}, rejection "
                f"{h['maha_ft']['strict_rejection_recall']['mean']:.3f}; max-softmax baseline "
                f"AUROC {h['msp']['auroc']['mean']:.3f}, rejection "
                f"{h['msp']['strict_rejection_recall']['mean']:.3f}."
            )
        ),
        tb(
            [
                wr.WeavePanelSummaryTable(table_name="trackb/methods"),
                wr.WeavePanelSummaryTable(table_name="trackb/operating_points"),
                wr.MediaBrowser(
                    title="ROC and score histograms",
                    media_keys=[
                        "trackb/roc",
                        "trackb/hist_maha_ft",
                        "trackb/hist_msp",
                        "trackb/saved_risk_coverage",
                    ],
                ),
            ]
        ),
        wr.H1(text="6. Supporting runs"),
        wr.MarkdownBlock(
            text="Bake-off aggregates and selection (`bakeoff-summary`), v1 vs v3 tradeoff "
            "(`tradeoff-v1-v3`) and the zero/few-shot LLM baseline (`llm-baseline`) are "
            "separate runs in this project; open them from the Overview links."
        ),
    ]


def main() -> None:
    """Build or update the report and write report/links.json."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    import wandb_workspaces.reports.v2 as wr

    mv = json.loads(Path("results/final/model_version.json").read_text(encoding="utf-8"))
    data = {
        "track_a": json.loads(Path("results/final/track_a.json").read_text(encoding="utf-8")),
        "headline": json.loads(Path("results/trackb/headline.json").read_text(encoding="utf-8")),
        "train_config": mv["train_config"],
        "fingerprint": mv["model_fingerprint"],
        "split_sha": __import__("hashlib")
        .sha256(Path("splits/splits.csv").read_bytes())
        .hexdigest(),
    }
    blocks = build_blocks(wr, data)
    existing = json.loads(LINKS.read_text(encoding="utf-8")) if LINKS.exists() else {}
    if existing.get("wandb") and "/reports/" in existing["wandb"]:
        report = wr.Report.from_url(existing["wandb"])  # update in place: no duplicate report
        report.title, report.blocks = TITLE, blocks
    else:
        report = wr.Report(
            project=PROJECT, entity=ENTITY, title=TITLE, blocks=blocks, width="readable",
            description="Training curves, saved test evaluation, config and Track B.",
        )  # fmt: skip
    report.save(draft=False)
    # wandb-workspaces builds the URL with os.path.join: backslashes on Windows.
    url = report.url.replace("\\", "/")
    print("report url:", url)
    links = merge_links(
        existing,
        {"wandb": url, "wandb_run": run_url(TRAIN_RUN), "wandb_project": project_url()},
    )
    LINKS.parent.mkdir(parents=True, exist_ok=True)
    LINKS.write_text(json.dumps(links, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
