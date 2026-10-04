"""Matplotlib figures for the report, drawn from saved results JSON only (deterministic, no RNG)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from report.data import DISPLAY_METHOD, ROOT  # noqa: E402

SEED = 42  # no stochastic element here; pinned so a future jitter/bootstrap stays reproducible
INK, GREY, GOOD, BAD, SHIP = "#1f2933", "#9aa5b1", "#2f855a", "#c53030", "#2b6cb0"
SHORT = {
    "msp": "MSP",
    "msp_temp": "MSP (T)",
    "max_logit": "max logit",
    "neg_energy": "neg. energy",
    "maha_ft": "Mahalanobis ft (shipped)",
    "knn1_ft": "kNN-1 ft",
    "knn5_ft": "kNN-5 ft",
    "maha_frozen": "Mahalanobis frozen",
    "knn1_frozen": "kNN-1 frozen",
    "knn5_frozen": "kNN-5 frozen",
}


def _j(root: Path, rel: str) -> Any:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _ops(root: Path) -> tuple[int, int]:
    """The two pre-registered retention operating points (percent), read from business.json."""
    rows = _j(root, "results/trackb_improve/business.json")["rows"]
    targets = sorted({r["calibration_retention_target"] for r in rows})
    return round(100 * targets[0]), round(100 * targets[-1])


def _style() -> None:
    np.random.seed(SEED)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def trackb_methods(root: Path, out: Path) -> Path:
    """Headline-holdout AUROC and strict rejection@95 per OOD score (mean +/- std over seeds)."""
    _style()
    h = _j(root, "results/trackb/headline.json")["methods"]
    _, op_hi = _ops(root)
    names = list(DISPLAY_METHOD)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, met, title in (
        (axs[0], "auroc", "AUROC (unknown = positive)"),
        (
            axs[1],
            "strict_rejection_recall",
            f"strict rejection recall at the {op_hi}% retention operating point",
        ),
    ):
        mean = [h[m][met]["mean"] for m in names]
        std = [h[m][met]["std"] for m in names]
        cols = [SHIP if m == "maha_ft" else GREY for m in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, mean, xerr=std, color=cols, ecolor=INK, error_kw={"lw": 0.8}, height=0.62)
        for yy, mu in zip(y, mean, strict=True):
            ax.text(mu + 0.015, yy, f"{mu:.3f}", va="center", fontsize=6.5)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[m] for m in names])
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontsize=7.5, loc="left")
    fig.tight_layout()
    return _save(fig, out / "trackb_methods.png")


def v1_v3_delta(root: Path, out: Path) -> Path:
    """Forest plot of v3 - v1 with 95% CIs (green = favours v3, red = favours v1, grey = n.s.)."""
    _style()
    t = _j(root, "results/tradeoff_v1_v3.json")
    th, rc = t["trackb_holdout"], t["robustness_cv"]
    op_lo, op_hi = _ops(root)
    rows: list[tuple[str, float, float, float, bool]] = []  # label, delta, lo, hi, higher_better

    def add(label: str, d: dict[str, Any], hb: bool) -> None:
        lo, hi = d["ci95"]
        rows.append((label, d["delta_v3_minus_v1"], lo, hi, hb))

    for part, nm in (("headline", "Track B headline"), ("confirm", "Track B CONFIRM")):
        add(f"{nm}: AUROC", th[part]["auroc"], True)
        add(f"{nm}: rejection @{op_hi}", th[part]["strict_recall_95"], True)
        add(f"{nm}: rejection @{op_lo}", th[part]["strict_recall_90"], True)
    cv = t["cv"]["v3_vs_v1_at_e_star"]
    rows.append(("CV macro-F1 (v1 at shipped epoch)", cv["delta_v3_minus_v1"], *cv["ci95"], True))
    cv2 = t["cv"]["v3_vs_v1_at_own_argmax_sensitivity"]
    rows.append(
        ("CV macro-F1 (v1 at own best epoch)", cv2["delta_v3_minus_v1"], *cv2["ci95"], True)
    )
    add(
        "neutral-swap flip rate (lower is better)",
        rc["neutral_swap_flip_rate (axis c, lower is better)"],
        False,
    )
    add("translation agreement", rc["translation_agreement (axis d, higher is better)"], True)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    y = np.arange(len(rows))[::-1]
    for yy, (_lab, d, lo, hi, hb) in zip(y, rows, strict=True):
        excl = lo > 0 or hi < 0
        favours_v3 = (d > 0) == hb
        col = GREY if not excl else (GOOD if favours_v3 else BAD)
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.6)
        ax.plot([d], [yy], "o", color=col, ms=4)
        ax.text(0.205, yy, f"{d:+.3f}", va="center", fontsize=6.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.3, 0.26)
    ax.set_xlabel(
        "v3 minus v1 (point and 95% CI); green: favours v3, red: favours v1, grey: CI includes 0"
    )
    fig.tight_layout()
    return _save(fig, out / "v1_v3_delta.png")


def loco_auroc(root: Path, out: Path) -> Path:
    """Per held-out-class AUROC for MSP vs the shipped Mahalanobis scorer (10 LOCO runs)."""
    _style()
    pc = _j(root, "results/trackb/loco.json")["per_class"]
    classes = sorted(pc)
    msp = [pc[c]["methods"]["msp"]["auroc"] for c in classes]
    mah = [pc[c]["methods"]["maha_ft"]["auroc"] for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.bar(x - 0.2, msp, 0.38, color=GREY, label="MSP")
    ax.bar(x + 0.2, mah, 0.38, color=SHIP, label="Mahalanobis, fine-tuned (shipped)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("shipment_information.", "ship.") for c in classes],
        rotation=30,
        ha="right",
        fontsize=6.5,
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout()
    return _save(fig, out / "loco_auroc.png")


def confusion_test(root: Path, out: Path) -> Path:
    """Test confusion counts (rows gold, columns predicted), drawn at print size so the class
    labels stay legible when the figure is placed at column width."""
    _style()
    a = _j(root, "results/final/track_a.json")["test"]
    cm = np.array(a["confusion_counts"])
    labels = [
        c.replace("shipment_information.", "ship.") for c in (x["label"] for x in a["classes"])
    ]
    k = len(labels)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    int(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if cm[i, j] > cm.max() / 2 else INK,
                )
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_ylabel("gold", fontsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return _save(fig, out / "confusion_test.png")


def make_all(out: Path, root: Path = ROOT) -> dict[str, Path]:
    """Draw every report figure into `out` (build/figs) and return name -> path."""
    return {
        "tb_methods": trackb_methods(root, out),
        "v1_v3": v1_v3_delta(root, out),
        "loco": loco_auroc(root, out),
        "confusion": confusion_test(root, out),  # replaces the small stored PNG in the report
    }
