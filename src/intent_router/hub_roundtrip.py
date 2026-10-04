"""Hub round trip: reproduce the saved TEST predictions from a Hub snapshot.

    python -m intent_router.hub_roundtrip --version v1 --revision main --write

Downloads the (private) repo revision into a clean temporary cache, loads it with
`AutoModelForSequenceClassification.from_pretrained`, runs the snapshot's own `predict.py` on the
test-split texts and compares with `results/<final dir>/test_predictions.csv` (labels and the
`prob_*` columns, which are the plain T = 1 softmax; the temperature is checked separately).
This is an INFERENCE reproduction check, not a new evaluation: with `--write` one
`hub_roundtrip_inference` line is appended (BEFORE inference, as for every other test-split call)
to results/final/test_eval_log.jsonl and the numbers are written to
results/hub/roundtrip_<version>.json. Dataset texts are read locally (data/ is confidential and
gitignored) and never written anywhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from intent_router.model_card import REPO_ID, VERSIONS

TEST_LOG = Path("results/final/test_eval_log.jsonl")
CALL_TYPE = "hub_roundtrip_inference"
BATCH = 16
# Saved arrays came from the training-venv fp32 eval path; CPU re-inference differs only by
# batch padding / kernel round-off. 1e-5 is the spec target; the measured max diff is reported.
PROB_TOL = 1e-5


def load_test_rows(root: Path, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(test rows with id/text/label in saved-prediction order, saved predictions)."""
    saved = pd.read_csv(root / VERSIONS[version]["results"] / "test_predictions.csv")
    data = pd.read_csv(root / "data/dataset.csv").set_index("id")
    splits = pd.read_csv(root / "splits/splits.csv").set_index("id")
    ids = saved["id"].tolist()
    if not (splits.loc[ids, "split"] == "test").all():
        raise ValueError("saved predictions include non-test ids")
    rows = data.loc[ids, ["text", "label"]].reset_index()
    return rows, saved


def load_predict_module(snapshot: Path) -> ModuleType:
    """Import the snapshot's own predict.py (not the repo's copy)."""
    spec = importlib.util.spec_from_file_location("hub_snapshot_predict", snapshot / "predict.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {snapshot / 'predict.py'}")
    sys.dont_write_bytecode = True  # never leave __pycache__ inside a snapshot / staged dir
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(
    snapshot: Path, rows: pd.DataFrame, saved: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run snapshot/predict.py on rows; return (comparison numbers, per-row predictions)."""
    from transformers import AutoModelForSequenceClassification

    # the plain AutoModel path must load too (custom config fields must not break it)
    plain = AutoModelForSequenceClassification.from_pretrained(str(snapshot))
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    n_labels = plain.config.num_labels
    del plain

    mod = load_predict_module(snapshot)
    router = mod.IntentRouter.from_pretrained(snapshot)
    texts = rows["text"].tolist()
    preds: list[dict[str, Any]] = []
    for i in range(0, len(texts), BATCH):
        preds.extend(router.predict_batch(texts[i : i + BATCH]))
    prob_cols = [f"prob_{k}" for k in range(n_labels)]
    # the saved prob_* / confidence columns are the plain (T = 1) softmax of the test logits
    got = np.array([p["probs_uncalibrated"] for p in preds], dtype=np.float64)
    want = saved[prob_cols].to_numpy(dtype=np.float64)
    diff = np.abs(got - want)
    # temperature-scaled outputs vs the saved T = 1 probabilities re-scaled: softmax(log p / T)
    temp = float(cfg["temperature"])
    z = np.log(want) / temp
    z -= z.max(axis=1, keepdims=True)
    want_t = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    got_t = np.array([p["probs"] for p in preds], dtype=np.float64)
    labels_got = [p["label"] for p in preds]
    agree = float(np.mean(np.array(labels_got) == saved["pred_label"].to_numpy()))
    return (
        {
            "n": len(rows),
            "n_labels": n_labels,
            "label_agreement": agree,
            "n_label_mismatch": int(np.sum(np.array(labels_got) != saved["pred_label"].to_numpy())),
            "max_abs_prob_diff": float(diff.max()),
            "mean_abs_prob_diff": float(diff.mean()),
            "max_abs_confidence_diff": float(
                np.abs(got.max(axis=1) - saved["confidence"].to_numpy()).max()
            ),
            "max_abs_prob_diff_temperature_scaled": float(np.abs(got_t - want_t).max()),
            "test_coverage_at_threshold": float(np.mean([not p["abstained"] for p in preds])),
            "temperature": cfg["temperature"],
            "ood_threshold": cfg["ood_threshold"],
            "model_fingerprint": cfg["model_fingerprint"],
            "probabilities": (
                "plain softmax, T = 1 (the saved prob_* convention); predict.py applies the "
                "temperature on top, checked via max_abs_prob_diff_temperature_scaled"
            ),
        },
        preds,
    )


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def append_log(path: Path, fingerprint: str, variant: str, n_rows: int) -> str:
    """Append one hub_roundtrip_inference line (same fields as the serve_bench lines)."""
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "call_type": CALL_TYPE,
        "model_fingerprint": fingerprint,
        "variant": variant,
        "n_rows": n_rows,
        "stage": "hub_roundtrip",
        "smoke": False,
    }
    with path.open("a", encoding="utf-8") as fh:  # append-only: old lines are never touched
        fh.write(json.dumps(entry) + "\n")
    return entry["timestamp"]


def download(repo_id: str, revision: str, cache_dir: Path) -> tuple[Path, str]:
    """Snapshot of repo_id@revision into a clean cache dir; returns (path, commit sha)."""
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(repo_id, revision=revision).sha
    path = snapshot_download(repo_id, revision=revision, cache_dir=str(cache_dir))
    return Path(path), str(sha)


def run_roundtrip(
    version: str,
    revision: str | None = None,
    repo_id: str = REPO_ID,
    root: Path = Path(),
    write: bool = False,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Download (or use `snapshot`), compare with the saved test predictions, optionally log."""
    revision = revision or VERSIONS[version]["branch"]
    rows, saved = load_test_rows(root, version)
    with tempfile.TemporaryDirectory(prefix="hub_roundtrip_") as tmp:
        if snapshot is None:
            snap, sha = download(repo_id, revision, Path(tmp))
        else:
            snap, sha = snapshot, "local"
        fp = json.loads((snap / "config.json").read_text(encoding="utf-8"))["model_fingerprint"]
        stamp = None
        if write:
            stamp = append_log(root / TEST_LOG, fp, f"hub_{revision}", len(rows))
        res, _ = compare(snap, rows, saved)
        files = sorted(p.name for p in snap.iterdir())
    out = {
        "version": version,
        "repo_id": repo_id,
        "revision": revision,
        "revision_sha": sha,
        **res,
        "tolerances": {"max_abs_prob_diff": PROB_TOL, "label_agreement": 1.0},
        "passes": res["label_agreement"] == 1.0 and res["max_abs_prob_diff"] <= PROB_TOL,
        "reference": f"{VERSIONS[version]['results']}/test_predictions.csv",
        "log_timestamp": stamp,
        "files_checked": files,
        "kind": "inference reproduction check on the test split (not a new evaluation)",
    }
    if write:
        dst = root / "results/hub" / f"roundtrip_{version}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", choices=sorted(VERSIONS), required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--snapshot", type=Path, default=None, help="local dir instead of the Hub")
    ap.add_argument("--write", action="store_true", help="log + write results/hub/roundtrip_*.json")
    a = ap.parse_args(argv)
    print(
        json.dumps(
            run_roundtrip(a.version, a.revision, write=a.write, snapshot=a.snapshot), indent=2
        )
    )


if __name__ == "__main__":
    main()
