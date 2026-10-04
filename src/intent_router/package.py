"""Package a trained model into a self-contained serve directory.

    python -m intent_router.package --results-dir results/final --model-dir outputs/final_model \
        --features outputs/final/features_logits.npz --out serve_model

Fits NOTHING new: temperature and the abstention threshold are read from the results dir, the
Mahalanobis bank is fitted (repo `fit_gaussian_lw`, exactly as phase4c.shipped_ood_threshold did)
from the saved TRAIN penultimate features. Needs the training stack (sklearn, pandas) because
the bank fit and the label lookup reuse repo code; run it from the training venv (read-only use)
and consume the output from `.venv-serve`. Only ids and labels of data/dataset.csv are read
(never written out). The serve dir layout:

    config.json, model.safetensors, tokenizer*   HF model + tokenizer (copied)
    router_config.json                           see predict.RouterConfig
    ood_bank.npz                                 means (K, d), precision (d, d), shrinkage
    onnx/                                        added later by `intent_router.onnx_export`
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from intent_router.predict import BANK_NAME, CONFIG_NAME, mahalanobis_score

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)
THRESHOLD_TOL = 1e-6  # recomputed val threshold must equal the shipped one to float round-off


def write_serve_dir(
    model_dir: Path,
    out: Path,
    *,
    labels: list[str],
    prefix: str,
    max_len: int,
    temperature: float,
    ood_threshold: float,
    retention: float,
    model_fingerprint: str,
    version: str,
    means: np.ndarray,
    precision: np.ndarray,
    shrinkage: float | None = None,
    note: str = "",
    ood_method: str = "maha_ft",
) -> Path:
    """Copy the HF files and write router_config.json + ood_bank.npz (no fitting, no sklearn)."""
    out.mkdir(parents=True, exist_ok=True)
    copied = [n for n in MODEL_FILES if (model_dir / n).exists()]
    if "config.json" not in copied or not any(n.endswith(".safetensors") for n in copied):
        raise FileNotFoundError(f"{model_dir}: need config.json and model.safetensors")
    for name in copied:
        shutil.copyfile(model_dir / name, out / name)
    np.savez(
        out / BANK_NAME,
        means=np.asarray(means, np.float64),
        precision=np.asarray(precision, np.float64),
        shrinkage=np.float64(np.nan if shrinkage is None else shrinkage),
    )
    cfg: dict[str, Any] = {
        "labels": labels,
        "id2label": {str(i): lab for i, lab in enumerate(labels)},
        "prefix": prefix,
        "max_len": int(max_len),
        "temperature": float(temperature),
        "ood_method": ood_method,
        "ood_threshold": float(ood_threshold),
        "retention": float(retention),
        "model_fingerprint": model_fingerprint,
        "version": version,
        "note": note,
        "feature_dim": int(np.asarray(means).shape[1]),
        "files": copied,
    }
    (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(
    results_dir: Path,
    model_dir: Path,
    features_npz: Path,
    out: Path,
    data_path: str,
    verify_fingerprint: bool,
) -> Path:
    """Read results + saved arrays, verify consistency, write the serve dir."""
    from intent_router import data as data_mod
    from intent_router.evaluate import threshold_at_retention
    from intent_router.ood import fit_gaussian_lw

    mv = _read_json(results_dir / "model_version.json")
    shipped = _read_json(results_dir / "ood_shipped.json")
    track_a = _read_json(results_dir / "track_a.json")
    temperature = float(track_a["calibration"]["temperature"])
    mcfg = _read_json(model_dir / "config.json")
    labels = [mcfg["id2label"][str(i)] for i in range(len(mcfg["id2label"]))]
    prefix = mcfg.get("query_prefix", mv.get("train_config", {}).get("query_prefix"))
    max_len = mcfg.get("max_length", mv.get("train_config", {}).get("max_len"))
    if prefix is None or max_len is None:
        raise ValueError("query_prefix / max_length not found in model config or model_version")
    if shipped["method"] not in ("maha_ft",):
        raise ValueError(f"shipped OOD method {shipped['method']!r} is not servable (maha_ft only)")

    if verify_fingerprint:
        import torch
        from transformers import AutoModelForSequenceClassification

        from intent_router.models import state_dict_sha256

        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float32
        )
        fp = state_dict_sha256(model)
        if fp != mv["model_fingerprint"]:
            raise ValueError(f"model fingerprint {fp} != model_version {mv['model_fingerprint']}")

    z = np.load(features_npz)
    full = data_mod.load_data(data_path).set_index("id")["label"]
    if list(data_mod.LABELS) != labels:
        raise ValueError("dataset label list differs from the model's id2label")
    l2i = {lab: i for i, lab in enumerate(labels)}
    y_tr = np.array([l2i[full[i]] for i in z["train_ids"]])
    g = fit_gaussian_lw(z["train_features"], y_tr, len(labels))

    # The bank must reproduce the shipped threshold from the saved VAL features.
    val_scores = mahalanobis_score(z["val_features"], g.means, g.precision)
    thr = threshold_at_retention(val_scores, float(shipped["retention_target"]))
    if abs(thr - float(shipped["threshold"])) > THRESHOLD_TOL:
        raise ValueError(f"recomputed threshold {thr} != shipped {shipped['threshold']}")

    write_serve_dir(
        model_dir,
        out,
        labels=labels,
        prefix=prefix,
        max_len=int(max_len),
        temperature=temperature,
        ood_threshold=float(shipped["threshold"]),
        retention=float(shipped["retention_target"]),
        model_fingerprint=mv["model_fingerprint"],
        version=str(mv["version"]),
        means=g.means,
        precision=g.precision,
        shrinkage=g.shrinkage,
        note=str(mv.get("note", "")),
        ood_method=shipped["method"],
    )
    print(f"[package] {out} version={mv['version']} fingerprint={mv['model_fingerprint'][:12]}")
    print(f"[package] T={temperature:.6f} threshold={shipped['threshold']:.6f} (recomputed ok)")
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True, help="features_logits.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", default="data/dataset.csv", help="only ids + labels are read")
    ap.add_argument(
        "--no-verify-fingerprint",
        action="store_true",
        help="skip loading the model to check its state-dict sha256 against model_version.json",
    )
    a = ap.parse_args(argv)
    build(a.results_dir, a.model_dir, a.features, a.out, a.data, not a.no_verify_fingerprint)


if __name__ == "__main__":
    main()
