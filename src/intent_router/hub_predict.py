"""Standalone inference for the Hub model repo. Shipped as `predict.py`.

Needs only torch + transformers + safetensors + numpy (+ huggingface_hub, a transformers
dependency, when given a repo id). Deliberately imports nothing from `intent_router`: the Hub
snapshot must work on its own. Same maths as `intent_router.predict` (the serving reference).

    from predict import IntentRouter
    router = IntentRouter.from_pretrained("path/to/snapshot")      # or "<user>/<repo>", revision=
    router.predict("some message")

Output of predict(text): label (argmax class, also returned when abstaining), confidence
(temperature-scaled max softmax prob), ood_score (MINUS the smallest class Mahalanobis distance of
the classification head's penultimate feature; higher = more in-distribution), abstained
(ood_score < ood_threshold), top3 ([{label, prob}], temperature-scaled). Also returned: `probs`
(full temperature-scaled vector) and `probs_uncalibrated` (plain softmax, T = 1).

Everything fitted at training time is read from the snapshot: `config.json` carries
id2label, temperature, ood_method, ood_threshold, query_prefix, max_length; the Mahalanobis bank is
`ood_bank.safetensors` (arrays `means` [K, d], `precision` [d, d]).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BANK_FILE = "ood_bank.safetensors"
SUPPORTED_OOD_METHODS = ("maha_ft",)
TOP_K = 3


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row softmax of logits / temperature in float64."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mahalanobis_score(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """-sqrt(min_k d2[n, k]) with d2 = x P x - 2 x P m_k + m_k P m_k, float64 (higher = known)."""
    x = np.asarray(features, dtype=np.float64)
    xp = x @ precision
    d2 = (
        np.einsum("nd,nd->n", xp, x)[:, None]
        - 2.0 * xp @ means.T
        + np.einsum("kd,kd->k", means @ precision, means)[None, :]
    )
    return -np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def resolve_dir(path_or_repo: str | Path, revision: str | None = None) -> Path:
    """A local directory as-is, else download the Hub repo snapshot (optionally at a revision)."""
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision))


class IntentRouter:
    """Fine-tuned classifier + temperature + Mahalanobis abstention, loaded from one directory."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from safetensors.numpy import load_file
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        self.prefix = str(cfg["query_prefix"])
        self.max_length = int(cfg["max_length"])
        self.temperature = float(cfg["temperature"])
        self.ood_method = str(cfg["ood_method"])
        self.ood_threshold = float(cfg["ood_threshold"])
        if self.ood_method not in SUPPORTED_OOD_METHODS:
            raise ValueError(f"unsupported ood_method {self.ood_method!r}")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        bank = load_file(str(model_dir / BANK_FILE))
        self.means = np.asarray(bank["means"], np.float64)
        self.precision = np.asarray(bank["precision"], np.float64)
        if self.means.shape[0] != len(self.labels):
            raise ValueError("ood bank class count does not match id2label")

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # eager attention = the setting the model was trained and evaluated with.
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), attn_implementation="eager"
            )
            .float()
            .eval()
        )
        self._captured: list[Any] = []
        # penultimate feature = the input of classifier.out_proj (dense + tanh of the XLM-R head)
        self.model.classifier.out_proj.register_forward_pre_hook(
            lambda _m, args: self._captured.append(args[0])
        )

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, revision: str | None = None) -> IntentRouter:
        """Load from a local snapshot directory or a Hub repo id (+ optional revision)."""
        return cls(resolve_dir(path_or_repo, revision))

    def raw(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(logits [n, K], penultimate features [n, d]) as float32 numpy, one padded batch."""
        enc = self.tokenizer(
            [self.prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self._captured.clear()
        with self._torch.inference_mode():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            logits = logits.logits
        feats = self._captured[-1]
        self._captured.clear()
        return logits.float().numpy(), feats.float().numpy()

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predictions for many texts in one forward pass."""
        if not texts:
            return []
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a list of str")
        logits, feats = self.raw(texts)
        probs = softmax(logits, self.temperature)
        probs_raw = softmax(logits, 1.0)
        ood = mahalanobis_score(feats, self.means, self.precision)
        order = np.argsort(-probs, axis=1, kind="stable")[:, : min(TOP_K, probs.shape[1])]
        out: list[dict[str, Any]] = []
        for i in range(len(probs)):
            top = [{"label": self.labels[j], "prob": float(probs[i, j])} for j in order[i]]
            out.append(
                {
                    "label": self.labels[int(order[i, 0])],
                    "confidence": top[0]["prob"],
                    "ood_score": float(ood[i]),
                    "abstained": bool(ood[i] < self.ood_threshold),
                    "top3": top,
                    "probs": [float(p) for p in probs[i]],
                    "probs_uncalibrated": [float(p) for p in probs_raw[i]],
                }
            )
        return out

    def predict(self, text: str) -> dict[str, Any]:
        """Prediction for one text."""
        return self.predict_batch([text])[0]


def main(argv: list[str]) -> None:
    """CLI: python predict.py <snapshot dir or repo id> "text" ["text" ...]"""
    if len(argv) < 2:
        raise SystemExit('usage: python predict.py <snapshot dir | repo id> "text" ["text" ...]')
    router = IntentRouter.from_pretrained(argv[0])
    for text, res in zip(argv[1:], router.predict_batch(argv[1:]), strict=True):
        res = {k: v for k, v in res.items() if not k.startswith("probs")}
        print(json.dumps({"text": text, **res}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
