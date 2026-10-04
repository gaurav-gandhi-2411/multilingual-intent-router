"""Lean inference for the intent router.

Serving imports only numpy + (torch + transformers | onnxruntime + transformers' tokenizer):
no sklearn / pandas / datasets / scipy. Everything fitted at training time (temperature,
Mahalanobis bank, abstention threshold) is read from a *serve directory* written by
`intent_router.package`; nothing here is hardwired to one model version.

Output semantics of `predict(text)` (one dict per text):
  label      argmax class name of the logits (also returned when the router abstains).
  confidence temperature-scaled maximum softmax probability, softmax(logits / T).max().
  ood_score  fine-tuned-feature Mahalanobis score: MINUS the smallest class Mahalanobis
             distance of the penultimate feature (the classification head's dense+tanh
             output, the input of classifier.out_proj); HIGHER = more in-distribution.
             Identical to `ood.score_mahalanobis` (the training-side reference).
  abstained  ood_score < threshold (threshold = val-derived 95%-retention cut, in the config).
  top3       [{label, prob}] of the three most probable classes (temperature-scaled
             probabilities, descending); confidence == top3[0]["prob"].
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_NAME = "router_config.json"
BANK_NAME = "ood_bank.npz"
BANK_ALT_NAME = "ood_bank.safetensors"  # Hub layout; same array names
BACKENDS = ("torch", "onnx_fp32", "onnx_int8", "onnx_int8_encoder")
ONNX_FILES = {
    "onnx_fp32": "onnx/model_fp32.onnx",
    "onnx_int8": "onnx/model_int8.onnx",  # whole-graph dynamic int8 (hurts v1: see results/serving)
    "onnx_int8_encoder": "onnx/model_int8_encoder.onnx",  # encoder Linear layers only
}
DEFAULT_BACKEND = "onnx_fp32"  # lossless vs torch (max prob diff ~1e-6), no torch needed at runtime
ONNX_OUTPUTS = ("logits", "features")  # output names of the exported graph (onnx_export.py)
SUPPORTED_OOD_METHODS = ("maha_ft",)
TOP_K = 3


# ------------------------------------------------------------------------ config + math
@dataclass(frozen=True)
class RouterConfig:
    """Contents of router_config.json (written by package.py)."""

    labels: list[str]  # index = class id
    prefix: str
    max_len: int
    temperature: float
    ood_method: str
    ood_threshold: float
    retention: float
    model_fingerprint: str
    version: str
    note: str = ""

    @classmethod
    def from_file(cls, path: Path) -> RouterConfig:
        """Read and validate router_config.json."""
        d = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls(
            labels=list(d["labels"]),
            prefix=str(d["prefix"]),
            max_len=int(d["max_len"]),
            temperature=float(d["temperature"]),
            ood_method=str(d["ood_method"]),
            ood_threshold=float(d["ood_threshold"]),
            retention=float(d["retention"]),
            model_fingerprint=str(d["model_fingerprint"]),
            version=str(d["version"]),
            note=str(d.get("note", "")),
        )
        if cfg.ood_method not in SUPPORTED_OOD_METHODS:
            raise ValueError(f"unsupported ood_method {cfg.ood_method!r}: {SUPPORTED_OOD_METHODS}")
        if cfg.temperature <= 0 or not cfg.labels:
            raise ValueError("temperature must be > 0 and labels non-empty")
        return cfg


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row softmax of logits / temperature in float64 (same maths as evaluate.softmax)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class MahalanobisBank:
    """Class means (K, d) and the shared Ledoit-Wolf precision (d, d), float64."""

    means: np.ndarray
    precision: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> MahalanobisBank:
        """Load ood_bank.npz or ood_bank.safetensors (arrays `means`, `precision`)."""
        if Path(path).suffix == ".safetensors":
            from safetensors.numpy import load_file

            z = load_file(str(path))
        else:
            z = np.load(path)
        return cls(np.asarray(z["means"], np.float64), np.asarray(z["precision"], np.float64))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Minus the smallest class Mahalanobis distance (higher = known); float64 [n]."""
        return mahalanobis_score(features, self.means, self.precision)


def mahalanobis_score(features: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Lean re-implementation of `ood.score_mahalanobis` (same expansion, same float64 maths).

    d2[n, k] = x P x - 2 x P m_k + m_k P m_k; the score is -sqrt(max(min_k d2, 0)). The
    expanded form (not a per-class loop) is kept on purpose so results match the reference
    to float64 round-off, which tests/test_predict_equivalence.py asserts.
    """
    x = np.asarray(features, dtype=np.float64)
    xp = x @ precision
    d2 = (
        np.einsum("nd,nd->n", xp, x)[:, None]
        - 2.0 * xp @ means.T
        + np.einsum("kd,kd->k", means @ precision, means)[None, :]
    )
    return -np.sqrt(np.maximum(d2.min(axis=1), 0.0))


# ---------------------------------------------------------------------------- backends
class TorchBackend:
    """fp32 eager PyTorch forward; features captured at classifier.out_proj's input."""

    def __init__(self, model_dir: Path, threads: int | None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        if threads:
            torch.set_num_threads(threads)
        # eager attention = the setting the model was trained/evaluated with (bitwise-stable).
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), dtype=torch.float32, attn_implementation="eager"
        ).eval()
        out_proj = getattr(getattr(self.model, "classifier", None), "out_proj", None)
        if out_proj is None:
            raise ValueError("model has no classifier.out_proj (XLM-R style head required)")
        self._captured: list[Any] = []
        out_proj.register_forward_pre_hook(lambda _m, args: self._captured.append(args[0]))
        self._torch = torch

    def run(self, enc: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(logits [n, K], features [n, d]) as float32 numpy."""
        torch = self._torch
        batch = {k: torch.from_numpy(v) for k, v in enc.items()}
        self._captured.clear()
        with torch.inference_mode():
            logits = self.model(**batch).logits
        feats = self._captured[-1]
        self._captured.clear()
        return logits.float().numpy(), feats.float().numpy()


class OnnxBackend:
    """onnxruntime CPU session over a graph that returns (logits, features)."""

    def __init__(self, onnx_path: Path, threads: int | None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1  # one request at a time (the router serialises calls)
        self.session = ort.InferenceSession(
            str(onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def run(self, enc: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(logits [n, K], features [n, d]) as float32 numpy."""
        feed = {n: np.ascontiguousarray(enc[n], dtype=np.int64) for n in self.input_names}
        logits, feats = self.session.run(list(ONNX_OUTPUTS), feed)
        return np.asarray(logits, np.float32), np.asarray(feats, np.float32)


# ------------------------------------------------------------------------------ router
class Router:
    """Tokenizer + backend + calibration + OOD bank; shared pre/post-processing."""

    def __init__(
        self,
        cfg: RouterConfig,
        bank: MahalanobisBank,
        tokenizer: Any,
        backend: TorchBackend | OnnxBackend,
        backend_name: str,
    ) -> None:
        self.cfg = cfg
        self.bank = bank
        self.tokenizer = tokenizer
        self.backend = backend
        self.backend_name = backend_name
        # Serialise inference: concurrent requests would only fight over the same CPU threads.
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, model_dir: str | Path, backend: str = "torch", threads: int | None = None
    ) -> Router:
        """Load a serve directory (see package.py) with the chosen backend."""
        from transformers import AutoTokenizer

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        root = Path(model_dir)
        cfg = RouterConfig.from_file(root / CONFIG_NAME)
        bank_path = root / BANK_NAME
        if not bank_path.exists() and (root / BANK_ALT_NAME).exists():
            bank_path = root / BANK_ALT_NAME
        bank = MahalanobisBank.from_file(bank_path)
        tok = AutoTokenizer.from_pretrained(str(root))
        be: TorchBackend | OnnxBackend
        if backend == "torch":
            be = TorchBackend(root, threads)
        else:
            path = root / ONNX_FILES[backend]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing: run `python -m intent_router.onnx_export`"
                )
            be = OnnxBackend(path, threads)
        if bank.means.shape[0] != len(cfg.labels):
            raise ValueError("ood bank class count does not match labels")
        return cls(cfg, bank, tok, be, backend)

    def encode(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Prefix + tokenise (pad to the longest, truncate to max_len) as int64 numpy."""
        enc = self.tokenizer(
            [self.cfg.prefix + t for t in texts],
            padding=True,
            truncation=True,
            max_length=self.cfg.max_len,
            return_tensors="np",
        )
        return {k: np.asarray(enc[k], dtype=np.int64) for k in ("input_ids", "attention_mask")}

    def infer(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Raw (logits, features) for texts, in order; thread-safe."""
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"text must be str, got {type(t).__name__}")
        enc = self.encode(texts)
        with self._lock:
            return self.backend.run(enc)

    def ood_scores(self, features: np.ndarray) -> np.ndarray:
        """Mahalanobis scores (higher = in-distribution) for penultimate features."""
        return self.bank.score(features)

    def postprocess(self, logits: np.ndarray, features: np.ndarray) -> list[dict[str, Any]]:
        """Turn raw backend outputs into the public prediction dicts (see module docstring)."""
        cfg = self.cfg
        probs = softmax(logits, cfg.temperature)
        ood = self.ood_scores(features)
        k = min(TOP_K, probs.shape[1])
        order = np.argsort(-probs, axis=1, kind="stable")[:, :k]
        out: list[dict[str, Any]] = []
        for i in range(len(probs)):
            top = [{"label": cfg.labels[j], "prob": float(probs[i, j])} for j in order[i]]
            out.append(
                {
                    "label": cfg.labels[int(order[i, 0])],
                    "confidence": top[0]["prob"],
                    "ood_score": float(ood[i]),
                    "abstained": bool(ood[i] < cfg.ood_threshold),
                    "top3": top,
                }
            )
        return out

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predictions for many texts in one forward pass (padded to the longest)."""
        if not texts:
            return []
        logits, feats = self.infer(texts)
        return self.postprocess(logits, feats)

    def predict(self, text: str) -> dict[str, Any]:
        """Prediction for a single text."""
        return self.predict_batch([text])[0]


# ------------------------------------------------------------- module-level convenience
_DEFAULT: Router | None = None
_DEFAULT_LOCK = threading.Lock()


def env_settings() -> tuple[str, str, int | None]:
    """(model_dir, backend, threads) from ROUTER_MODEL_DIR / ROUTER_BACKEND / ROUTER_THREADS."""
    model_dir = os.environ.get("ROUTER_MODEL_DIR", "serve_model")
    backend = os.environ.get("ROUTER_BACKEND", DEFAULT_BACKEND)
    raw = os.environ.get("ROUTER_THREADS", "").strip()
    return model_dir, backend, int(raw) if raw else None


def get_router() -> Router:
    """Process-wide router, loaded once from the environment settings."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            model_dir, backend, threads = env_settings()
            _DEFAULT = Router.load(model_dir, backend, threads)
        return _DEFAULT


def predict(text: str) -> dict[str, Any]:
    """predict(text) -> {label, confidence, ood_score, abstained, top3} via the default router."""
    return get_router().predict(text)


def predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Batch variant of predict()."""
    return get_router().predict_batch(texts)
