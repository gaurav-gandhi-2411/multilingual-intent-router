from __future__ import annotations

import re
from typing import Any

import numpy as np

# Short names used in run ids / W&B run names.
MODEL_SHORT: dict[str, str] = {
    "intfloat/multilingual-e5-base": "e5",
    "FacebookAI/xlm-roberta-base": "xlmr",
    "microsoft/mdeberta-v3-base": "mdeberta",
    "intfloat/multilingual-e5-large": "e5large",
}

# Applied identically at train and inference time via prepare_text().
QUERY_PREFIX: dict[str, str] = {
    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
}

_LOAD_RE = re.compile(r"LD-\d+")
_PO_RE = re.compile(r"PO-\d+")
_REF_RE = re.compile(r"\b[A-Z]{2,4}-\d+")


def model_short(model_name: str) -> str:
    """Short key for a model name (falls back to the last path component)."""
    return MODEL_SHORT.get(model_name, model_name.split("/")[-1])


def mask_entities(text: str) -> str:
    """Replace identifiers: LD-n -> [LOAD_ID], PO-n -> [PO_ID], other AB-n -> [REF_ID]."""
    text = _LOAD_RE.sub("[LOAD_ID]", text)
    text = _PO_RE.sub("[PO_ID]", text)
    return _REF_RE.sub("[REF_ID]", text)


def prepare_text(text: str, model_name: str, mask: bool = False) -> str:
    """Single source of truth for model input text (prefix + optional entity masking)."""
    if mask:
        text = mask_entities(text)
    return QUERY_PREFIX.get(model_name, "") + text


def build_model(
    model_name: str, labels: list[str], attn_implementation: str | None = None
) -> tuple[Any, Any]:
    """Return (tokenizer, AutoModelForSequenceClassification) with a len(labels)-way head.

    attn_implementation None keeps the library default (what the CV bake-off used); "eager" or
    "sdpa" is forwarded to from_pretrained.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label = dict(enumerate(labels))
    label2id = {lab: i for i, lab in id2label.items()}
    extra: dict[str, Any] = {}
    if attn_implementation is not None:
        extra["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        # Master weights must be fp32: recent transformers honours the checkpoint dtype, and
        # mdeberta ships fp16 weights, which breaks GradScaler ("unscale FP16 gradients").
        dtype=torch.float32,
        **extra,
    )
    return tokenizer, model


def predict(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    model_name: str,
    max_len: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Eval-mode fp32 inference: (logits [n, n_labels], features [n, hidden]) as float32.

    features are the classification head's dense+tanh output, i.e. the input of
    classifier.out_proj (XLMRobertaClassificationHead; dropout is the identity in eval),
    captured with a forward pre-hook so the head itself is never re-implemented.
    Texts get the model's query prefix via prepare_text; order is preserved.
    """
    import torch

    out_proj = getattr(getattr(model, "classifier", None), "out_proj", None)
    if out_proj is None:
        raise ValueError(
            f"{type(model).__name__} has no classifier.out_proj; penultimate features are only "
            "defined for the XLM-R style head"
        )
    captured: list[Any] = []
    handle = out_proj.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                chunk = [prepare_text(t, model_name) for t in texts[s : s + batch_size]]
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                captured.clear()
                logits = model(**enc).logits
                logits_parts.append(logits.float().cpu().numpy())
                feat_parts.append(captured[-1].float().cpu().numpy())
    finally:
        handle.remove()
        model.train(was_training)
    n_labels = int(model.config.num_labels)
    hidden = int(model.config.hidden_size)
    if not texts:
        return np.zeros((0, n_labels), np.float32), np.zeros((0, hidden), np.float32)
    return (
        np.concatenate(logits_parts).astype(np.float32),
        np.concatenate(feat_parts).astype(np.float32),
    )


def predict_hidden(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    model_name: str,
    max_len: int,
    batch_size: int,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eval-mode fp32 inference: (logits, head features, pooled hidden states) as float32.

    logits [n, n_labels] and features [n, hidden] are exactly what predict() returns. pooled
    [n, len(layers), hidden] holds the attention-masked mean of hidden_states[l] for each l in
    layers (0 = embedding output, -1 = last encoder layer, -4..-1 = last four layers); padding
    positions never enter the mean. Texts get the query prefix via prepare_text; order is kept.
    """
    import torch

    out_proj = getattr(getattr(model, "classifier", None), "out_proj", None)
    if out_proj is None:
        raise ValueError(f"{type(model).__name__} has no classifier.out_proj")
    captured: list[Any] = []
    handle = out_proj.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    pool_parts: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                chunk = [prepare_text(t, model_name) for t in texts[s : s + batch_size]]
                enc = tokenizer(
                    chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
                ).to(device)
                captured.clear()
                out = model(**enc, output_hidden_states=True)
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = [(out.hidden_states[i].float() * m).sum(1) / m.sum(1) for i in layers]
                logits_parts.append(out.logits.float().cpu().numpy())
                feat_parts.append(captured[-1].float().cpu().numpy())
                pool_parts.append(torch.stack(pooled, dim=1).cpu().numpy())
    finally:
        handle.remove()
        model.train(was_training)
    n_labels, hidden = int(model.config.num_labels), int(model.config.hidden_size)
    if not texts:
        return (
            np.zeros((0, n_labels), np.float32),
            np.zeros((0, hidden), np.float32),
            np.zeros((0, len(layers), hidden), np.float32),
        )
    return (
        np.concatenate(logits_parts).astype(np.float32),
        np.concatenate(feat_parts).astype(np.float32),
        np.concatenate(pool_parts).astype(np.float32),
    )


def state_dict_sha256(model: Any) -> str:
    """sha256 over (name, dtype, shape, raw bytes) of every tensor in sorted-name order."""
    import hashlib

    import torch

    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
