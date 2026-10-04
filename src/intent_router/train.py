from __future__ import annotations

import gc
import hashlib
import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intent_router import data as data_mod
from intent_router import gpu_lock
from intent_router.models import build_model, model_short, prepare_text
from intent_router.seeding import seed_everything
from intent_router.stats import accuracy, macro_f1

PRECISIONS = ("fp16", "bf16", "fp32")
SIBLING_PREFIX = "shipment_information."


@dataclass
class TrainConfig:
    """Everything that defines one fine-tuning run (logged verbatim to W&B)."""

    model_name: str
    lr: float
    epochs: int = 20
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_len: int = 64
    precision: str = "fp16"
    model_seed: int = 0
    class_weighted: bool = False
    label_smoothing: float = 0.0
    layer_decay: float | None = None  # None => no layer-wise LR decay
    entity_masking: bool = False
    hier_aux: bool = False
    hier_lambda: float = 0.5
    max_grad_norm: float = 1.0
    # None => do not pass attn_implementation (library default, sdpa for these encoders): keeps
    # CV behaviour unchanged. "eager" is used by the final model for bitwise determinism.
    attn_implementation: str | None = None
    # Stop after this epoch while keeping the LR schedule horizon at `epochs` (None => run all).
    stop_epoch: int | None = None
    # Open-set scorer comparison C2: total loss = CE + supcon_lambda * SupCon on the L2-normalised,
    # attention-masked
    # mean-pooled last hidden state. None or 0 => off, and the forward/backward path is unchanged.
    supcon_lambda: float | None = None
    supcon_temperature: float = 0.1
    # Open-set scorer comparison C3: split each batch into this many micro-batches (gradient
    # accumulation). 1 => off.
    grad_accum: int = 1
    # Robustness and open-set training fixes A1: with this probability each ID occurrence in a
    # TRAINING text gets a random prefix
    # (re-drawn every epoch). 0 => off, and the pre-tokenised open-set evaluation batch path is
    # unchanged.
    id_randomize_p: float = 0.0
    # Robustness and open-set training fixes A2: outlier exposure. total loss = CE + oe_lambda *
    # CE(uniform) on a mini-batch of
    # oe_batch_size texts passed to train_model(oe_texts=...). 0 => off.
    oe_lambda: float = 0.0
    oe_batch_size: int = 16
    wandb_project: str = "intent-router"
    wandb_group: str = "bakeoff"

    def __post_init__(self) -> None:
        if self.attn_implementation not in (None, "eager", "sdpa"):
            raise ValueError(f"attn_implementation must be eager|sdpa|None: {self}")
        if self.stop_epoch is not None and not 1 <= self.stop_epoch <= self.epochs:
            raise ValueError(f"stop_epoch {self.stop_epoch} must be in [1, epochs={self.epochs}]")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1: {self.grad_accum}")
        if self.supcon_lambda is not None and self.supcon_lambda < 0:
            raise ValueError(f"supcon_lambda must be >= 0: {self.supcon_lambda}")
        if not 0.0 <= self.id_randomize_p <= 1.0:
            raise ValueError(f"id_randomize_p must be in [0, 1]: {self.id_randomize_p}")
        if self.oe_lambda < 0 or self.oe_batch_size < 1:
            raise ValueError(f"oe_lambda must be >= 0 and oe_batch_size >= 1: {self}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class TrainHooks:
    """Open-set improvement round training extras, kept out of TrainConfig so run configs (and the
    equality checks
    that reuse earlier curves) are unchanged. None / defaults => the original behaviour.

    head_init_seed: seed in force while the model is built (classifier-head init); afterwards the
        RNGs are re-seeded with cfg.model_seed, so members of a soup share one head init and
        differ in data order / dropout.
    aux_loss: callable(logits=, idx=, epoch=, forward=, texts=, row_ids=, model_seed=,
        id_randomize_p=) -> scalar tensor, added to the loss of every optimizer step (I4 twin KL).
        `forward(texts)` runs the model in its current (train) mode on extra texts -> logits.
    """

    head_init_seed: int | None = None
    aux_loss: Callable[..., Any] | None = None


@dataclass
class FoldResult:
    """Outcome of one fold-run. probs holds per-epoch eval softmax (E, n_eval, 12)."""

    epochs: list[dict[str, float]]
    probs: np.ndarray
    eval_ids: list[str]
    gold: np.ndarray
    wall_clock_s: float
    peak_vram_mb: float
    wandb_url: str | None
    wandb_logged: bool
    nan_detected: bool
    precision: str
    config: dict[str, Any] = field(default_factory=dict)
    # GPU exclusivity record (owner decision: every run records whether it had the GPU alone).
    gpu_snapshot_start: dict[str, Any] = field(default_factory=dict)
    gpu_snapshot_end: dict[str, Any] = field(default_factory=dict)
    gpu_exclusive: bool = False
    gpu_foreign_seen: list[dict[str, Any]] = field(default_factory=list)
    # Timing breakdown (seconds): model load, and per-epoch train / eval.
    load_s: float = 0.0
    train_epoch_s: list[float] = field(default_factory=list)
    eval_epoch_s: list[float] = field(default_factory=list)


# ---------------------------------------------------------------- label helpers
def _label_maps() -> tuple[list[str], dict[str, int]]:
    if not data_mod.LABELS:
        data_mod.load_data()
    return list(data_mod.LABELS), data_mod.label2id()


def parent_index(labels: list[str]) -> list[int]:
    """Map each label to a parent id (the 3 shipment_information.* share one: 12 -> 10)."""
    parents: list[int] = []
    sibling_parent: int | None = None
    n_parents = 0
    for lab in labels:
        if lab.startswith(SIBLING_PREFIX):
            if sibling_parent is None:
                sibling_parent = n_parents
                n_parents += 1
            parents.append(sibling_parent)
        else:
            parents.append(n_parents)
            n_parents += 1
    return parents


def parent_logits(logits: Any, parent_of: list[int]) -> Any:
    """Parent logits: logsumexp over a parent's children (identity for singleton parents)."""
    import torch

    n_parents = max(parent_of) + 1
    cols = []
    for p in range(n_parents):
        idx = [i for i, q in enumerate(parent_of) if q == p]
        cols.append(torch.logsumexp(logits[:, idx], dim=1))
    return torch.stack(cols, dim=1)


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights normalised to mean 1 over classes (absent class -> count 1)."""
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1).astype(float)
    w = 1.0 / counts
    return w * n_classes / w.sum()


def supcon_loss(z: Any, labels: Any, temperature: float) -> Any:
    """Supervised contrastive loss (Khosla et al. 2020, L_out, one view) over one batch.

    z: [N, d] L2-normalised embeddings, labels: [N]. For each anchor i with at least one other
    same-class row, loss_i = -mean over positives p of log(exp(z_i.z_p / T) / sum over a != i of
    exp(z_i.z_a / T)); the result is the mean over such anchors. A batch with no anchor gives 0
    (with a graph connection to z, so backward stays valid).
    """
    import torch

    n = z.shape[0]
    sim = (z @ z.T) / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))  # self never enters the denominator
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(dim=1)
    anchors = n_pos > 0
    if not bool(anchors.any()):
        return z.sum() * 0.0
    # masked_fill (not multiply): log_prob on the diagonal is -inf and -inf * 0 would be NaN.
    pos_sum = log_prob.masked_fill(~pos, 0.0).sum(dim=1)
    return -(pos_sum[anchors] / n_pos[anchors]).mean()


def _micro_slices(n: int, k: int) -> list[slice | None]:
    """[None] (whole batch) when k <= 1, else k near-equal consecutive slices of range(n)."""
    if k <= 1:
        return [None]
    size = math.ceil(n / k)
    return [slice(s, min(s + size, n)) for s in range(0, n, size)]


# ------------------------------------------------------------ Robustness and open-set training
# fixes training-time helpers
# Same shape as robustness._ID_RE (PO-n, LD-n and any other [A-Z]{2,4}-n); kept local because
# robustness imports this module.
_ID_RE = re.compile(r"\b([A-Z]{2,4})-(\d+)")
ID_PREFIX_CHOICES = ("PO", "LD", "REF", None)  # None => a random 2-4 letter uppercase prefix
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _seeded_rng(*parts: object) -> np.random.Generator:
    """Generator from a sha256 of the parts: independent of row order and of other streams."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def randomize_ids(text: str, p: float, rng: np.random.Generator) -> str:
    """A1: with probability p per ID occurrence, swap its prefix; the digits are kept.

    The replacement is drawn uniformly from {PO, LD, REF, random 2-4 uppercase letters} (it may
    equal the original prefix). Draws are consumed left to right, only for occurrences that fire.
    """

    def repl(m: re.Match[str]) -> str:
        if rng.random() >= p:
            return m.group(0)
        choice = ID_PREFIX_CHOICES[int(rng.integers(len(ID_PREFIX_CHOICES)))]
        if choice is None:
            n = int(rng.integers(2, 5))
            choice = "".join(_UPPER[int(i)] for i in rng.integers(0, 26, size=n))
        return f"{choice}-{m.group(2)}"

    return _ID_RE.sub(repl, text)


def id_rng(seed: int, epoch: int, row_id: str) -> np.random.Generator:
    """The A1 generator of one (model seed, epoch, row id): deterministic, differs per epoch."""
    return _seeded_rng("idrand", seed, epoch, row_id)


def oe_batch_indices(seed: int, epoch: int, step: int, n_oe: int, batch_size: int) -> np.ndarray:
    """A2: indices of the OE mini-batch of one step, seeded by (seed, epoch, step).

    Without replacement when the pool has at least batch_size items, else with replacement.
    """
    rng = _seeded_rng("oe", seed, epoch, step)
    return rng.choice(n_oe, size=batch_size, replace=n_oe < batch_size)


def oe_uniform_loss(logits: Any) -> Any:
    """A2: mean over items of CE to the uniform distribution over K classes = -mean log_softmax.

    Equals log K for uniform logits (its minimum over the simplex of predictions per item).
    """
    import torch.nn.functional as F  # noqa: N812

    return -F.log_softmax(logits.float(), dim=-1).mean()


# ------------------------------------------------------------ optimizer groups
_LAYER_RE = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")


def build_param_groups(
    model: Any, lr: float, weight_decay: float, layer_decay: float | None
) -> list[dict[str, Any]]:
    """Build AdamW param groups.

    With layer_decay d: layer i gets lr*d^(L-1-i), embeddings lr*d^L, and everything else
    (classifier head, pooler, encoder-level rel_embeddings/LayerNorm) the base lr.
    Biases and LayerNorm weights get no weight decay.
    """
    n_layers = int(model.config.num_hidden_layers)
    groups: dict[tuple[float, bool], list[Any]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = 1.0
        if layer_decay is not None:
            m = _LAYER_RE.search(name)
            if m:
                scale = layer_decay ** (n_layers - 1 - int(m.group(1)))
            elif ".embeddings." in f".{name}":
                scale = layer_decay**n_layers
        no_decay = name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name
        groups.setdefault((lr * scale, no_decay), []).append(p)
    return [
        {"params": ps, "lr": g_lr, "weight_decay": 0.0 if nd else weight_decay}
        for (g_lr, nd), ps in groups.items()
    ]


# ------------------------------------------------------------------- wandb glue
def _wandb_init(cfg: TrainConfig, run_meta: dict[str, Any]) -> Any:
    """Start a W&B run; on any failure return None so training continues unlogged."""
    try:
        import os

        import wandb

        # Keep local W&B files out of the repo root (outputs/ is not committed).
        Path("outputs").mkdir(exist_ok=True)
        os.environ.setdefault("WANDB_DIR", "outputs")
        return wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=run_meta["run_id"],
            config={**asdict(cfg), **{k: v for k, v in run_meta.items() if k != "run_id"}},
            reinit=True,
            settings=wandb.Settings(init_timeout=90),
        )
    except Exception as exc:  # noqa: BLE001 - wandb must never kill a training run
        print(f"[wandb] init failed, continuing without logging: {exc!r}")
        return None


def _wandb_log(run: Any, payload: dict[str, Any], step: int, state: dict[str, bool]) -> None:
    if run is None or not state["ok"]:
        return
    try:
        run.log(payload, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed, disabling logging for this run: {exc!r}")
        state["ok"] = False


# ------------------------------------------------------------------ train fold
def _encode_texts(
    tokenizer: Any, raw_texts: Sequence[str], cfg: TrainConfig
) -> list[dict[str, list[int]]]:
    texts = [prepare_text(t, cfg.model_name, cfg.entity_masking) for t in raw_texts]
    enc = tokenizer(texts, truncation=True, max_length=cfg.max_len)
    keys = list(enc.keys())
    return [{k: enc[k][i] for k in keys} for i in range(len(texts))]


def _encode(tokenizer: Any, df: pd.DataFrame, cfg: TrainConfig) -> list[dict[str, list[int]]]:
    return _encode_texts(tokenizer, list(df["text"]), cfg)


def _batches(
    feats: list[dict[str, list[int]]], order: np.ndarray, bs: int, tokenizer: Any, device: Any
) -> Any:
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        batch = tokenizer.pad([feats[i] for i in idx], return_tensors="pt")
        yield idx, {k: v.to(device) for k, v in batch.items()}


def _batches_id_randomized(
    texts: Sequence[str],
    row_ids: Sequence[str],
    order: np.ndarray,
    cfg: TrainConfig,
    epoch: int,
    tokenizer: Any,
    device: Any,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """A1 batches: each row's IDs are re-randomised (seeded by model seed, epoch, row id)."""
    for s in range(0, len(order), cfg.batch_size):
        idx = order[s : s + cfg.batch_size]
        raw = [
            randomize_ids(texts[i], cfg.id_randomize_p, id_rng(cfg.model_seed, epoch, row_ids[i]))
            for i in idx
        ]
        batch = tokenizer.pad(_encode_texts(tokenizer, raw, cfg), return_tensors="pt")
        yield idx, {k: v.to(device) for k, v in batch.items()}


def _check_extra_rows(extra: pd.DataFrame, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """A3 leakage guard: extra rows need id/text/label, unique new ids, no held-out source."""
    missing = {"id", "text", "label"} - set(extra.columns)
    if missing:
        raise ValueError(f"extra_train_rows is missing columns {sorted(missing)}")
    ids = set(extra["id"].astype(str))
    if len(ids) != len(extra) or ids & set(train_df["id"].astype(str)):
        raise AssertionError("extra_train_rows ids must be unique and not already in train_df")
    if ids & set(eval_df["id"].astype(str)):
        raise AssertionError("extra_train_rows ids overlap eval_df")
    if "src_id" in extra.columns and set(extra["src_id"].astype(str)) & set(
        eval_df["id"].astype(str)
    ):
        raise AssertionError("extra_train_rows contains a copy of an eval_df row (src_id leak)")


def train_fold(
    cfg: TrainConfig | dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    run_meta: dict[str, Any],
    *,
    oe_texts: Sequence[str] | None = None,
    extra_train_rows: pd.DataFrame | None = None,
    hooks: TrainHooks | None = None,
) -> FoldResult:
    """Fine-tune on train_df, evaluate on eval_df after every epoch. Never touches test rows.

    Custom loop (not HF Trainer): per-epoch probabilities, deterministic data order from
    model_seed, optional hierarchical aux loss and layer-wise LR decay are all direct.
    oe_texts (A2 outlier-exposure pool) and extra_train_rows (A3 augmented copies, columns
    id/text/label) are Robustness and open-set training fixes additions; None => the original
    behaviour.
    """
    return _train(
        cfg, train_df, eval_df, run_meta, keep_model=False,
        oe_texts=oe_texts, extra_train_rows=extra_train_rows, hooks=hooks,
    )[0]  # fmt: skip


def train_model(
    cfg: TrainConfig | dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    run_meta: dict[str, Any],
    labels: list[str] | None = None,
    *,
    oe_texts: Sequence[str] | None = None,
    extra_train_rows: pd.DataFrame | None = None,
    hooks: TrainHooks | None = None,
) -> tuple[FoldResult, Any, Any]:
    """Like train_fold but also returns (result, trained model, tokenizer); model stays on device.

    The model holds the weights after the last epoch run (stop_epoch if set). Caller frees it.
    labels (default: all dataset labels) is the model's label space, index = position; Track B
    passes the non-held-out classes so the head has exactly that many outputs.
    """
    res, model, tokenizer = _train(
        cfg, train_df, eval_df, run_meta, keep_model=True, labels=labels,
        oe_texts=oe_texts, extra_train_rows=extra_train_rows, hooks=hooks,
    )  # fmt: skip
    return res, model, tokenizer


def _train(
    cfg: TrainConfig | dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    run_meta: dict[str, Any],
    keep_model: bool,
    labels: list[str] | None = None,
    oe_texts: Sequence[str] | None = None,
    extra_train_rows: pd.DataFrame | None = None,
    hooks: TrainHooks | None = None,
) -> tuple[FoldResult, Any, Any]:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from transformers import get_linear_schedule_with_warmup

    cfg = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg)
    if cfg.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {cfg.precision!r}")
    for name, df in (("train_df", train_df), ("eval_df", eval_df)):
        if "split" in df.columns and (df["split"] == "test").any():
            raise AssertionError(f"{name} contains test-split rows")
    if set(train_df["id"]) & set(eval_df["id"]):
        raise AssertionError("train_df and eval_df overlap")
    use_oe = cfg.oe_lambda > 0
    aux_loss = hooks.aux_loss if hooks is not None else None
    if aux_loss is not None and cfg.grad_accum != 1:
        raise ValueError("TrainHooks.aux_loss needs grad_accum == 1")
    if use_oe and not oe_texts:
        raise ValueError("oe_lambda > 0 needs a non-empty oe_texts pool")
    if extra_train_rows is not None and len(extra_train_rows):
        _check_extra_rows(extra_train_rows, train_df, eval_df)
        cols = ["id", "text", "label"]
        train_df = pd.concat([train_df[cols], extra_train_rows[cols]], ignore_index=True)

    if labels is None:
        labels, l2i = _label_maps()
    else:
        labels = list(labels)
        l2i = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = cfg.precision if device.type == "cuda" else "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    use_amp = amp_dtype is not None

    head_seed = hooks.head_init_seed if hooks is not None else None
    seed_everything(cfg.model_seed if head_seed is None else head_seed)
    # Taken before our model is loaded so our own allocation cannot read as a foreign one.
    snap_start = gpu_lock.gpu_snapshot()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    tokenizer, model = build_model(cfg.model_name, labels, cfg.attn_implementation)
    if head_seed is not None:  # head drawn from head_seed; everything after follows the member seed
        seed_everything(cfg.model_seed)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load

    if not train_df["label"].isin(l2i).all():
        raise ValueError("train rows carry labels outside the model's label space")
    y_train = train_df["label"].map(l2i).to_numpy()
    y_eval = eval_df["label"].map(l2i).to_numpy()
    train_feats = _encode(tokenizer, train_df, cfg)
    eval_feats = _encode(tokenizer, eval_df, cfg)
    oe_feats = _encode_texts(tokenizer, list(oe_texts), cfg) if use_oe and oe_texts else []
    need_texts = cfg.id_randomize_p > 0 or aux_loss is not None
    train_texts = list(train_df["text"]) if need_texts else []
    train_ids = [str(i) for i in train_df["id"]] if need_texts else []

    def _forward_extra(texts: list[str]) -> Any:
        """Model logits (current mode, autocast) on extra texts; used by hooks.aux_loss."""
        enc = tokenizer.pad(_encode_texts(tokenizer, texts, cfg), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            return model(**enc).logits.float()

    weight = (
        torch.tensor(class_weights(y_train, n_classes), dtype=torch.float32, device=device)
        if cfg.class_weighted
        else None
    )
    parent_of = parent_index(labels)
    parent_y = torch.tensor([parent_of[i] for i in y_train], device=device)
    y_train_t = torch.tensor(y_train, device=device)

    optim = torch.optim.AdamW(build_param_groups(model, cfg.lr, cfg.weight_decay, cfg.layer_decay))
    steps_per_epoch = math.ceil(len(train_feats) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    gen = torch.Generator().manual_seed(cfg.model_seed)
    use_supcon = bool(cfg.supcon_lambda)  # None / 0 => the open-set evaluation code path, unchanged
    fwd_kwargs: dict[str, Any] = {"output_hidden_states": True} if use_supcon else {}

    run = _wandb_init(cfg, {**run_meta, "gpu_snapshot_start": snap_start})
    wb_state = {"ok": run is not None}
    wandb_url = getattr(run, "url", None) if run is not None else None

    epochs_out: list[dict[str, float]] = []
    probs_out: list[np.ndarray] = []
    nan_detected = False
    foreign_seen: list[dict[str, Any]] = list(snap_start["foreign"])
    train_s: list[float] = []
    eval_s: list[float] = []
    step = 0
    t0 = time.perf_counter()
    try:
        for epoch in range(1, (cfg.stop_epoch or cfg.epochs) + 1):
            t_ep = time.perf_counter()
            model.train()
            order = torch.randperm(len(train_feats), generator=gen).numpy()
            losses: list[float] = []
            if cfg.id_randomize_p > 0:
                batches = _batches_id_randomized(
                    train_texts, train_ids, order, cfg, epoch, tokenizer, device
                )
            else:
                batches = _batches(train_feats, order, cfg.batch_size, tokenizer, device)
            for idx, batch in batches:
                idx_t = torch.as_tensor(idx, device=device)
                optim.zero_grad(set_to_none=True)
                loss_val = 0.0
                for sl in _micro_slices(len(idx), cfg.grad_accum):
                    mb = batch if sl is None else {k: v[sl] for k, v in batch.items()}
                    idx_mb = idx_t if sl is None else idx_t[sl]
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(**mb, **fwd_kwargs)
                    logits = out.logits.float()
                    loss = F.cross_entropy(
                        logits,
                        y_train_t[idx_mb],
                        weight=weight,
                        label_smoothing=cfg.label_smoothing,
                    )
                    if cfg.hier_aux:
                        loss = loss + cfg.hier_lambda * F.cross_entropy(
                            parent_logits(logits, parent_of), parent_y[idx_mb]
                        )
                    if use_supcon:
                        m = mb["attention_mask"].unsqueeze(-1).float()
                        pooled = (out.hidden_states[-1].float() * m).sum(1) / m.sum(1)
                        loss = loss + cfg.supcon_lambda * supcon_loss(
                            F.normalize(pooled, dim=-1), y_train_t[idx_mb], cfg.supcon_temperature
                        )
                    if aux_loss is not None:
                        loss = loss + aux_loss(
                            logits=logits, idx=idx, epoch=epoch, forward=_forward_extra,
                            texts=train_texts, row_ids=train_ids, model_seed=cfg.model_seed,
                            id_randomize_p=cfg.id_randomize_p,
                        )  # fmt: skip
                    if sl is not None:  # micro-batch share of the full-batch mean loss
                        loss = loss * (len(idx_mb) / len(idx))
                    loss_val += float(loss.detach())
                    scaler.scale(loss).backward()
                if use_oe:
                    # A2: one OE mini-batch per optimizer step, seeded by (seed, epoch, step); its
                    # gradient is accumulated on top of the CE gradient (same as summing losses).
                    oe_idx = oe_batch_indices(
                        cfg.model_seed, epoch, step, len(oe_feats), cfg.oe_batch_size
                    )
                    oe_batch = tokenizer.pad([oe_feats[i] for i in oe_idx], return_tensors="pt")
                    oe_batch = {k: v.to(device) for k, v in oe_batch.items()}
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        oe_logits = model(**oe_batch).logits
                    oe_loss = cfg.oe_lambda * oe_uniform_loss(oe_logits)
                    loss_val += float(oe_loss.detach())
                    scaler.scale(oe_loss).backward()
                if math.isfinite(loss_val):
                    losses.append(loss_val)
                else:
                    nan_detected = True
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                sched.step()
                step += 1
                _wandb_log(
                    run,
                    {"lr": sched.get_last_lr()[-1], "train/step_loss": loss_val},
                    step,
                    wb_state,
                )

            if device.type == "cuda":
                torch.cuda.synchronize()
            train_s.append(time.perf_counter() - t_ep)
            t_ev = time.perf_counter()
            model.eval()
            probs = np.zeros((len(eval_feats), n_classes), dtype=np.float32)
            eval_loss_sum = 0.0
            with torch.no_grad():
                for idx, batch in _batches(
                    eval_feats, np.arange(len(eval_feats)), cfg.batch_size, tokenizer, device
                ):
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                        logits = model(**batch).logits
                    logits = logits.float()
                    yb = torch.as_tensor(y_eval[idx], device=device)
                    eval_loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
                    probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            eval_s.append(time.perf_counter() - t_ev)
            # Cheap per-epoch check: any other GPU user during this run voids exclusivity.
            foreign_seen += [{**f, "epoch": epoch} for f in gpu_lock.foreign_gpu_processes()]
            if not np.isfinite(probs).all():
                nan_detected = True
            pred = probs.argmax(axis=1)
            rec = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "eval_loss": eval_loss_sum / len(eval_feats),
                "macro_f1": macro_f1(y_eval, pred, n_classes),
                "accuracy": accuracy(y_eval, pred),
            }
            epochs_out.append(rec)
            probs_out.append(probs)
            _wandb_log(
                run,
                {
                    "epoch": epoch,
                    "train/loss": rec["train_loss"],
                    "eval/loss": rec["eval_loss"],
                    "eval/macro_f1": rec["macro_f1"],
                    "eval/accuracy": rec["accuracy"],
                },
                step,
                wb_state,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        snap_end = gpu_lock.gpu_snapshot()
        foreign_seen += [{**f, "epoch": "end"} for f in snap_end["foreign"]]
        gpu_excl = not foreign_seen
        if run is not None:
            try:
                run.summary.update(
                    {
                        "gpu_snapshot_end": snap_end,
                        "gpu_exclusive": gpu_excl,
                        "gpu_foreign_seen": foreign_seen,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] summary update failed: {exc!r}")
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] finish failed: {exc!r}")
        del optim, sched, scaler
        if not keep_model:
            model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logged = bool(wb_state["ok"] and run is not None and not getattr(run, "disabled", False))
    result = FoldResult(
        epochs=epochs_out,
        probs=np.stack(probs_out),
        eval_ids=[str(i) for i in eval_df["id"]],
        gold=y_eval,
        wall_clock_s=wall,
        peak_vram_mb=peak,
        wandb_url=wandb_url,
        wandb_logged=logged,
        nan_detected=nan_detected,
        precision=precision,
        config={**asdict(cfg), "model_short": model_short(cfg.model_name)},
        gpu_snapshot_start=snap_start,
        gpu_snapshot_end=snap_end,
        gpu_exclusive=gpu_excl,
        gpu_foreign_seen=foreign_seen,
        load_s=load_s,
        train_epoch_s=train_s,
        eval_epoch_s=eval_s,
    )
    return result, (model if keep_model else None), (tokenizer if keep_model else None)
