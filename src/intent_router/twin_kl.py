"""Open-set improvement round I4: symmetric-KL consistency between a row and its prefix-swapped
twin.

I4 = A1 (ID-prefix randomisation, p = 0.5) + lambda * symmetric KL(p(row) || p(twin)), where the
twin is the same text with every ID prefix redrawn from {PO, LD, REF, random 2-4 uppercase}, digits
kept. Rows without an ID contribute exactly 0 (no twin is run for them). The loss is summed over the
rows of a batch and divided by the batch size (rows without an ID count as zeros).

symmetric KL(p, q) = KL(p || q) + KL(q || p) (Jeffreys divergence, no 1/2 factor; lambda 0.5 / 1.0
are the pre-registered weights). Both sides carry gradient (train mode, dropout active on both).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from intent_router.ood_variants import ID_RE
from intent_router.train import ID_PREFIX_CHOICES, _seeded_rng, id_rng, randomize_ids

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def has_id(text: str) -> bool:
    """True when the text contains at least one ID (the A1 / swap pattern)."""
    return ID_RE.search(text) is not None


def redraw_prefixes(text: str, rng: np.random.Generator) -> str:
    """Redraw EVERY ID prefix uniformly from {PO, LD, REF, random 2-4 uppercase}; digits kept.

    Same draw scheme as train.randomize_ids with p = 1 (the new prefix may equal the old one).
    """

    def repl(m: Any) -> str:
        choice = ID_PREFIX_CHOICES[int(rng.integers(len(ID_PREFIX_CHOICES)))]
        if choice is None:
            n = int(rng.integers(2, 5))
            choice = "".join(_UPPER[int(i)] for i in rng.integers(0, 26, size=n))
        return f"{choice}-{m.group(2)}"

    return ID_RE.sub(repl, text)


def twin_text(text: str, model_seed: int, epoch: int, row_id: str) -> str:
    """Deterministic twin of one (model seed, epoch, row id); text without an ID is unchanged."""
    return redraw_prefixes(text, _seeded_rng("twin", model_seed, epoch, row_id))


def symmetric_kl(logits_a: Any, logits_b: Any) -> Any:
    """Per-row KL(p_a || p_b) + KL(p_b || p_a) of the softmaxes of two [B, K] logit tensors."""
    import torch.nn.functional as F  # noqa: N812

    la = F.log_softmax(logits_a.float(), dim=-1)
    lb = F.log_softmax(logits_b.float(), dim=-1)
    pa, pb = la.exp(), lb.exp()
    return ((pa * (la - lb)).sum(-1)) + ((pb * (lb - la)).sum(-1))


def anchor_texts(
    texts: Sequence[str],
    idx: Sequence[int],
    row_ids: Sequence[str],
    epoch: int,
    seed: int,
    p: float,
) -> list[str]:
    """The texts the model trains on this step: A1-randomised when p > 0 (same draws as train)."""
    if p <= 0:
        return [texts[i] for i in idx]
    return [randomize_ids(texts[i], p, id_rng(seed, epoch, row_ids[i])) for i in idx]


def make_twin_loss(lam: float) -> Callable[..., Any]:
    """The `aux_loss` hook of train.TrainHooks for I4 (see the module docstring)."""
    if lam <= 0:
        raise ValueError(f"twin KL weight must be > 0, got {lam}")

    def aux(
        *,
        logits: Any,
        idx: np.ndarray,
        epoch: int,
        forward: Callable[[list[str]], Any],
        texts: Sequence[str],
        row_ids: Sequence[str],
        model_seed: int,
        id_randomize_p: float,
    ) -> Any:
        anchors = anchor_texts(texts, idx, row_ids, epoch, model_seed, id_randomize_p)
        sel = [j for j, a in enumerate(anchors) if has_id(a)]
        if not sel:
            return logits.sum() * 0.0  # keeps the graph connected; no ID => no consistency term
        twins = [twin_text(anchors[j], model_seed, epoch, row_ids[idx[j]]) for j in sel]
        twin_logits = forward(twins)
        import torch

        sel_t = torch.as_tensor(sel, device=logits.device)
        return lam * symmetric_kl(logits[sel_t], twin_logits).sum() / logits.shape[0]

    return aux
