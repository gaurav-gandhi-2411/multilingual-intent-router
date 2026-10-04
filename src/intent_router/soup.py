"""Open-set improvement round I3: seed soups (weight averaging of fine-tuned members that share a
head init).

Pure helpers: member rotation, state-dict averaging, the Wortsman-style greedy order/inclusion rule
and the seeded inner 85% / 15% split. No model is built here; callers pass state dicts / callbacks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def rotated_members(pool_size: int, n_members: int, replicate: int, shift: int = 2) -> list[int]:
    """Members of soup replicate r: n_members consecutive pool indices from shift * r (mod pool).

    pool 7, 5 members, shift 2: r=0 -> [0..4], r=1 -> [2..6], r=2 -> [4, 5, 6, 0, 1].
    """
    if not 1 <= n_members <= pool_size:
        raise ValueError(f"n_members {n_members} must be in [1, pool_size={pool_size}]")
    return [(shift * replicate + j) % pool_size for j in range(n_members)]


def average_state_dicts(sds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Uniform average of floating-point tensors (accumulated in float64, cast back to the input
    dtype); non-floating tensors (integer buffers) must be identical in every member and are copied.
    """
    import torch

    if not sds:
        raise ValueError("no state dicts to average")
    keys = list(sds[0])
    for sd in sds[1:]:
        if list(sd) != keys:
            raise ValueError("state dicts have different keys")
    out: dict[str, Any] = {}
    for k in keys:
        first = sds[0][k]
        if not torch.is_floating_point(first):
            for sd in sds[1:]:
                if not torch.equal(sd[k], first):
                    raise ValueError(f"non-float tensor {k!r} differs between members")
            out[k] = first.clone()
            continue
        acc = first.detach().to(torch.float64).clone()
        for sd in sds[1:]:
            if sd[k].shape != first.shape:
                raise ValueError(f"shape mismatch for {k!r}")
            acc += sd[k].detach().to(torch.float64)
        out[k] = (acc / len(sds)).to(first.dtype)
    return out


def greedy_soup(
    scores: Sequence[float], eval_subset: Callable[[list[int]], float]
) -> dict[str, Any]:
    """Greedy soup: members sorted by their own score (desc, ties by index); the best starts the
    soup and each next member is added iff the averaged soup's score does not drop (>=).

    `eval_subset(indices)` scores the uniform average of those members on the inner holdout (the
    caller binds it to inner-holdout rows only). Returns selected indices (in inclusion order), the
    sort order, the final score and a trace of every decision.
    """
    if not scores:
        raise ValueError("no members")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    selected = [order[0]]
    best = float(eval_subset(list(selected)))
    trace: list[dict[str, Any]] = [{"member": order[0], "score": best, "accepted": True}]
    for i in order[1:]:
        s = float(eval_subset([*selected, i]))
        ok = s >= best
        trace.append({"member": i, "score": s, "accepted": bool(ok)})
        if ok:
            selected.append(i)
            best = s
    return {"selected": selected, "order": order, "final_score": best, "trace": trace}


def inner_split(
    train_df: pd.DataFrame, unit: str, holdout_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded (fit, inner-holdout) split of TRAINING rows only: ceil(frac * n) rows are held out.

    `unit` names the training set (e.g. "cv_f2", "loco_orders", "headline"): each unit gets its own
    deterministic draw, independent of row order; the same split serves every soup of that unit.
    Callers pass the fold-TRAIN rows, so the evaluated fold can never reach either side.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1): {holdout_frac}")
    ids = sorted(train_df["id"].astype(str))
    h = hashlib.sha256(f"inner|{seed}|{unit}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    n_hold = int(np.ceil(holdout_frac * len(ids)))
    if n_hold < 1 or n_hold >= len(ids):
        raise ValueError(f"cannot hold out {n_hold} of {len(ids)} rows")
    hold = set(rng.permutation(ids)[:n_hold].tolist())
    is_hold = train_df["id"].astype(str).isin(hold)
    return train_df[~is_hold].reset_index(drop=True), train_df[is_hold].reset_index(drop=True)


# -------------------------------------------------------------------------- member persistence
def save_member(sd: Mapping[str, Any], path: Path) -> None:
    """Write a state dict (tensors cloned to contiguous CPU copies) as safetensors."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in sd.items()}, str(tmp))
    tmp.replace(path)


def load_member(path: Path) -> dict[str, Any]:
    """Read a state dict written by save_member (CPU tensors)."""
    from safetensors.torch import load_file

    return load_file(str(path))
