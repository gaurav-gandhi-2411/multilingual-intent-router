from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from intent_router.data import load_data
from intent_router.dedup import compute_dup_groups


def _fill_target(groups: list[tuple[int, int]], target: int, rng: np.random.Generator) -> set[int]:
    """Greedily pick (group_id, size) units, in seeded random order, to reach `target` rows.

    A unit is only taken if it does not overshoot the target. Because near-duplicate groups
    are tiny (size 1-2), the result is within 1 row of the target.
    """
    order = rng.permutation(len(groups))
    chosen: set[int] = set()
    count = 0
    for k in order:
        gid, size = groups[k]
        if count + size <= target:
            chosen.add(gid)
            count += size
        if count == target:
            break
    return chosen


def assign_splits(
    df: pd.DataFrame,
    dup_group: np.ndarray,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> pd.Series:
    """Per-class exact-count split, grouped by dup_group.

    Test then val are drawn from each class with target round(n_c * frac); the remainder is
    train. A dup_group is never split across partitions. Groups spanning multiple labels are
    rejected (none exist in this dataset; the check keeps the per-class logic valid).
    """
    labels = df["label"].to_numpy()
    grp_labels = pd.DataFrame({"g": dup_group, "y": labels}).groupby("g")["y"].nunique()
    if (grp_labels > 1).any():
        raise ValueError("dup_group spans multiple labels; per-class split is not defined")
    rng = np.random.default_rng(seed)
    split = np.full(len(df), "train", dtype=object)
    for label in sorted(df["label"].unique()):
        idx = np.flatnonzero(labels == label)
        n_c = len(idx)
        sizes = pd.Series(dup_group[idx]).value_counts().sort_index()
        # sorted by group id first so the seeded permutation is reproducible
        units = [(int(g), int(s)) for g, s in sizes.items()]
        test_groups = _fill_target(units, round(n_c * test_frac), rng)
        split[idx[np.isin(dup_group[idx], list(test_groups))]] = "test"
        rest_units = [(g, s) for g, s in units if g not in test_groups]
        val_groups = _fill_target(rest_units, round(n_c * val_frac), rng)
        split[idx[np.isin(dup_group[idx], list(val_groups))]] = "val"
    return pd.Series(split, index=df.index, name="split")


def assign_cv_folds(
    df: pd.DataFrame, dup_group: np.ndarray, split: pd.Series, n_splits: int, seed: int
) -> np.ndarray:
    """StratifiedGroupKFold folds on train+val rows; -1 for test rows."""
    folds = np.full(len(df), -1, dtype=int)
    mask = (split != "test").to_numpy()
    idx = np.flatnonzero(mask)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["label"].to_numpy()[idx]
    for fold, (_, te) in enumerate(sgkf.split(np.zeros(len(idx)), y, groups=dup_group[idx])):
        folds[idx[te]] = fold
    return folds


def build_splits(cfg: dict) -> pd.DataFrame:
    """Build the splits frame (id, split, dup_group, cv_fold_s*) from config."""
    df = load_data(cfg["data_path"])
    dup_group, _, _ = compute_dup_groups(df, cfg["dup_threshold"])
    sc = cfg["split"]
    split = assign_splits(df, dup_group, sc["test_frac"], sc["val_frac"], cfg["seed"])
    out = pd.DataFrame({"id": df["id"], "split": split, "dup_group": dup_group})
    for k, s in enumerate(sc["cv_seeds"]):
        out[f"cv_fold_s{k}"] = assign_cv_folds(df, dup_group, split, sc["cv_folds"], s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = build_splits(cfg)
    path = Path(cfg["splits_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")
    print(out["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
