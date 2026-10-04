from __future__ import annotations

import pandas as pd
import pytest
import yaml

from intent_router.data import load_data, load_splits

CFG_PATH = "configs/base.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def frame(cfg: dict) -> pd.DataFrame:
    data = load_data(cfg["data_path"])
    splits = load_splits(cfg["splits_path"])
    return data.merge(splits, on="id", how="inner")


def test_covers_all_ids_exactly_once(cfg: dict) -> None:
    data = load_data(cfg["data_path"])
    splits = load_splits(cfg["splits_path"])
    assert len(splits) == len(data)
    assert splits["id"].is_unique
    assert set(splits["id"]) == set(data["id"])


def test_no_id_overlap_between_splits(frame: pd.DataFrame) -> None:
    sets = {s: set(g["id"]) for s, g in frame.groupby("split")}
    assert set(sets) == {"train", "val", "test"}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not sets[a] & sets[b], f"id overlap between {a} and {b}"


def test_dup_group_not_split_across_splits(frame: pd.DataFrame) -> None:
    assert (frame.groupby("dup_group")["split"].nunique() == 1).all()


def test_dup_group_not_split_across_folds(frame: pd.DataFrame, cfg: dict) -> None:
    trval = frame[frame["split"] != "test"]
    for k in range(len(cfg["split"]["cv_seeds"])):
        assert (trval.groupby("dup_group")[f"cv_fold_s{k}"].nunique() == 1).all()


def test_each_class_in_every_split(frame: pd.DataFrame) -> None:
    for split, g in frame.groupby("split"):
        assert g["label"].nunique() == frame["label"].nunique(), split


def test_per_class_proportions_within_one_row(frame: pd.DataFrame, cfg: dict) -> None:
    sc = cfg["split"]
    targets = {
        "test": sc["test_frac"],
        "val": sc["val_frac"],
        "train": 1 - sc["test_frac"] - sc["val_frac"],
    }
    counts = frame.groupby(["label", "split"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    for split, frac in targets.items():
        dev = (counts[split] - totals * frac).abs()
        assert (dev <= 1).all(), f"{split}: max deviation {dev.max():.2f}\n{dev[dev > 1]}"


def test_cv_fold_values(frame: pd.DataFrame, cfg: dict) -> None:
    n_folds = cfg["split"]["cv_folds"]
    for k in range(len(cfg["split"]["cv_seeds"])):
        col = frame[f"cv_fold_s{k}"]
        assert (col[frame["split"] == "test"] == -1).all()
        assert col[frame["split"] != "test"].isin(range(n_folds)).all()
