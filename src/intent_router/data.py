from __future__ import annotations

import pandas as pd

DEFAULT_DATA_PATH = "data/dataset.csv"
DEFAULT_SPLITS_PATH = "splits/splits.csv"
REQUIRED_COLUMNS = ["id", "text", "label"]

# Derived from the data at load time; populated by load_data().
LABELS: list[str] = []


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the dataset (columns, nulls, unique ids); refresh LABELS."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    df = df[REQUIRED_COLUMNS].copy()
    if df.isna().any().any():
        raise ValueError(f"{path}: null values found: {df.isna().sum().to_dict()}")
    if not df["id"].is_unique:
        raise ValueError(f"{path}: duplicate ids")
    LABELS[:] = sorted(df["label"].unique())
    return df


def label2id() -> dict[str, int]:
    """Map label -> index using the sorted LABELS list (call load_data first)."""
    return {label: i for i, label in enumerate(LABELS)}


def id2label() -> dict[int, str]:
    """Map index -> label using the sorted LABELS list (call load_data first)."""
    return dict(enumerate(LABELS))


def load_splits(path: str = DEFAULT_SPLITS_PATH) -> pd.DataFrame:
    """Load splits/splits.csv (id, split, dup_group, cv_fold_s0..s2)."""
    return pd.read_csv(path)


def get_frame(
    split_names: list[str],
    data_path: str = DEFAULT_DATA_PATH,
    splits_path: str = DEFAULT_SPLITS_PATH,
) -> pd.DataFrame:
    """Return data rows merged with split info, restricted to the given split names."""
    unknown = set(split_names) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    df = load_data(data_path)
    splits = load_splits(splits_path)
    merged = df.merge(splits, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(df):
        raise ValueError("splits.csv does not cover all dataset ids; re-run split")
    return merged[merged["split"].isin(split_names)].reset_index(drop=True)
