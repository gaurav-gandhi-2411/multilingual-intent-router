from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def char_tfidf_cosine(texts: list[str]) -> np.ndarray:
    """Dense pairwise cosine similarity of char_wb 3-5 TF-IDF vectors (rows are L2-normed)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x = vec.fit_transform(texts)
    return (x @ x.T).toarray()


def find_pairs(sim: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Index pairs (i<j) with cosine strictly above threshold."""
    iu, ju = np.triu_indices_from(sim, k=1)
    mask = sim[iu, ju] > threshold
    return [
        (int(i), int(j), float(s))
        for i, j, s in zip(iu[mask], ju[mask], sim[iu, ju][mask], strict=True)
    ]


def union_find_groups(n: int, pairs: list[tuple[int, int, float]]) -> np.ndarray:
    """Connected components over pairs; returns a group id per row (singletons get own id).

    Group ids are assigned in order of first appearance, so they are deterministic.
    """
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=int)


def compute_dup_groups(
    df: pd.DataFrame, threshold: float = 0.8
) -> tuple[np.ndarray, list[tuple[int, int, float]], np.ndarray]:
    """Return (dup_group per row, pairs above threshold, full cosine matrix).

    Shared by eda.py and split.py so both use the identical grouping.
    """
    sim = char_tfidf_cosine(df["text"].tolist())
    pairs = find_pairs(sim, threshold)
    groups = union_find_groups(len(df), pairs)
    return groups, pairs, sim
