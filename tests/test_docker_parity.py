"""Model-free tests for scripts/docker_parity.py (the synthetic list + the comparison logic)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "docker_parity.py"
    spec = importlib.util.spec_from_file_location("docker_parity", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_texts_are_20_unique_nonblank() -> None:
    texts = _load().SYNTHETIC_TEXTS
    assert len(texts) == 20 and len(set(texts)) == 20
    assert all(t.strip() and len(t) <= 2000 for t in texts)


def test_compare_detects_label_and_prob_drift() -> None:
    dp = _load()
    top = [{"label": "a", "prob": 0.7}, {"label": "b", "prob": 0.2}, {"label": "c", "prob": 0.1}]
    base = {"label": "a", "confidence": 0.7, "ood_score": -3.0, "abstained": False, "top3": top}
    same = dp.compare(base, base)
    assert same["label_equal"] and same["max_abs_diff_top3_prob"] == 0.0
    other = {**base, "label": "b", "confidence": 0.69, "top3": [{**top[0], "prob": 0.69}, *top[1:]]}
    d = dp.compare(other, base)
    assert not d["label_equal"]
    assert d["max_abs_diff_top3_prob"] == pytest.approx(0.01)
