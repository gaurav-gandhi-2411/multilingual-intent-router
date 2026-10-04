from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_tradeoff", ROOT / "scripts" / "build_tradeoff.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc() -> dict:
    needed = [
        "phase4e/final_v1_vs_v3.json",
        "phase4e/axes.json",
        "phase4e/confirm_a1a3.json",
        "final_v1/track_a.json",
        "final_v3/track_a.json",
    ]
    if not all((RESULTS / p).exists() for p in needed):
        pytest.skip("results files not present")
    return _load_module().build_tradeoff()


def test_test_split_numbers_match_sources(doc: dict) -> None:
    src = json.loads((RESULTS / "phase4e/final_v1_vs_v3.json").read_text(encoding="utf-8"))
    assert doc["track_a_test"]["v1"]["macro_f1"]["point"] == src["v1"]["macro_f1"]["point"]
    assert doc["track_a_test"]["v1"]["macro_f1"]["point"] == pytest.approx(0.9465160656337127)
    assert doc["track_a_test"]["v3"]["macro_f1"]["point"] == pytest.approx(0.960812879930527)


def test_headline_strict_rej95_delta(doc: dict) -> None:
    r = doc["trackb_holdout"]["headline"]["strict_recall_95"]
    assert r["delta_v3_minus_v1"] == pytest.approx(-0.133, abs=1e-3)
    assert r["ci95"][1] < 0  # CI excludes zero: v1 better


def test_reduction_ci_negated_into_v3_minus_v1(doc: dict) -> None:
    flip = doc["robustness_cv"]["neutral_swap_flip_rate (axis c, lower is better)"]
    assert flip["ci_negated_from_reduction"] is True
    lo, hi = flip["ci95"]
    assert lo < flip["delta_v3_minus_v1"] < hi < 0


def test_every_number_node_has_source_and_verdict_text(doc: dict) -> None:
    assert doc["track_a_test"]["v3"]["temperature"]["source"].startswith("phase4e/")
    assert all(isinstance(t, str) and t for t in doc["verdict"]["text"])
    assert any("v1 = ship default" in g for g in doc["deployment_guidance"])


def test_build_is_deterministic() -> None:
    mod = _load_module()
    if not (RESULTS / "phase4e/axes.json").exists():
        pytest.skip("results files not present")
    assert json.dumps(mod.build_tradeoff()) == json.dumps(mod.build_tradeoff())
