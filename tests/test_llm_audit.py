from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intent_router import llm_audit as la

LABELS = [f"L{i}" for i in range(12)]


def _oof(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    gold = np.arange(n) % 12
    pred = gold.copy()
    pred[rng.choice(n, 20, replace=False)] += 1
    pred %= 12
    conf = rng.uniform(0.5, 1.0, n)
    conf[: n // 2] = 0.995
    return pd.DataFrame({"id": [f"r{i:04d}" for i in range(n)], "gold": gold, "pred": pred,
                         "conf": conf})  # fmt: skip


def test_parse_exact_and_case() -> None:
    assert la.parse_choice("  L3\n", LABELS) == {"valid": True, "choice": "L3", "pair": ""}
    assert la.parse_choice("l3", LABELS)["choice"] == "L3"


def test_parse_ambiguous() -> None:
    r = la.parse_choice("Ambiguous: L1 | L2", LABELS)
    assert r == {"valid": True, "choice": "ambiguous", "pair": "L1|L2"}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "L3 because",
        "The answer is L3",
        "ambiguous: L1 | L1",
        "ambiguous: L1 | X",
        "ambiguous",
        "L3.",
    ],
)
def test_parse_invalid(raw: str) -> None:
    r = la.parse_choice(raw, LABELS)
    assert not r["valid"] and r["choice"] == "invalid"


def test_fleiss_kappa_textbook() -> None:
    # Wikipedia worked example: 10 subjects, 14 raters, 5 categories, kappa = 0.210.
    m = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                  [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                  [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])  # fmt: skip
    assert la.fleiss_kappa(m) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_perfect_agreement() -> None:
    m = np.array([[3, 0], [0, 3], [3, 0]])
    assert la.fleiss_kappa(m) == pytest.approx(1.0)


def test_cohen_and_consensus() -> None:
    assert la.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert la.consensus(["a", "a", "b"]) == "a"
    assert la.consensus(["a", "b", "invalid"]) is None
    assert la.consensus(["ambiguous", "ambiguous", "a"]) == "ambiguous"


def test_sampling_deterministic_no_overlap() -> None:
    oof = _oof()
    a = la.sample_items(oof, 42)
    b = la.sample_items(oof, 42)
    pd.testing.assert_frame_equal(a, b)
    assert la.sample_items(oof, 7)["id"].tolist() != a["id"].tolist()
    assert a["id"].is_unique
    counts = a["item_type"].value_counts().to_dict()
    assert counts == {"error": 20, "control": 20, "calibration": 10}
    cal = a[a["item_type"] == "calibration"]
    assert cal["gold"].is_unique and (cal["model_conf"] >= 0.99).all()
    ctrl = a[a["item_type"] == "control"]
    assert (ctrl["gold"] == ctrl["model_pred"]).all()
    assert (
        a[a["item_type"] == "error"]["gold"] != a[a["item_type"] == "error"]["model_pred"]
    ).all()


@pytest.mark.parametrize(
    "url", ["http://8.8.8.8:11434", "http://example.com", "https://127.0.0.1:11434",
            "http://192.168.1.5:11434", "http://127.0.0.1.evil.com:11434"],
)  # fmt: skip
def test_loopback_guard_rejects(url: str) -> None:
    with pytest.raises(ValueError):
        la.validate_loopback_url(url)


def test_loopback_guard_accepts() -> None:
    assert la.validate_loopback_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
    assert la.validate_loopback_url("http://localhost:11434")
    assert la.validate_loopback_url("http://[::1]:11434")


def test_judge_constructor_rejects_remote() -> None:
    with pytest.raises(ValueError):
        la.OllamaJudge("http://10.0.0.2:11434", "x", None, {})
