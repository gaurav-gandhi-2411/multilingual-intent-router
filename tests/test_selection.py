from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from intent_router.aggregate import chosen_epoch
from intent_router.selection import (
    ablation_result,
    decide,
    noise_band,
    select,
    verify_run,
)

CELLS = {(fs, f) for fs in range(3) for f in range(5)}


def _run(
    fs: int, fold: int, curve: list[float], ms: int = 0, abl: str | None = None, wall: float = 100.0
) -> dict[str, Any]:
    return {
        "run_id": f"x_fs{fs}_ms{ms}_f{fold}_{abl}",
        "ablation": abl,
        "fold_seed_idx": fs,
        "model_seed": ms,
        "fold": fold,
        "n_train": 80,
        "n_eval": 20,
        "wall_clock_s": wall,
        "peak_vram_mb": 1.0,
        "gpu_exclusive": True,
        "epochs": [{"macro_f1": v, "accuracy": v} for v in curve],
    }


def _runs(curve_fn: Any, abl: str | None = None, n: int = 15) -> list[dict[str, Any]]:
    cells = sorted(CELLS)[:n]
    return [_run(fs, f, curve_fn(i), abl=abl) for i, (fs, f) in enumerate(cells)]


def test_decide_branches() -> None:
    assert decide(0.02, 0.005, 0.01)["adopt"] is True
    by_band = decide(0.004, 0.005, 0.01)
    assert not by_band["adopt"] and "noise band" in by_band["verdict"]
    by_p = decide(0.02, 0.005, 0.2)
    assert not by_p["adopt"] and "p >= 0.05" in by_p["verdict"]


def test_incomplete_ablation_gives_no_decision() -> None:
    base = _runs(lambda i: [0.8, 0.9])
    abl = _runs(lambda i: [0.85, 0.95], abl="lld", n=14)
    out = ablation_result("lld", abl, base, 0.003, 2, CELLS)
    assert out["status"] == "incomplete"
    assert out["n_finished"] == 14 and out["decision"] is None


def test_epoch_argmax_is_symmetric_and_global_sensitivity() -> None:
    # baseline peaks at epoch 1 (own argmax) but the global epoch is 2; ablation peaks at epoch 2.
    base = _runs(lambda i: [0.90 + 0.001 * (i % 3), 0.85])
    abl = _runs(lambda i: [0.80, 0.92 + 0.001 * (i % 3)], abl="lld")
    assert chosen_epoch(base) == 1 and chosen_epoch(abl) == 2
    out = ablation_result("lld", abl, base, 0.003, 2, CELLS)
    assert out["baseline_epoch_own_argmax"] == 1 and out["ablation_epoch"] == 2
    assert out["baseline_macro_f1"] == pytest.approx(0.901)
    assert out["delta_macro_f1"] == pytest.approx(0.921 - 0.901)
    # sensitivity line uses the baseline at the global epoch 2 (0.85) -> larger delta
    assert out["delta_macro_f1_vs_baseline_at_global_epoch"] == pytest.approx(0.921 - 0.85)


def test_complete_ablation_adopts_when_big_and_consistent() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.002, 15)
    base = _runs(lambda i: [0.90 + noise[i]])
    abl = _runs(lambda i: [0.95 + noise[i] + 0.001 * (i % 2)], abl="lld")
    out = ablation_result("lld", abl, base, 0.003, 1, CELLS)
    assert out["status"] == "complete" and out["decision"]["adopt"] is True
    assert out["ttest"]["J"] == 15 and out["ttest"]["df"] == 14


def test_noise_band_is_std_across_model_seeds() -> None:
    runs = [_run(fs, f, [0.9 + 0.01 * ms], ms=ms) for ms in range(3) for fs, f in sorted(CELLS)]
    nb = noise_band(runs, 1)
    assert nb["value"] == pytest.approx(0.01)  # std(ddof=1) of [0.90, 0.91, 0.92]
    assert nb["std_ddof0"] == pytest.approx(np.std([0.9, 0.91, 0.92]))


def test_select_rule2_flags_smaller_close_candidate() -> None:
    def cfg(level: float, wall: float) -> list[dict[str, Any]]:
        return [
            _run(fs, f, [level + 0.01 * ((fs + f) % 2)], ms=ms, wall=wall)
            for fs, f in sorted(CELLS)
            for ms in range(3)
        ]

    cands = {"big": cfg(0.95, 100.0), "small": cfg(0.945, 80.0), "far": cfg(0.80, 50.0)}
    out = select(cands, None, {"big": 300, "small": 100, "far": 50})
    assert out["winner"] == "big"
    assert out["rule2_triggered"] == ["small"]
    assert out["top2_ttest"]["b"] == "small" and out["top2_ttest"]["J"] == 15


def test_verify_run_detects_mismatch(tmp_path: Any) -> None:
    gold = np.array([0, 1, 2, 3])
    probs = np.eye(12, dtype=np.float32)[[0, 1, 2, 0]][None]  # 1 epoch, 3/4 correct
    np.savez(tmp_path / "r.npz", probs=probs, gold=gold, ids=np.array(list("abcd")))
    good = {"run_id": "r", "epochs": [{"macro_f1": 0.0, "accuracy": 0.75}]}
    with pytest.raises(AssertionError):  # stored macro_f1 deliberately wrong
        verify_run(good, tmp_path)
    assert verify_run({"run_id": "missing", "epochs": []}, tmp_path) is None
