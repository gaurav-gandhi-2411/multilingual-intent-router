from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from intent_router import selection_check as sc
from intent_router.aggregate import summarize_config
from intent_router.cv import build_grid, main


def make_run(
    short: str, lr: float, curve: list[float], fold: int, fs: int = 0, ms: int = 0
) -> dict[str, Any]:
    return {
        "run_id": f"{short}_lr{lr:g}_fs{fs}_ms{ms}_f{fold}",
        "fold_seed_idx": fs,
        "model_seed": ms,
        "fold": fold,
        "n_train": 340,
        "n_eval": 86,
        "ablation": None,
        "epochs": [{"macro_f1": f, "accuracy": f} for f in curve],
        "wall_clock_s": 100.0,
        "peak_vram_mb": 5000.0,
        "gpu_exclusive": True,
        "config": {"model_short": short, "lr": lr, "model_name": short},
    }


def runs_for(short: str, lr: float, curve: list[float]) -> list[dict[str, Any]]:
    return [make_run(short, lr, curve, f) for f in range(5)]


def test_dry_run_lists_exactly_30(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # temp dirs (not live results/): the grid must be listed as entirely un-run
    argv = ["--stage", "selcheck", "--dry-run"]
    argv += ["--results-dir", str(tmp_path / "results"), "--outputs-dir", str(tmp_path / "out")]
    assert main(argv) == 0
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines() if "_ep30_" in ln]
    assert len(lines) == len(set(lines)) == 30
    assert "xlmr_lr5e-05_ep30_fs0_ms0_f0" in lines


def test_grid_ids_never_collide_with_bakeoff() -> None:
    cfg = yaml.safe_load(Path("configs/bakeoff.yaml").read_text())
    sel = build_grid("selcheck", cfg, cfg["selection_check"]["models"], {})
    assert all("_ep30_" in s.run_id for s in sel)
    conf = build_grid("selcheck_confirm", cfg, ["microsoft/mdeberta-v3-base"], {}, lrs=[7e-5])
    assert len(conf) == 45
    with pytest.raises(ValueError):
        build_grid("selcheck_confirm", cfg, ["microsoft/mdeberta-v3-base"], {})


def test_chosen_epoch_is_argmax_of_mean_curve() -> None:
    curve = [0.5, 0.9, 0.8, 0.95]
    s = summarize_config(runs_for("m", 5e-5, curve))
    assert s["chosen_epoch"] == 4
    assert s["macro_f1_mean"] == pytest.approx(0.95)


def test_decide_strict_threshold_and_edge_flags() -> None:
    summ = {
        "a_lr5e-05": summarize_config(runs_for("a", 5e-5, [0.9, 0.96])),  # beats, ep 2
        "b_lr7e-05": summarize_config(runs_for("b", 7e-5, [0.9, 0.9441])),  # tie: not > comp
        "c_lr0.0001": summarize_config(runs_for("c", 1e-4, [0.8, 0.85])),
    }
    lrs = {"a_lr5e-05": 5e-5, "b_lr7e-05": 7e-5, "c_lr0.0001": 1e-4}
    d = sc.decide(summ, lrs, 0.9441)
    assert d["confirm"] == ["a_lr5e-05"]
    assert d["edge_lr_1e-4"] == ["c_lr0.0001"]
    assert d["edge_epoch_30"] == []
    summ["d_lr5e-05"] = summarize_config(runs_for("d", 5e-5, [0.1] * 29 + [0.2]))
    assert sc.decide(summ, lrs, 0.9441)["edge_epoch_30"] == ["d_lr5e-05"]


def test_comparator_assertion_fires_on_mismatch() -> None:
    runs = runs_for("e5", 3e-5, [0.9, 0.95])
    with pytest.raises(AssertionError):
        sc.comparator(runs)
    ok = runs_for("e5", 3e-5, [0.9, sc.COMPARATOR_EXPECTED])
    assert sc.comparator(ok)["exact_score"] == pytest.approx(sc.COMPARATOR_EXPECTED)


def test_s0_runs_filters_other_seeds() -> None:
    runs = [make_run("m", 1e-4, [0.5], 0), make_run("m", 1e-4, [0.5], 0, fs=1)]
    assert len(sc.s0_runs(runs)) == 1


def test_confirm_selection_applies_rules() -> None:
    def full(short: str, lr: float, curve: list[float]) -> list[dict[str, Any]]:
        return [
            make_run(short, lr, [c + 0.001 * ((fs + ms + f) % 3) for c in curve], f, fs, ms)
            for fs in range(3)
            for ms in range(3)
            for f in range(5)
        ]

    e5 = full("e5", 3e-5, [0.90, 0.94])
    cand = {"xlmr_lr7e-05": full("xlmr", 7e-5, [0.90, 0.97])}
    out = sc.confirm_selection(cand, e5, {"e5_lr3e-05": 1, "xlmr_lr7e-05": 2}, 45)
    assert out["winner"] == "xlmr_lr7e-05"
    assert out["top2_ttest"]["J"] == 15
    with pytest.raises(AssertionError):
        sc.confirm_selection(cand, e5[:5], {}, 45)
