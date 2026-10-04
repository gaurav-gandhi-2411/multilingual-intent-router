"""Multi-axis selection: pure pieces only (synthetic arrays, no dataset text, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # evaluate.fit_temperature and the phase4c imports need torch

from intent_router import phase4c as p4  # noqa: E402
from intent_router import phase4e as p5  # noqa: E402
from intent_router import trackb_improve as ti  # noqa: E402
from intent_router.evaluate import ece_bins, softmax, threshold_at_retention  # noqa: E402

TH = {
    "a_max_drop": 0.010, "b_max_auroc_drop": 0.010, "b_max_rej95_drop": 0.03,
    "c_max_flip_increase": 0.02, "d_max_agreement_drop": 0.010, "e_max_ece_increase": 0.02,
}  # fmt: skip
S, F, N_CLS, N_ROWS, K, N_CLUS, N_IDS, L = 3, 5, 2, 60, 4, 20, 30, 4


# ============================================================================ epoch selection
def test_epoch_ties_go_to_the_earliest() -> None:
    curves = np.array([[0.1, 0.9, 0.5, 0.9], [0.1, 0.9, 0.7, 0.9]])
    out = p5.epoch_summary(curves)
    assert out["e_star"] == 2 and out["own_argmax"] == 2  # epochs 2 and 4 tie exactly
    assert out["f1_at_e_star"] == pytest.approx(0.9)


def test_epoch_uses_pooled_mean_not_a_single_run() -> None:
    curves = np.array([[0.5, 0.6, 0.1], [0.5, 0.2, 0.9]])  # means 0.5, 0.4, 0.5 -> tie 1 vs 3
    assert p5.epoch_summary(curves)["e_star"] == 1
    curves = np.array([[0.5, 0.6, 0.7], [0.5, 0.2, 0.9]])  # means 0.5, 0.4, 0.8
    assert p5.epoch_summary(curves)["e_star"] == 3


def test_v1_fixed_epoch_9_keeps_its_own_argmax_and_value_recorded() -> None:
    mean = np.linspace(0.0, 0.9, 20)
    mean[7] = 0.95  # own argmax = epoch 8
    out = p5.epoch_summary(np.stack([mean, mean]), fixed_epoch=9)
    assert out["e_star"] == 9 and out["fixed"] is True
    assert out["own_argmax"] == 8
    assert out["f1_at_e_star"] == pytest.approx(mean[8])
    assert out["f1_at_own_argmax"] == pytest.approx(0.95)
    with pytest.raises(ValueError):
        p5.epoch_summary(np.stack([mean]), fixed_epoch=21)


# ================================================================================ selection rule
def row(key: str, eligible: bool, improved: tuple[str, ...], n_factors: int, a: float = 0.0):
    return p5.SelRow(key, eligible, improved, n_factors, a)


def test_ineligible_candidate_with_many_improvements_loses() -> None:
    out = p5.select_candidate([
        row("a1a3", False, ("a", "b", "c", "d", "e"), 2, 0.05),
        row("a3", True, ("d",), 1, 0.0),
    ])  # fmt: skip
    assert out["chosen"] == "a3"
    assert out["path"][0]["kept"] == ["a3"]


def test_most_improved_axes_beats_fewer_factors() -> None:
    out = p5.select_candidate([row("a3", True, ("d",), 1), row("a1a3", True, ("c", "d"), 2)])
    assert out["chosen"] == "a1a3"


def test_ties_go_to_fewer_factors_then_higher_axis_a() -> None:
    rows = [row("a1a3", True, ("c", "d"), 2, 0.09), row("a1", True, ("c", "d"), 1, 0.01)]
    out = p5.select_candidate(rows)
    assert out["chosen"] == "a1"
    assert [s["step"] for s in out["path"]][1:] == ["most_improved_axes", "fewer_factors",
                                                    "higher_axis_a"]  # fmt: skip
    rows = [row("a3", True, ("c",), 1, 0.01), row("a1", True, ("d",), 1, 0.03)]
    out = p5.select_candidate(rows)
    assert out["chosen"] == "a1" and out["unresolved_tie"] is False
    rows = [row("a3", True, ("c",), 1, 0.02), row("a1", True, ("d",), 1, 0.02)]
    out = p5.select_candidate(rows)
    assert (
        out["chosen"] == "a3" and out["unresolved_tie"] is True
    )  # exact tie: input order, flagged


def test_nobody_eligible_and_improved_keeps_v1() -> None:
    assert p5.select_candidate([row("a1", False, ("a",), 1)])["chosen"] == "v1"
    assert p5.select_candidate([row("a1", True, (), 1)])["chosen"] == "v1"  # eligible, not improved
    assert p5.select_candidate([])["chosen"] == "v1"
    assert p5.select_candidate([row("v1", True, ("a",), 0)])["chosen"] == "v1"  # ref never wins


# ============================================================================= thresholds (edges)
def test_axis_a_threshold_boundary() -> None:
    assert p5.eligible_drop(0.9341 - 0.9441, 0.010)  # exactly -0.010 (float noise) is eligible
    assert p5.eligible_drop(-0.010, 0.010)
    assert not p5.eligible_drop(-0.0101, 0.010)
    assert not p5.eligible_drop(float("nan"), 0.010)  # fails closed


def test_flip_and_ece_threshold_boundary() -> None:
    assert p5.eligible_rise(0.10 + 0.02, 0.10, 0.02)
    assert p5.eligible_rise(0.12, 0.10, 0.02)  # +0.02 exactly
    assert not p5.eligible_rise(0.1201, 0.10, 0.02)
    assert not p5.eligible_rise(float("nan"), 0.10, 0.02)


def test_rejection_threshold_boundary() -> None:
    assert p5.eligible_drop(0.60 - 0.63, 0.03)  # -0.03 exactly
    assert not p5.eligible_drop(-0.0301, 0.03)


# ========================================================================== synthetic candidates
def make_unit(rng: np.random.Generator, shift: float = 0.0, cal_shift: float = 0.0) -> ti.BootUnit:
    known = rng.normal(1.0 + shift, 1.0, 40)
    unknown = rng.normal(-1.0, 1.0, 30)
    cal = rng.normal(1.0 + cal_shift, 1.0, 40)
    thr = {0.95: threshold_at_retention(cal, 0.95)}
    return ti.BootUnit(known, unknown, np.zeros(30, dtype=bool), thr)


def make_cd(key: str, f1: float = 0.9, flip: float = 0.3, agree: float = 0.8, n_factors: int = 0,
            shift: float = 0.0, sharp: float = 3.0, same_rows: int = 7) -> p5.CandData:  # fmt: skip
    rng = np.random.default_rng(same_rows)  # identical row structure for every candidate
    gold = rng.integers(0, K, N_ROWS)
    probs = []
    for s in range(S):
        r2 = np.random.default_rng(100 + s + len(key))
        logits = r2.normal(size=(N_ROWS, K))
        logits[np.arange(N_ROWS), gold] += sharp * (r2.random(N_ROWS) < 0.85)
        probs.append(softmax(logits))
    units = [[make_unit(np.random.default_rng(10 * c + s), shift) for s in range(S)]
             for c in range(N_CLS)]  # fmt: skip
    inst = np.full(N_CLUS, 4.0)
    r3 = np.random.default_rng(3 + len(key))
    flips = np.array([(r3.random(N_CLUS) < flip) * inst for _ in range(S)])
    kept = (np.random.default_rng(5).random((N_IDS, L)) < 0.7).astype(float)
    ag = (np.random.default_rng(6 + len(key)).random((S, N_IDS, L)) < agree).astype(float)
    keys = {"clean": ("ids",), "swap": ("ids",), "mt": ("ids",), "tb": ("x", "y")}
    return p5.CandData(key, n_factors, 9, np.full((S, F), f1), units, flips, inst, ag, kept,
                       np.array(probs), gold, keys)  # fmt: skip


def run_axes(cands: dict[str, p5.CandData], n_boot: int = 200) -> dict:
    return p5.compute_axes(cands, TH, n_boot, 42, 0.95, 15, "v1")


def test_identical_candidate_has_zero_delta_and_is_not_improved() -> None:
    v1 = make_cd("v1")
    out = run_axes({"v1": v1, "twin": make_cd("v1")})["candidates"]["twin"]
    assert out["eligible"] and out["improved_axes"] == []
    for a in ("a", "c", "d", "e"):
        assert out["axes"][a]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert out["axes"]["b"]["auroc"]["delta"] == pytest.approx(0.0, abs=1e-12)


def test_axis_a_resamples_matched_seed_fold_pairs_and_flags_improvement() -> None:
    idx = p4.boot_indices(S * F, 200, 42)
    assert idx.shape == (200, S * F)  # 15 matched (seed, fold) pairs
    ref, cd = make_cd("v1", f1=0.90), make_cd("c1", f1=0.92)
    rec = p5.axis_a(ref.f1, cd, idx, TH, 0.95)
    assert rec["delta"] == pytest.approx(0.02)
    assert rec["lo"] == pytest.approx(0.02) and rec["hi"] == pytest.approx(0.02)  # constant delta
    assert rec["improved"] and rec["eligible"]
    worse = p5.axis_a(ref.f1, make_cd("c2", f1=0.8899), idx, TH, 0.95)
    assert not worse["eligible"] and not worse["improved"]  # -0.0101 is below the -0.010 floor


def test_flip_samples_average_seeds_inside_each_resample() -> None:
    flips = np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    inst = np.array([2.0, 2.0, 2.0, 2.0])
    idx = np.array([[0, 1], [2, 2], [3, 0]])
    got = p5.flip_rate_samples(flips, inst, idx)
    assert got.shape == (3,)
    manual = [
        np.mean([(1 + 0) / 4, (0 + 0) / 4]),
        np.mean([(2 + 2) / 4, (1 + 1) / 4]),
        np.mean([(1 + 1) / 4, (1 + 0) / 4]),
    ]
    assert got == pytest.approx(manual)
    assert p5.flip_rate_point(flips, inst) == pytest.approx(np.mean([4 / 8, 2 / 8]))


def test_agreement_samples_use_only_kept_rows_and_average_seeds_and_languages() -> None:
    agree = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]])
    kept = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    idx = np.array([[0, 1, 2]])
    vals = []
    for s in range(2):
        vals.append((agree[s, 0, 0] + agree[s, 2, 0]) / 2)  # lang 0 keeps ids 0 and 2
        vals.append((agree[s, 0, 1] + agree[s, 1, 1]) / 2)  # lang 1 keeps ids 0 and 1
    assert p5.agreement_samples(agree, kept, idx)[0] == pytest.approx(np.mean(vals))
    assert p5.agreement_point(agree, kept) == pytest.approx(np.mean(vals))


def test_axis_b_improved_if_either_sub_metric_improves() -> None:
    # Perfect separation everywhere (AUROC = 1 for both), but the candidate's calibration scores
    # give a threshold above every unknown, so only rejection@95 improves.
    def unit(cal_lo: float) -> ti.BootUnit:
        known, unknown = np.linspace(10, 11, 40), np.linspace(0, 1, 30)
        cal = np.linspace(cal_lo, cal_lo + 1, 40)
        return ti.BootUnit(known, unknown, np.zeros(30, dtype=bool),
                           {0.95: threshold_at_retention(cal, 0.95)})  # fmt: skip

    ref, cd = make_cd("v1"), make_cd("c1")
    ref.tb_units = [[unit(-5.0) for _ in range(S)] for _ in range(N_CLS)]  # t < unknowns: rej 0
    cd.tb_units = [[unit(9.0) for _ in range(S)] for _ in range(N_CLS)]  # t > unknowns: rej 1
    b = run_axes({"v1": ref, "c1": cd})["candidates"]["c1"]["axes"]["b"]
    assert b["auroc"]["delta"] == pytest.approx(0.0) and not b["auroc"]["improved"]
    assert b["rej95"]["delta"] == pytest.approx(1.0) and b["rej95"]["improved"]
    assert b["improved"] and b["eligible"]


def test_axis_b_eligibility_needs_both_sub_metrics() -> None:
    ref, cd = make_cd("v1"), make_cd("c1", shift=-1.5)  # known scores drop => AUROC collapses
    b = run_axes({"v1": ref, "c1": cd})["candidates"]["c1"]["axes"]["b"]
    assert b["auroc"]["delta"] < -0.010 and not b["eligible"]


def test_resamples_are_shared_across_candidates_and_deterministic() -> None:
    cands = {"v1": make_cd("v1"), "c1": make_cd("c1", flip=0.1)}
    a, b = run_axes(cands), run_axes(cands)
    assert a["candidates"]["c1"]["axes"]["c"] == b["candidates"]["c1"]["axes"]["c"]
    assert a["candidates"]["c1"]["axes"]["c"]["improvement"] > 0  # fewer flips = a reduction


def test_misaligned_rows_are_refused() -> None:
    other = make_cd("c1")
    other.keys = {**other.keys, "clean": ("different",)}
    with pytest.raises(ValueError, match="not row-aligned"):
        run_axes({"v1": make_cd("v1"), "c1": other})


# ================================================================================ calibration
def test_temperature_recovers_a_known_value_and_ece_is_zero() -> None:
    # K=2, margin m=2, 8 of 10 rows correct: NLL is minimal where sigmoid(m/T) = 0.8
    # => T = m / logit(0.8) = 2 / ln 4.
    probs = np.tile(softmax(np.array([[2.0, 0.0]])), (10, 1))
    gold = np.array([0] * 8 + [1] * 2)
    cal = p5.fit_calibration(probs, gold, 15)
    assert cal["T"] == pytest.approx(2.0 / np.log(4.0), rel=1e-4)
    assert cal["conf"] == pytest.approx(np.full(10, 0.8), abs=1e-4)
    assert cal["ece"] == pytest.approx(0.0, abs=1e-4)  # confidence 0.8 == accuracy 0.8


def test_ece_samples_match_the_reference_ece_on_the_identity_resample() -> None:
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(200, 5)) * 2)
    gold = rng.integers(0, 5, 200)
    ref, _ = ece_bins(probs, gold, 15)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == gold
    got = p5.ece_samples(conf, correct, np.arange(200)[None, :], 15)
    assert got[0] == pytest.approx(ref)
    # a doubled resample of the same rows has the same ECE
    assert p5.ece_samples(conf, correct, np.r_[np.arange(200), np.arange(200)][None, :], 15)[
        0
    ] == pytest.approx(ref)


# ==================================================================== deletion guard + helpers
def test_delete_fold_model_only_inside_phase4e_fold_models(tmp_path: Path) -> None:
    good = tmp_path / "outputs" / "phase4e" / "v1" / "s0" / "fold_models" / "fold0"
    good.mkdir(parents=True)
    (good / "model.safetensors").write_text("x")
    assert p5.delete_fold_model(good) is True
    assert not good.exists() and good.parent.exists()
    assert p5.delete_fold_model(good) is False  # already gone
    for bad in (
        tmp_path / "outputs" / "phase4c" / "a1" / "fold_models" / "fold0",  # wrong project
        tmp_path / "outputs" / "phase4e" / "v1" / "s0" / "trackb_arrays",  # not a fold model
        tmp_path / "outputs" / "phase4e" / "v1" / "s0" / "fold_models",  # the parent itself
    ):
        bad.mkdir(parents=True, exist_ok=True)
        with pytest.raises(AssertionError):
            p5.delete_fold_model(bad)
        assert bad.exists()


def test_recipes_and_keys() -> None:
    assert p5.recipe_for("v1").factors == ()
    assert p5.recipe_for("a1a3").factors == ("a1", "a3") and p5.recipe_for("a1a3").key == "a1a3"
    assert p5.recipe_for("a3").factors == ("a3",)
    with pytest.raises(KeyError):
        p5.recipe_for("comb")


def test_config_diff_ignores_wandb_fields_only() -> None:
    a = {"lr": 3e-5, "id_randomize_p": 0.5, "wandb_group": "phase4c-a1", "wandb_project": "x"}
    b = {"lr": 3e-5, "id_randomize_p": 0.5, "wandb_group": "phase4e-a1", "wandb_project": "x"}
    assert p5.config_diff(a, b) == {}
    assert p5.config_diff(a, b | {"id_randomize_p": 0.0}) == {"id_randomize_p": [0.5, 0.0]}
    assert "lr" in p5.config_diff(a, {k: v for k, v in b.items() if k != "lr"})


def test_class_seeds_are_deterministic_and_distinct() -> None:
    assert p5.class_seeds(5, 42) == p5.class_seeds(5, 42)
    assert len(set(p5.class_seeds(5, 42))) == 5


def test_write_json_turns_nan_into_null(tmp_path: Path) -> None:
    import json

    p5.write_json(tmp_path / "x.json", {"a": float("nan"), "b": [np.float64("inf"), 1.5]})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": [None, 1.5]}


def test_selection_from_saved_axes_and_v1_epoch_sensitivity() -> None:
    def ax(elig: bool, imp: bool, delta: float = 0.0) -> dict:
        return {"eligible": elig, "improved": imp, "delta": delta}

    def cand(a_main: dict, a_alt: dict) -> dict:
        axes = {"a": a_main, "b": ax(True, False), "c": ax(True, True), "d": ax(True, False),
                "e": ax(True, False)}  # fmt: skip
        return {"n_factors": 1, "axes": axes, "sensitivity_a_v1_at_own_argmax": a_alt}

    axes = {"candidates": {"a3": cand(ax(True, False, 0.0), ax(False, False, 0.0))}}
    assert p5.select_candidate(p5.rows_from_axes(axes))["chosen"] == "a3"  # improves c only
    alt = p5.select_candidate(p5.rows_from_axes(axes, v1_alt=True))
    assert alt["chosen"] == "v1"  # against v1's own-argmax epoch the candidate is ineligible
