"""
Multi-axis selection post-selection stages: pure pieces only (tmp dirs, no dataset text, no GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # the phase4c / phase4e imports need torch

from intent_router import data as data_mod  # noqa: E402
from intent_router import phase4e as p5  # noqa: E402
from intent_router import phase4e_final as pf  # noqa: E402
from intent_router import robustness as rb  # noqa: E402
from intent_router import trackb_improve as ti  # noqa: E402

PCFG = {
    "factors": {"a1": {"id_randomize_p": 0.5}, "a2": {"oe_lambda": 0.5, "oe_batch_size": 16}},
    "final_v3": {
        "version": "v3", "model_dir": "outputs/final_model_v3",
        "results_dir": "results/phase4e/final_v3", "sdpa_timing_run": False,
        "oof_predictions_per_id": 3,
    },
}  # fmt: skip
FCFG = {
    "results_dir": "results/final", "figures_dir": "results/figures",
    "outputs_dir": "outputs/final", "model_dir": "outputs/final_model",
    "test_eval_log": "results/final/test_eval_log.jsonl",
    "selection_path": "results/bakeoff/selection.json",
    "train": {"model_name": "m", "stop_epoch": 9, "model_seed": 42, "lr": 3e-5},
    "determinism": {"repeat_run": True, "sdpa_timing_run": True},
    "wandb": {"project": "p", "group": "final", "enabled": True},
    "analysis": {"oof_config_id": "e5_lr3e-05", "oof_dir": "results/oof",
                 "oof_predictions_per_id": 9},
}  # fmt: skip


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


# ============================================================================ selection guard
def selection_dir(tmp_path: Path, chosen: str, smoke: bool = False, epoch: bool = True) -> Path:
    write(tmp_path / "selection.json", {"chosen": chosen, "smoke": smoke})
    if epoch and chosen != "v1":
        write(tmp_path / chosen / "epoch.json", {"e_star": 6, "smoke": False})
    return tmp_path


def test_resolve_selection_reads_candidate_and_epoch_from_files(tmp_path: Path) -> None:
    rec, e_star, sel = pf.resolve_selection(selection_dir(tmp_path, "a1a3"))
    assert rec.key == "a1a3" and rec.factors == ("a1", "a3") and e_star == 6
    assert sel["chosen"] == "a1a3"
    write(tmp_path / "a3" / "epoch.json", {"e_star": 4, "smoke": False})
    write(tmp_path / "selection.json", {"chosen": "a3", "smoke": False})
    rec, e_star, _ = pf.resolve_selection(tmp_path)  # nothing is hard-coded to a1a3 / 6
    assert rec.factors == ("a3",) and e_star == 4


def test_resolve_selection_refuses_v1_smoke_and_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no .*selection.json"):
        pf.resolve_selection(tmp_path)
    with pytest.raises(SystemExit, match="chose v1"):
        pf.resolve_selection(selection_dir(tmp_path, "v1"))
    with pytest.raises(SystemExit, match="smoke"):
        pf.resolve_selection(selection_dir(tmp_path, "a1a3", smoke=True))
    sub = tmp_path / "x"
    with pytest.raises(SystemExit, match="epoch.json"):
        pf.resolve_selection(selection_dir(sub, "a1a3", epoch=False))


def test_final_retrain_and_confirm_refuse_when_v1_was_chosen(tmp_path: Path) -> None:
    """The guard fires before anything is archived, trained or logged."""
    selection_dir(tmp_path / "res", "v1")
    env = SimpleNamespace(pcfg={"results_dir": str(tmp_path / "res")}, smoke=False)
    for stage in (pf.stage_final_retrain, pf.stage_confirm_final):
        with pytest.raises(SystemExit, match="chose v1"):
            stage(env)  # type: ignore[arg-type]
    assert sorted(p.name for p in (tmp_path / "res").iterdir()) == ["selection.json"]


# ============================================================================ config derivation
def test_derived_config_has_stop_epoch_model_dir_and_recipe_flags() -> None:
    rec = p5.recipe_for("a1a3")
    snapshot = json.dumps(FCFG, sort_keys=True)
    cfg = pf.derive_final_cfg(FCFG, rec, 6, PCFG, smoke=False)
    assert cfg["train"]["stop_epoch"] == 6
    assert cfg["train"]["id_randomize_p"] == 0.5  # A1 is a TrainConfig field; A3 acts on data
    assert "oe_lambda" not in cfg["train"]
    assert cfg["model_dir"] == "outputs/final_model_v3"
    assert cfg["determinism"]["sdpa_timing_run"] is False
    assert cfg["analysis"]["oof_predictions_per_id"] == 3
    assert cfg["analysis"]["oof_config_id"] == "p4e_a1a3"
    assert cfg["analysis"]["oof_dir"] == "results/phase4e/final_v3/oof"
    assert cfg["selection_path"] == "results/phase4e/final_v3/selection_v3.json"
    # the shared append-only log and the artifact locations are NOT redirected
    assert cfg["test_eval_log"] == FCFG["test_eval_log"]
    assert cfg["results_dir"] == "results/final" and cfg["outputs_dir"] == "outputs/final"
    assert json.dumps(FCFG, sort_keys=True) == snapshot  # input not mutated


def test_derived_config_smoke_keeps_v1_analysis_inputs() -> None:
    cfg = pf.derive_final_cfg(FCFG, p5.recipe_for("a3"), 6, PCFG, smoke=True)
    assert cfg["analysis"] == FCFG["analysis"] and cfg["selection_path"] == FCFG["selection_path"]
    assert "id_randomize_p" not in cfg["train"]  # a3 alone: no TrainConfig override


# ================================================================================== CV stand-ins
def oof_frame(seed: int, ids: list[str], gold: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"id": ids, "model_seed": seed, "gold": gold, "pred": gold,
                         "prob_0": 0.6, "prob_1": 0.4})  # fmt: skip


def test_stand_in_oof_needs_exactly_three_predictions_per_id() -> None:
    ids, gold = ["a", "b", "c"], [0, 1, 0]
    out = pf.build_stand_in_oof([oof_frame(s, ids, gold) for s in (0, 1, 2)], 3)
    assert len(out) == 9 and (out.groupby("id").size() == 3).all()
    with pytest.raises(ValueError, match="without 3 predictions"):
        pf.build_stand_in_oof([oof_frame(s, ids, gold) for s in (0, 1)], 3)
    bad = oof_frame(2, ids, [1, 1, 0])  # gold of id "a" disagrees between seeds
    with pytest.raises(ValueError, match="gold label differs"):
        pf.build_stand_in_oof([oof_frame(0, ids, gold), oof_frame(1, ids, gold), bad], 3)
    with pytest.raises(ValueError, match="different columns"):
        pf.build_stand_in_oof([oof_frame(0, ids, gold), oof_frame(1, ids, gold).iloc[:, :-1]], 2)


def test_stand_in_oof_is_readable_by_the_analysis_loader(tmp_path: Path) -> None:
    from intent_router.analysis import load_oof_mean

    ids = [f"id{i}" for i in range(4)]
    frames = []
    for s in range(3):
        f = oof_frame(s, ids, [0, 1, 0, 1])
        for k in range(2, 12):
            f[f"prob_{k}"] = 0.0
        frames.append(f)
    pf.build_stand_in_oof(frames, 3).to_csv(tmp_path / "c.csv", index=False)
    mean = load_oof_mean(tmp_path / "c.csv", 12, 3)
    assert len(mean) == 4


def test_selection_stand_in_summarises_the_fold_runs() -> None:
    f1 = np.array([0.9, 0.92, 0.94, 0.96])
    out = pf.selection_stand_in("c", 6, f1, f1 + 0.01, "n")["candidates"]["c"]
    assert out["chosen_epoch"] == 6 and out["n_runs"] == 4
    assert out["macro_f1_mean"] == pytest.approx(0.93)
    assert out["macro_f1_std"] == pytest.approx(np.std(f1, ddof=1))
    assert out["accuracy_mean"] == pytest.approx(0.94)


# ============================================================================== archive of v1
def make_results(root: Path, version: str) -> tuple[Path, Path, Path]:
    res, fig, out = root / "final", root / "figures", root / "final_out"
    write(res / "model_version.json", {"version": version, "model_fingerprint": "fp-" + version})
    write(res / "track_a.json", {"x": 1})
    (res / "test_eval_log.jsonl").write_text('{"call_type": "evaluation"}\n', encoding="utf-8")
    fig.mkdir(parents=True)
    (fig / "final_reliability.png").write_bytes(b"png")
    (fig / "class_counts.png").write_bytes(b"other")
    out.mkdir()
    (out / "features_logits.npz").write_bytes(b"npz")
    return res, fig, out


def test_archive_copies_v1_and_leaves_source_and_shared_log_in_place(tmp_path: Path) -> None:
    res, fig, out = make_results(tmp_path, "v1")
    dst_r, dst_o = tmp_path / "final_v1", tmp_path / "final_v1_out"
    assert pf.archive_v1(res, dst_r, fig, out, dst_o) == "archived"
    assert (dst_r / "track_a.json").exists() and (dst_o / "features_logits.npz").exists()
    assert (dst_r / "figures" / "final_reliability.png").exists()
    assert not (dst_r / "figures" / "class_counts.png").exists()  # only final_*.png
    assert not (dst_r / "test_eval_log.jsonl").exists()  # shared append-only log is not copied
    assert (res / "test_eval_log.jsonl").read_text(encoding="utf-8").strip()  # ...and stays put
    assert (res / "track_a.json").exists() and (out / "features_logits.npz").exists()  # copy


def test_archive_refuses_to_overwrite_an_existing_final_v1(tmp_path: Path) -> None:
    res, fig, out = make_results(tmp_path, "v1")
    dst_r, dst_o = tmp_path / "final_v1", tmp_path / "final_v1_out"
    write(dst_r / "model_version.json", {"version": "v2"})  # something else lives there
    with pytest.raises(SystemExit, match="not v1"):
        pf.archive_v1(res, dst_r, fig, out, dst_o)
    assert json.loads((dst_r / "model_version.json").read_text())["version"] == "v2"
    assert not dst_o.exists()
    write(dst_r / "model_version.json", {"version": "v1", "keep": True})  # a real v1 archive
    assert pf.archive_v1(res, dst_r, fig, out, dst_o) == "exists"
    assert json.loads((dst_r / "model_version.json").read_text())["keep"] is True  # untouched


def test_archive_refuses_a_non_v1_source_and_stale_partials(tmp_path: Path) -> None:
    res, fig, out = make_results(tmp_path, "v3")
    dst_r, dst_o = tmp_path / "final_v1", tmp_path / "final_v1_out"
    with pytest.raises(SystemExit, match="is not v1"):
        pf.archive_v1(res, dst_r, fig, out, dst_o)
    assert not dst_r.exists() and not dst_o.exists()
    write(res / "model_version.json", {"version": "v1"})
    dst_o.mkdir()  # outputs archive without a results archive: ambiguous, stop
    with pytest.raises(SystemExit, match="resolve manually"):
        pf.archive_v1(res, dst_r, fig, out, dst_o)


def test_training_done_only_for_a_finished_new_model(tmp_path: Path) -> None:
    res, model = tmp_path / "res", tmp_path / "model_v3"
    assert not pf.training_done(res, model, "fp-v1")  # no train_summary yet
    write(res / "train_summary.json", {"model_fingerprint": "fp-v1", "model_dir": str(model)})
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert not pf.training_done(res, model, "fp-v1")  # still v1's summary
    write(res / "train_summary.json", {"model_fingerprint": "fp-v3", "model_dir": str(model)})
    assert pf.training_done(res, model, "fp-v1")
    write(res / "train_summary.json", {"model_fingerprint": "fp-v3", "model_dir": "elsewhere"})
    assert not pf.training_done(res, model, "fp-v1")


# ================================================================================== reporting
def track_a(f1: float, fp: str) -> dict:
    return {"model_fingerprint": fp,
            "test": {"macro_f1": {"point": f1}, "accuracy": {"point": 0.9},
                     "selective": {"at_threshold": {"coverage": 0.9}}},
            "calibration": {"temperature": 1.1, "test": {"ece_after": 0.05}},
            "cv": {"oof_probability_averaged_426_rows": {"note": "over the 9 OOF"}}}  # fmt: skip


def test_pick_and_cv_note_patch(tmp_path: Path) -> None:
    p = pf.pick(track_a(0.95, "fp"))
    assert p["macro_f1"] == {"point": 0.95} and p["temperature"] == 1.1
    assert p["ece_test_after"] == 0.05 and p["selective_at_threshold"]["coverage"] == 0.9
    write(tmp_path / "track_a.json", track_a(0.95, "fp"))
    pf.patch_cv_note(tmp_path / "track_a.json", 3, [0, 1, 2])
    note = json.loads((tmp_path / "track_a.json").read_text())["cv"][
        "oof_probability_averaged_426_rows"
    ]["note"]
    assert "3 OOF predictions per id" in note and "[0, 1, 2]" in note


# ======================================================================= confirm comparator
def test_missing_v1_tables_lists_absent_comparator_runs(tmp_path: Path) -> None:
    cfg = {"loco_seed": 42, "confirm_classes": ["c1", "c2"], "headline_seeds": [42, 43],
           "wandb": {"group_prefix": "g"}}  # fmt: skip
    fcfg = {"train": {"model_name": "m", "lr": 1e-5}}
    ctx = SimpleNamespace(cfg=cfg, fcfg=fcfg, P=ti.Paths(tmp_path, tmp_path, tmp_path / "l", False))
    assert pf.missing_v1_tables(ctx) == [  # type: ignore[arg-type]
        "loco_c1_s42",
        "loco_c2_s42",
        "headline_s42",
        "headline_s43",
    ]
    (tmp_path / "scores").mkdir()
    for rid in ("loco_c1_s42", "headline_s43"):
        (tmp_path / "scores" / f"{rid}.csv").write_text("x", encoding="utf-8")
    assert pf.missing_v1_tables(ctx) == ["loco_c2_s42", "headline_s42"]  # type: ignore[arg-type]


# ============================================================ robustness sources restriction
def test_load_cohort_sources_restriction_is_optional_and_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = ["l1", "l2", "l3"]
    monkeypatch.setattr(data_mod, "label2id", lambda: {x: i for i, x in enumerate(labels)})
    frame = pd.DataFrame({
        "id": ["a", "b", "c"], "text": ["t"] * 3, "label": labels,
        "split": ["train", "val", "test"], "cv_fold_s0": [0, 1, -1],
    })  # fmt: skip
    pd.DataFrame({"id": ["a", "b", "c"], "lang": "en"}).to_csv(tmp_path / "eda.csv", index=False)
    monkeypatch.setattr(rb, "get_frame", lambda *a, **k: frame.copy())
    fcfg = {"data_path": "d", "splits_path": "s",
            "analysis": {"eda_rows_path": str(tmp_path / "eda.csv")}}  # fmt: skip
    base = {"fold_models": {"fold_seed_idx": 0}}
    both = rb.load_cohort(base, fcfg, False)
    assert sorted(both["source"].unique()) == ["oof", "test"] and len(both) == 3  # unchanged
    only = rb.load_cohort(base | {"sources": ["test"]}, fcfg, False)
    assert only["source"].unique().tolist() == ["test"] and only["id"].tolist() == ["c"]


def test_confirm_and_final_stages_are_registered() -> None:
    assert set(p5.FINAL_STAGES) <= set(p5.STAGES)  # `all` runs a fixed list that excludes them
