from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from intent_router import phase6a_diag as pd6
from intent_router.evaluate import threshold_at_retention
from intent_router.ood import build_holdout_sets

SRC = Path(__file__).resolve().parents[1] / "src" / "intent_router"


# ------------------------------------------------------------------ neutralisation / counts
def test_neutralize_ids_rewrites_every_prefix_and_keeps_digits() -> None:
    t = "ship LD-123 and PO-45 for ABCD-9 then xy-7 plus REF-1"
    assert pd6.neutralize_ids(t) == "ship REF-123 and REF-45 for REF-9 then xy-7 plus REF-1"


def test_neutralize_ids_leaves_idless_text_and_is_idempotent() -> None:
    assert (
        pd6.neutralize_ids("no identifiers here, A-1 or ABCDE")
        == "no identifiers here, A-1 or ABCDE"
    )
    once = pd6.neutralize_ids("LD-1 PO-2")
    assert pd6.neutralize_ids(once) == once


def test_neutralize_ids_pattern_is_literal_without_word_boundary() -> None:
    # the pre-registered `[A-Z]{2,4}-\d+` verbatim: a 5-letter prefix is matched on its last
    # four letters
    assert pd6.neutralize_ids("ABCDE-12") == "AREF-12"


def test_id_prefix_counts_counts_rows_once_per_prefix() -> None:
    c = pd6.id_prefix_counts(["LD-1 and LD-2", "PO-9", "nothing", "LD-5 PO-6"])
    assert c["n_rows"] == 4 and c["n_rows_with_id"] == 3 and c["n_id_occurrences"] == 5
    assert c["rows_by_prefix"] == {"LD": 2, "PO": 2}


def test_id_prefix_counts_empty() -> None:
    c = pd6.id_prefix_counts([])
    assert c["n_rows"] == 0 and c["frac_rows_with_id"] is None and c["rows_by_prefix"] == {}


# ------------------------------------------------------------------------- grouped CV / probe
def _toy(n_groups: int = 60, per_group: int = 3, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    # unknown iff the group index is a multiple of 4 (groups are never split across classes)
    y = (groups % 4 == 0).astype(int)
    return rng, groups, y


def test_grouped_cv_folds_has_no_group_leakage_and_all_folds_non_empty() -> None:
    _, groups, y = _toy()
    folds = pd6.grouped_cv_folds(y, groups, 5, 42)
    assert set(folds.tolist()) == set(range(5))
    for g in np.unique(groups):
        assert len(set(folds[groups == g].tolist())) == 1  # a group lives in exactly one fold
    for k in range(5):
        assert set(y[folds == k].tolist()) == {0, 1}  # both classes in every held-out fold


def test_grouped_cv_folds_is_deterministic() -> None:
    _, groups, y = _toy()
    assert np.array_equal(
        pd6.grouped_cv_folds(y, groups, 5, 42), pd6.grouped_cv_folds(y, groups, 5, 42)
    )


def test_probe_auroc_is_near_one_on_separable_and_near_half_on_noise() -> None:
    rng, groups, y = _toy(n_groups=80)
    folds = pd6.grouped_cv_folds(y, groups, 5, 42)
    sep = rng.normal(size=(len(y), 12)) + 4.0 * y[:, None] * np.eye(12)[0]
    noise = rng.normal(size=(len(y), 12))
    s_sep = pd6.probe_oof_scores(sep, y, folds)
    s_noise = pd6.probe_oof_scores(noise, y, folds)
    assert np.isfinite(s_sep).all()
    assert pd6.auroc_pos(s_sep[y == 1], s_sep[y == 0]) > 0.95
    assert 0.3 < pd6.auroc_pos(s_noise[y == 1], s_noise[y == 0]) < 0.7


def test_auroc_pos_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, size=200).astype(float)
    y = rng.integers(0, 2, size=200)
    assert pd6.auroc_pos(s[y == 1], s[y == 0]) == pytest.approx(roc_auc_score(y, s))


def test_group_bootstrap_auroc_shapes_point_and_ci() -> None:
    rng, groups, y = _toy(n_groups=80)
    folds = pd6.grouped_cv_folds(y, groups, 5, 42)
    s = rng.normal(size=len(y)) + 1.5 * y
    r = pd6.group_bootstrap_auroc(y, s, folds, groups, n_boot=300, seed=42)
    assert r["auroc_pooled_oof"] == pytest.approx(roc_auc_score(y, s))
    assert len(r["per_fold_auroc"]) == 5 and r["n_groups"] == 80 and r["n_boot"] == 300
    assert r["auroc_mean_of_folds"] == pytest.approx(np.nanmean(r["per_fold_auroc"]))
    for ci_key, pt_key in (("auroc_pooled_oof_ci", "auroc_pooled_oof"),
                           ("auroc_mean_of_folds_ci", "auroc_mean_of_folds")):  # fmt: skip
        lo, hi = r[ci_key]
        assert lo < r[pt_key] < hi and 0.0 <= lo <= hi <= 1.0
    again = pd6.group_bootstrap_auroc(y, s, folds, groups, n_boot=300, seed=42)
    assert again == r  # deterministic


def test_group_bootstrap_resamples_groups_not_rows() -> None:
    # perfectly separable scores: every resample has AUROC exactly 1 -> a zero-width CI
    _, groups, y = _toy(n_groups=40)
    folds = pd6.grouped_cv_folds(y, groups, 5, 42)
    r = pd6.group_bootstrap_auroc(y, y.astype(float), folds, groups, n_boot=100, seed=1)
    assert r["auroc_pooled_oof_ci"] == [1.0, 1.0]


# ---------------------------------------------------------------------------- N2 / reversal
def _auc(mean: float, pooled: float) -> dict[str, float]:
    return {"auroc_mean_of_folds": mean, "auroc_pooled_oof": pooled}


def test_n2_decision_boundary_and_disagreement() -> None:
    d = pd6.n2_decision(_auc(0.80, 0.80), _auc(0.82, 0.82), 0.02)
    assert d["run_n2"] is True and d["oracle"] is True  # exactly +0.02 passes (>=)
    assert d["rule"] == "frozen e5-large headline oracle AUROC >= frozen e5-base + 0.02"
    d = pd6.n2_decision(_auc(0.80, 0.80), _auc(0.8199, 0.8199), 0.02)
    assert d["run_n2"] is False and d["primary_and_pooled_agree"] is True
    d = pd6.n2_decision(_auc(0.80, 0.80), _auc(0.83, 0.81), 0.02)
    assert d["run_n2"] is True and d["run_n2_pooled_variant"] is False
    assert d["primary_and_pooled_agree"] is False


def _head(v1: tuple[float, float], v3: tuple[float, float]) -> dict[str, dict[str, float]]:
    return {
        "v1": {"strict_rej95": v1[0], "auroc": v1[1]},
        "v3": {"strict_rej95": v3[0], "auroc": v3[1]},
    }


def test_ranking_reversal_none_when_v1_wins_in_both_modes() -> None:
    r = pd6.ranking_reversal(_head((0.34, 0.87), (0.21, 0.83)), _head((0.30, 0.86), (0.25, 0.85)))
    assert r["ranking_reversal"] is False and r["ranking_reversal_on_both_metrics"] is False


def test_ranking_reversal_flags_either_metric_and_both() -> None:
    raw = _head((0.34, 0.87), (0.21, 0.83))
    one = pd6.ranking_reversal(raw, _head((0.30, 0.86), (0.40, 0.85)))  # only rej95 flips
    assert one["ranking_reversal"] is True and one["ranking_reversal_on_both_metrics"] is False
    assert (
        one["by_metric"]["strict_rej95"]["reversed"] and not one["by_metric"]["auroc"]["reversed"]
    )
    both = pd6.ranking_reversal(raw, _head((0.30, 0.80), (0.40, 0.85)))
    assert both["ranking_reversal"] is True and both["ranking_reversal_on_both_metrics"] is True


def test_ranking_reversal_tie_is_not_a_flip() -> None:
    r = pd6.ranking_reversal(_head((0.34, 0.87), (0.21, 0.83)), _head((0.30, 0.86), (0.30, 0.86)))
    assert r["ranking_reversal"] is False
    assert r["by_metric"]["strict_rej95"]["neutral"]["winner"] == "tie"


# ------------------------------------------------------------------------------ slope fit
def test_fit_slope_recovers_exact_line() -> None:
    x = np.log2([0.25, 0.5, 0.75, 1.0])
    slope, intercept = pd6.fit_slope(x, 0.9 + 0.03 * x)
    assert slope == pytest.approx(0.03) and intercept == pytest.approx(0.9)


def test_slope_seed_bootstrap_exact_line_has_zero_width_ci() -> None:
    fr = [0.25, 0.5, 0.75, 1.0]
    base = 0.8 + 0.05 * np.log2(fr)
    vals = np.stack([base, base, base], axis=1)  # [4 fractions, 3 identical seeds]
    r = pd6.slope_seed_bootstrap(vals, fr, 200, 42)
    assert r["slope_per_doubling"] == pytest.approx(0.05)
    assert r["ci"][0] == pytest.approx(0.05) and r["ci"][1] == pytest.approx(0.05)
    assert r["per_seed_slope"] == pytest.approx([0.05] * 3) and r["n_seeds"] == 3


def test_slope_seed_bootstrap_ci_brackets_point_with_noisy_seeds() -> None:
    fr = [0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(0)
    vals = 0.8 + 0.05 * np.log2(fr)[:, None] + rng.normal(0, 0.01, size=(4, 5))
    r = pd6.slope_seed_bootstrap(vals, fr, 500, 42)
    assert r["ci"][0] <= r["slope_per_doubling"] <= r["ci"][1] and r["ci"][0] < r["ci"][1]


# ------------------------------------------------------------------------- subsampling
def _train_frame() -> pd.DataFrame:
    rows = [(f"{lab}-{i:02d}", lab) for lab in ("a", "b", "c") for i in range(20)]
    return pd.DataFrame(rows, columns=["id", "label"]).assign(text="x", split="train")


def test_subsample_is_class_stratified_nested_and_deterministic() -> None:
    tr = _train_frame()
    s25, s50, s100 = (pd6.subsample_train_rows(tr, f, 42) for f in (0.25, 0.5, 1.0))
    assert s25["label"].value_counts().to_dict() == {"a": 5, "b": 5, "c": 5}
    assert s50["label"].value_counts().to_dict() == {"a": 10, "b": 10, "c": 10}
    assert set(s25["id"]) <= set(s50["id"]) <= set(s100["id"]) and len(s100) == 60
    assert s25.equals(pd6.subsample_train_rows(tr, 0.25, 42))
    assert set(s25["id"]) != set(pd6.subsample_train_rows(tr, 0.25, 43)["id"])  # seed matters


def test_subsample_keeps_at_least_one_row_per_class_and_rejects_bad_fraction() -> None:
    tr = pd.concat(
        [
            _train_frame(),
            pd.DataFrame({"id": ["z-0"], "label": ["z"], "text": "x", "split": "train"}),
        ]
    )
    assert (pd6.subsample_train_rows(tr, 0.1, 42)["label"] == "z").sum() == 1
    with pytest.raises(ValueError):
        pd6.subsample_train_rows(tr, 0.0, 42)


# -------------------------------------------------------------------------- D3 thresholds
def test_threshold_bootstrap_matches_threshold_at_retention_per_resample() -> None:
    rng = np.random.default_rng(3)
    cal = rng.normal(size=101)
    ke, un = rng.normal(size=50), rng.normal(loc=-2.0, size=40)
    b = pd6.threshold_bootstrap(cal, ke, un, 0.95, n_boot=50, seed=7)
    assert (
        b["threshold"].shape == b["retention_known"].shape == b["rejection_unknown"].shape == (50,)
    )
    idx = np.random.default_rng(7).integers(0, len(cal), size=(50, len(cal)))
    for i in (0, 17, 49):
        t = threshold_at_retention(cal[idx[i]], 0.95)
        assert b["threshold"][i] == t
        assert b["retention_known"][i] == np.mean(ke >= t)
        assert b["rejection_unknown"][i] == np.mean(un < t)


def test_summarize_keys_and_order() -> None:
    s = pd6.summarize(np.arange(101, dtype=float))
    assert (
        s["mean"] == 50.0 and s["p2_5"] == pytest.approx(2.5) and s["p97_5"] == pytest.approx(97.5)
    )
    assert s["sd"] == pytest.approx(np.arange(101).std(ddof=1))


def test_crossfit_maha_scores_is_complete_and_separates_far_points() -> None:
    rng = np.random.default_rng(0)
    n_cls, per, d = 3, 30, 6
    centers = rng.normal(scale=6.0, size=(n_cls, d))
    y = np.repeat(np.arange(n_cls), per)
    x = centers[y] + rng.normal(size=(len(y), d))
    groups = np.arange(len(y))
    oof = pd6.crossfit_maha_scores(x, y, groups, n_cls, 5, 42)
    assert oof.shape == (len(y),) and np.isfinite(oof).all() and (oof <= 0).all()
    far = pd6.crossfit_maha_scores(
        np.vstack([x, x[:5] + 50.0]),
        np.r_[y, y[:5]],
        np.r_[groups, groups[:5] + 1000],
        n_cls,
        5,
        42,
    )
    assert far[-5:].mean() < oof.mean()  # displaced rows score lower (less known)


# --------------------------------------------------------------------------- pooling
def test_pool_hidden_mean_ignores_padding_and_is_normalised() -> None:
    torch = pytest.importorskip("torch")
    h = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    e = pd6.pool_hidden(h, mask, "mean")
    assert torch.allclose(e, torch.tensor([[1.0, 0.0]]))  # mean of the two real tokens, unit norm
    c = pd6.pool_hidden(h, mask, "cls")
    assert torch.allclose(c, torch.tensor([[1.0, 0.0]]))


def test_pool_hidden_dense_tanh_and_errors() -> None:
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(2, 2)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(2) * 10.0)
        lin.bias.zero_()
    h = torch.tensor([[[0.3, -0.4], [0.0, 0.0]]])
    e = pd6.pool_hidden(h, torch.ones(1, 2), "cls_dense_tanh", lin)
    want = torch.nn.functional.normalize(torch.tanh(torch.tensor([[3.0, -4.0]])), dim=-1)
    assert torch.allclose(e, want, atol=1e-6)
    with pytest.raises(ValueError):
        pd6.pool_hidden(h, torch.ones(1, 2), "cls_dense_tanh")
    with pytest.raises(ValueError):
        pd6.pool_hidden(h, torch.ones(1, 2), "bogus")


# ------------------------------------------------------------------------ run ids / tables
def test_make_run_id_matches_store_naming() -> None:
    assert pd6.make_run_id("v1", "headline", 42) == "v1_headline_s42"
    assert pd6.make_run_id("a1a3", "loco", 42, "orders") == "a1a3_loco_orders_s42"
    assert pd6.make_run_id("v1", "headline", 43, None, 0.25) == "v1_headline_f025_s43"
    assert (
        pd6.make_run_id("v1", "headline", 43, None, 1.0) == "v1_headline_s43"
    )  # D4 f=1.0 == D2 run


def _synthetic_world() -> tuple[pd.DataFrame, dict[str, np.ndarray], object]:
    rng = np.random.default_rng(0)
    labels = ["c0", "c1", "c2", "c3"]
    rows = []
    for lab in labels:
        for split, n in (("train", 12), ("val", 6), ("test", 6)):
            for i in range(n):
                rows.append((f"{lab}-{split}-{i}", f"msg LD-{i} {lab}", lab, split, len(rows)))
    df = pd.DataFrame(rows, columns=["id", "text", "label", "split", "dup_group"])
    sets = build_holdout_sets(df, ["c3"])
    d = 8
    centers = {lab: rng.normal(scale=5.0, size=d) for lab in labels}

    def feats(frame: pd.DataFrame) -> np.ndarray:
        return np.stack([centers[lab] for lab in frame["label"]]) + rng.normal(size=(len(frame), d))

    def logits(frame: pd.DataFrame) -> np.ndarray:
        lg = rng.normal(size=(len(frame), len(sets.labels)))
        for n, lab in enumerate(frame["label"]):
            if lab in sets.labels:
                lg[n, sets.labels.index(lab)] += 4.0
        return lg

    A: dict[str, np.ndarray] = {}
    for part, frame in (("train", sets.train), ("cal", sets.cal), ("ek", sets.eval_known),
                        ("unk", sets.eval_unknown)):  # fmt: skip
        A[f"{part}_ids"] = frame["id"].to_numpy(dtype=str)
        A[f"{part}_logits"], A[f"{part}_features"] = logits(frame), feats(frame)
    for part in ("cal", "ek", "unk"):
        A[f"{part}_logits_n"], A[f"{part}_features_n"] = (
            A[f"{part}_logits"] + 0.0,
            A[f"{part}_features"] + 0.0,
        )
    return df, A, sets


def test_score_table_layout_modes_and_separation() -> None:
    _, A, sets = _synthetic_world()
    raw = pd6.score_table(sets, A, neutral=False)
    neu = pd6.score_table(sets, A, neutral=True)
    for t in (raw, neu):
        assert set(t["set"]) == {"cal", "eval"} and "maha_ft" in t.columns
        assert t[t["set"] == "cal"]["is_unknown"].sum() == 0
        assert t["is_unknown"].sum() == len(sets.eval_unknown)
    assert pd.testing.assert_frame_equal(raw, neu) is None  # identical arrays -> identical tables
    ev = raw[raw["set"] == "eval"]
    unk = ev["is_unknown"].astype(bool)
    assert pd6.auroc_unknown_positive(ev.loc[~unk, "maha_ft"], ev.loc[unk, "maha_ft"]) > 0.9


def test_score_table_neutral_mode_uses_neutral_arrays_only_for_scored_rows() -> None:
    _, A, sets = _synthetic_world()
    shifted = dict(A)
    shifted["unk_features_n"] = (
        A["unk_features"] + 100.0
    )  # unknown rows look far away in neutral mode
    raw = pd6.score_table(sets, shifted, neutral=False)
    neu = pd6.score_table(sets, shifted, neutral=True)
    ru, nu = raw[raw["is_unknown"].astype(bool)], neu[neu["is_unknown"].astype(bool)]
    assert (nu["maha_ft"].to_numpy() < ru["maha_ft"].to_numpy()).all()
    # the scorer is fit on the unchanged training features: known rows are unaffected
    assert raw[~raw["is_unknown"].astype(bool)]["maha_ft"].equals(
        neu[~neu["is_unknown"].astype(bool)]["maha_ft"]
    )


def test_score_table_rejects_misaligned_ids() -> None:
    _, A, sets = _synthetic_world()
    bad = dict(A)
    bad["cal_ids"] = A["cal_ids"][::-1].copy()
    with pytest.raises(AssertionError):
        pd6.score_table(sets, bad, neutral=False)


def test_paired_cells_and_group_bootstrap_detect_a_shift() -> None:
    _, A, sets = _synthetic_world()
    t_good = pd6.score_table(sets, A, neutral=False)
    worse = dict(A)
    n_unk = len(A["unk_features"])
    worse["unk_features"] = A["train_features"][:n_unk]  # unknowns look like known rows
    t_bad = pd6.score_table(sets, worse, neutral=False)
    kw = dict(
        is_headline=True, score="maha_ft", retention=0.95, safe=[], n_boot=200, seed=42, level=0.95
    )
    good = pd6.group_bootstrap([t_good], **kw)
    bad = pd6.group_bootstrap([t_bad], **kw)
    cells = pd6.paired_cells(good, bad, 0.95)
    assert set(cells) == {"auroc", "strict_rej95", "retention95"}
    assert cells["auroc"]["delta"] > 0.3 and cells["auroc"]["ci_excludes_zero"]
    assert cells["auroc"]["lo"] <= cells["auroc"]["delta"] <= cells["auroc"]["hi"] + 1e-9
    c = pd6.cells_of(good)
    assert c["auroc"]["lo"] <= c["auroc"]["point"] <= c["auroc"]["hi"]
    assert pd6.row_fingerprint(t_good) == pd6.row_fingerprint(t_bad)  # same rows -> paired draws


def test_gather_looks_up_ids_across_sources() -> None:
    a = (np.array([[1.0], [2.0]]), np.array([[10.0], [20.0]]))
    b = (np.array([[3.0]]), np.array([[30.0]]))
    lg, ft = pd6._gather(["y", "x", "z"], [(["x", "y"], a), (["z"], b)])
    assert lg.ravel().tolist() == [2.0, 1.0, 3.0] and ft.ravel().tolist() == [20.0, 10.0, 30.0]


# --------------------------------------------------------------------------- isolation
def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_no_other_module_imports_or_reads_the_diagnostics() -> None:
    """The training/selection module must never see D1 (ORACLE) output: phase6a.py and friends
    neither import phase6a_diag nor mention its results directory."""
    offenders = []
    for p in sorted(SRC.glob("*.py")):
        # model_card.py renders one report-only oracle statistic (d1_oracle.json) into the HF
        # card; it is documentation output and never feeds training or selection.
        if p.name in ("phase6a_diag.py", "model_card.py"):
            continue
        text = p.read_text(encoding="utf-8")
        imports = _imports_of(p)
        if any(i.endswith("phase6a_diag") or ".phase6a_diag" in i for i in imports):
            offenders.append(f"{p.name}: imports phase6a_diag")
        for needle in ("phase6a/diag", "phase6a\\diag", "phase6a_diag"):
            if needle in text:
                offenders.append(f"{p.name}: mentions {needle!r}")
    assert not offenders, offenders


def test_diag_module_does_not_import_the_selection_module() -> None:
    imports = _imports_of(SRC / "phase6a_diag.py")
    assert not any(i.endswith(".phase6a") or i == "phase6a" for i in imports), imports
    assert not any("ood_variants" in i for i in imports)


def test_oracle_outputs_are_labelled_oracle() -> None:
    assert "ORACLE" in pd6.ORACLE_LABEL
    assert pd6.n2_decision(_auc(0.8, 0.8), _auc(0.9, 0.9))["oracle"] is True


def test_stage_order_runs_the_longest_stage_last() -> None:
    assert pd6.ALL_ORDER == ("d1", "d3", "d4", "d2")
    assert set(pd6.ALL_ORDER) == {"d1", "d2", "d3", "d4"}


def test_threshold_ceil_rule_helper_agrees_for_boundary_n() -> None:
    # the vectorised keep-count of threshold_bootstrap must equal threshold_at_retention's ceil rule
    for n in (19, 20, 21, 100, 426):
        keep = max(1, math.ceil(0.95 * n - 1e-9))
        cal = np.arange(n, dtype=float)
        b = pd6.threshold_bootstrap(cal, cal, cal, 0.95, n_boot=1, seed=0)
        idx = np.random.default_rng(0).integers(0, n, size=(1, n))
        assert b["threshold"][0] == threshold_at_retention(cal[idx[0]], 0.95)
        assert keep == max(1, math.ceil(0.95 * n - 1e-9))
