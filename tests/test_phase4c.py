from __future__ import annotations

import importlib.util
import math
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from intent_router import phase4c as p4  # noqa: E402
from intent_router import train as train_mod  # noqa: E402
from intent_router.models import state_dict_sha256  # noqa: E402

# the last commit before the Robustness and open-set training fixes additions to train.py
BASE_SHA = "d33fbf9"


# ============================================================================ A1: ID randomisation
def _ids_of(text: str) -> list[tuple[str, str]]:
    return re.findall(r"\b([A-Z]{2,4})-(\d+)", text)


def test_randomize_ids_changes_prefix_only_and_keeps_digits() -> None:
    text = "move PO-1234 to LD-99 and note ABC-7 plus hello world"
    for seed in range(20):
        out = train_mod.randomize_ids(text, 1.0, train_mod.id_rng(seed, 1, "r1"))
        assert [d for _, d in _ids_of(out)] == ["1234", "99", "7"]  # digits kept, count kept
        assert out.endswith(" plus hello world")
        assert out.startswith("move ") and " to " in out and " and note " in out
        for pre, _ in _ids_of(out):
            assert re.fullmatch(r"[A-Z]{2,4}", pre)


def test_randomize_ids_p_zero_is_identity_and_no_id_text_untouched() -> None:
    text = "where is PO-12 and LD-5"
    assert train_mod.randomize_ids(text, 0.0, train_mod.id_rng(0, 1, "x")) == text
    plain = "no identifiers here at all"
    assert train_mod.randomize_ids(plain, 1.0, train_mod.id_rng(0, 1, "x")) == plain


def test_randomize_ids_deterministic_per_seed_epoch_row_and_differs_per_epoch() -> None:
    text = "PO-1 LD-2 XY-3 PO-4 LD-5 AB-6 PO-7 LD-8"
    a = train_mod.randomize_ids(text, 0.5, train_mod.id_rng(42, 3, "row-9"))
    b = train_mod.randomize_ids(text, 0.5, train_mod.id_rng(42, 3, "row-9"))
    assert a == b
    outs = {train_mod.randomize_ids(text, 0.5, train_mod.id_rng(42, e, "row-9")) for e in range(8)}
    assert len(outs) > 1  # re-drawn every epoch
    rows = {
        train_mod.randomize_ids(text, 0.5, train_mod.id_rng(42, 3, f"row-{i}")) for i in range(8)
    }
    assert len(rows) > 1  # differs per row id


def test_randomize_ids_firing_rate_is_about_p_and_prefix_pool_is_uniform() -> None:
    p, n = 0.5, 4000
    fired = 0
    pool: dict[str, int] = {"PO": 0, "LD": 0, "REF": 0, "random": 0}
    rng = np.random.default_rng(0)
    for i in range(n):
        out = train_mod.randomize_ids(
            "PO-1", p, train_mod.id_rng(7, int(rng.integers(100)), f"r{i}")
        )
        pre = _ids_of(out)[0][0]
        # a fired draw lands on PO with prob 1/4 too, so compare changed-or-not via the pool only
        if pre in ("PO", "LD", "REF"):
            pool[pre] += 1
        else:
            pool["random"] += 1
        fired += pre != "PO"
    # P(not PO) = p * (1 - 1/4 - P(random draw == "PO") ~ 0)
    assert fired / n == pytest.approx(p * 0.75, abs=0.03)
    # among fired draws the three outcomes {LD, REF, random} are ~ equally likely
    shares = [pool["LD"], pool["REF"], pool["random"]]
    assert max(shares) - min(shares) < 0.06 * n


def test_randomize_ids_random_prefix_has_2_to_4_uppercase_letters() -> None:
    lengths = set()
    for i in range(400):
        out = train_mod.randomize_ids("ZZ-1", 1.0, train_mod.id_rng(1, 1, f"r{i}"))
        pre = _ids_of(out)[0][0]
        if pre not in ("PO", "LD", "REF"):
            lengths.add(len(pre))
    assert lengths == {2, 3, 4}


# ================================================================================== A2: OE loss
def test_oe_loss_uniform_logits_equal_log_k_exactly() -> None:
    for k in (2, 7, 12):
        logits = torch.zeros(5, k)
        assert float(train_mod.oe_uniform_loss(logits)) == pytest.approx(math.log(k), abs=1e-6)
        assert float(train_mod.oe_uniform_loss(logits + 3.7)) == pytest.approx(
            math.log(k), abs=1e-6
        )


def test_oe_loss_is_minimised_at_uniform_and_matches_formula() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    manual = -torch.log_softmax(logits, dim=-1).mean()
    assert float(train_mod.oe_uniform_loss(logits)) == pytest.approx(float(manual), abs=1e-7)
    assert float(train_mod.oe_uniform_loss(logits)) > math.log(3)  # confident item raises it
    g = logits.clone().requires_grad_(True)
    train_mod.oe_uniform_loss(g).backward()
    assert torch.isfinite(g.grad).all()
    # gradient w.r.t. an already-uniform item is zero
    assert torch.allclose(g.grad[1], torch.zeros(3), atol=1e-7)


def test_oe_batch_indices_seeded_by_seed_epoch_step() -> None:
    a = train_mod.oe_batch_indices(42, 2, 17, 600, 16)
    assert a.shape == (16,) and len(set(a.tolist())) == 16 and a.max() < 600
    assert np.array_equal(a, train_mod.oe_batch_indices(42, 2, 17, 600, 16))
    assert not np.array_equal(a, train_mod.oe_batch_indices(42, 2, 18, 600, 16))
    assert not np.array_equal(a, train_mod.oe_batch_indices(42, 3, 17, 600, 16))
    small = train_mod.oe_batch_indices(42, 1, 1, 5, 16)  # pool smaller than the batch: replacement
    assert small.shape == (16,) and small.max() < 5


def test_train_config_validates_new_fields() -> None:
    train_mod.TrainConfig(model_name="m", lr=1e-5, id_randomize_p=0.5, oe_lambda=0.5)
    with pytest.raises(ValueError):
        train_mod.TrainConfig(model_name="m", lr=1e-5, id_randomize_p=1.5)
    with pytest.raises(ValueError):
        train_mod.TrainConfig(model_name="m", lr=1e-5, oe_lambda=-1.0)
    cfg = train_mod.TrainConfig(model_name="m", lr=1e-5)
    assert cfg.id_randomize_p == 0.0 and cfg.oe_lambda == 0.0 and cfg.oe_batch_size == 16


# ============================================================================== A3: fold exclusion
def _aug(src_ids: list[str]) -> pd.DataFrame:
    rows = []
    for s in src_ids:
        rows += [(s, "mt", "es", f"mt {s}"), (s, "noise", "en", f"noise {s}")]
    return pd.DataFrame(rows, columns=["src_id", "kind", "lang", "text"])


def test_a3_rows_exclude_heldout_fold_copies() -> None:
    train = pd.DataFrame({"id": ["a", "b", "c"], "label": ["x", "y", "x"], "text": ["1", "2", "3"]})
    held = pd.DataFrame({"id": ["d", "e"], "label": ["x", "y"], "text": ["4", "5"]})
    aug = _aug(["a", "b", "c", "d", "e"])
    rows, n_excl = p4.a3_rows_for(aug, train, set(held["id"]))
    assert sorted(rows["src_id"].unique()) == ["a", "b", "c"]
    assert n_excl == 4 and len(rows) == 6
    assert not set(rows["src_id"]) & set(held["id"])
    assert rows["id"].is_unique and not set(rows["id"]) & set(train["id"])
    lab = dict(zip(train["id"], train["label"], strict=True))
    assert all(r.label == lab[r.src_id] for r in rows.itertuples())  # copy keeps its source label


def test_a3_rows_assert_when_a_forbidden_source_is_in_train() -> None:
    train = pd.DataFrame({"id": ["a", "b"], "label": ["x", "y"], "text": ["1", "2"]})
    with pytest.raises(AssertionError, match="held-out"):
        p4.a3_rows_for(_aug(["a", "b"]), train, {"b"})


def test_train_rejects_extra_rows_whose_source_is_in_eval() -> None:
    train = pd.DataFrame({"id": ["a"], "label": ["x"], "text": ["1"]})
    ev = pd.DataFrame({"id": ["e"], "label": ["x"], "text": ["2"]})
    bad = pd.DataFrame({"id": ["e|mt"], "text": ["t"], "label": ["x"], "src_id": ["e"]})
    with pytest.raises(AssertionError, match="src_id leak"):
        train_mod._check_extra_rows(bad, train, ev)
    ok = pd.DataFrame({"id": ["a|mt"], "text": ["t"], "label": ["x"], "src_id": ["a"]})
    train_mod._check_extra_rows(ok, train, ev)


def test_validate_aug_rejects_test_sources_and_incomplete_coverage() -> None:
    cv = ["a", "b"]
    p4.validate_aug(_aug(["a", "b"]), cv, ["mt", "noise"], full=True)
    with pytest.raises(AssertionError):
        p4.validate_aug(_aug(["a", "t1"]), cv, ["mt", "noise"], full=False)  # t1 not train+val
    with pytest.raises(ValueError, match="does not cover"):
        p4.validate_aug(_aug(["a"]), cv, ["mt", "noise"], full=True)


def test_oe_keep_mask_drops_close_items_only() -> None:
    held = np.array([[1.0, 0.0, 0.0]])
    syn = np.array([[1.0, 0.0, 0.0], [0.9, 0.436, 0.0], [0.0, 1.0, 0.0]])
    keep = p4.oe_keep_mask(syn, held, 0.85)
    assert keep.tolist() == [False, False, True]  # cos 1.0 and ~0.9 dropped, 0.0 kept
    assert p4.oe_keep_mask(syn, held, 0.95).tolist() == [False, True, True]


# =================================================================== adoption rules (each branch)
def _g(point: float, lo: float, hi: float | None = None) -> dict[str, float]:
    return {"point": point, "lo": lo, "hi": hi if hi is not None else point + 0.05}


def test_a1_rule_branches() -> None:
    assert p4.evaluate_a1({"a1_flip_reduction": _g(0.1, 0.02)}, True)["adopted"]
    assert not p4.evaluate_a1({"a1_flip_reduction": _g(0.1, 0.0)}, True)["adopted"]  # lo == 0
    assert not p4.evaluate_a1({"a1_flip_reduction": _g(0.1, -0.01)}, True)["adopted"]
    res = p4.evaluate_a1({"a1_flip_reduction": _g(0.1, 0.02)}, False)  # guard fails
    assert not res["adopted"] and res["adoption_metrics"] == ["a1_flip_reduction"]


def test_a2_rule_branches() -> None:
    m = 0.0115

    def g(au_point: float, rej_lo: float) -> dict[str, Any]:
        return {"a2_auroc_gain": _g(au_point, au_point - 0.02), "a2_rej95_gain": _g(0.05, rej_lo)}

    assert p4.evaluate_a2(g(0.02, -0.01), True, m)["adoption_metrics"] == ["a2_auroc_gain"]
    assert p4.evaluate_a2(g(0.0115, 0.01), True, m)["adoption_metrics"] == ["a2_rej95_gain"]
    both = p4.evaluate_a2(g(0.02, 0.01), True, m)
    assert both["adopted"] and both["adoption_metrics"] == ["a2_auroc_gain", "a2_rej95_gain"]
    assert not p4.evaluate_a2(g(0.0115, -0.01), True, m)["adopted"]  # margin is strict (>)
    assert not p4.evaluate_a2(g(0.02, 0.01), False, m)["adopted"]  # guard


def test_a3_rule_branches() -> None:
    def g(zh: float, mean: float, noise: float) -> dict[str, Any]:
        return {
            "a3_zh_agreement_gain": _g(0.1, zh),
            "a3_mean_agreement_gain": _g(0.1, mean),
            "a3_noise_drop_reduction": _g(0.05, noise),
        }

    agree = p4.evaluate_a3(g(0.01, 0.01, -0.01), True)
    assert agree["adopted"] and agree["adoption_metrics"] == [
        "a3_zh_agreement_gain", "a3_mean_agreement_gain",
    ]  # fmt: skip
    assert not p4.evaluate_a3(g(0.01, -0.01, -0.01), True)["adopted"]  # needs zh AND mean
    assert not p4.evaluate_a3(g(-0.01, 0.01, -0.01), True)["adopted"]
    noise = p4.evaluate_a3(g(-0.01, -0.01, 0.01), True)
    assert noise["adopted"] and noise["adoption_metrics"] == ["a3_noise_drop_reduction"]
    both = p4.evaluate_a3(g(0.01, 0.01, 0.01), True)
    assert len(both["adoption_metrics"]) == 3
    assert not p4.evaluate_a3(g(0.01, 0.01, 0.01), False)["adopted"]  # guard


def test_comb_rules_point_must_reach_alone_lower_bound() -> None:
    alone = {
        "a1": {
            "adoption_metrics": ["a1_flip_reduction"],
            "gains": {"a1_flip_reduction": _g(0.2, 0.10)},
        },
        "a2": {"adoption_metrics": ["a2_rej95_gain"], "gains": {"a2_rej95_gain": _g(0.06, 0.02)}},
    }
    ok = {"a1_flip_reduction": _g(0.10, 0.0), "a2_rej95_gain": _g(0.02, 0.0)}  # == lo: passes
    res = p4.evaluate_comb(["a1", "a2"], alone, ok, True)
    assert res["adopted"] and all(c["passed"] for c in res["checks"])
    low = {"a1_flip_reduction": _g(0.0999, 0.0), "a2_rej95_gain": _g(0.05, 0.0)}
    res = p4.evaluate_comb(["a1", "a2"], alone, low, True)
    assert not res["adopted"]
    assert [c["passed"] for c in res["checks"]] == [False, True]
    assert not p4.evaluate_comb(["a1", "a2"], alone, ok, False)["adopted"]  # guard must pass


def test_shipped_score_selection_rule() -> None:
    order = ["maha_ft", "neg_energy", "fuse_maha_energy"]
    au = {"maha_ft": 0.90, "neg_energy": 0.80, "fuse_maha_energy": 0.895}
    rj = {"maha_ft": 0.20, "neg_energy": 0.90, "fuse_maha_energy": 0.30}
    sel = p4.select_shipped_score(au, rj, 0.0115, order)
    assert sel["chosen"] == "fuse_maha_energy"  # neg_energy is out of tolerance despite rej@95
    assert sel["eligible"] == ["maha_ft", "fuse_maha_energy"]
    au2 = {**au, "neg_energy": 0.8886}  # inside the tolerance (0.9 - 0.8886 = 0.0114)
    assert p4.select_shipped_score(au2, rj, 0.0115, order)["chosen"] == "neg_energy"
    au3 = {**au, "neg_energy": 0.8884}  # 0.0116 away: outside
    assert p4.select_shipped_score(au3, rj, 0.0115, order)["chosen"] == "fuse_maha_energy"
    tie = {"maha_ft": 0.2, "neg_energy": 0.2, "fuse_maha_energy": 0.2}
    assert p4.select_shipped_score(au, tie, 0.0115, order)["chosen"] == "maha_ft"  # higher AUROC
    assert p4.best_score_by_auroc(au, order) == "maha_ft"


def test_comb_recipe_resolution() -> None:
    assert p4.comb_recipe(["a2"]) == p4.Recipe("a2", a2=True)  # single factor reuses its artifacts
    r = p4.comb_recipe(["a1", "a3"])
    assert r.key == "comb" and r.factors == ("a1", "a3")
    with pytest.raises(ValueError, match="no factors adopted"):
        p4.comb_recipe([])
    pcfg = {
        "factors": {"a1": {"id_randomize_p": 0.5}, "a2": {"oe_lambda": 0.5, "oe_batch_size": 16}}
    }
    assert p4.recipe_overrides(r, pcfg) == {"id_randomize_p": 0.5}
    assert p4.recipe_overrides(p4.Recipe("current"), pcfg) == {}


# ================================================================== paired bootstrap + summaries
def test_guard_summary_argmax_and_threshold() -> None:
    g = {"threshold": 0.9341, "comparator": 0.9441}
    cur = {"deployed_epoch": 2}
    out = p4.guard_summary(
        "a1", [[0.90, 0.95, 0.94], [0.92, 0.95, 0.93]], [[0.9] * 3] * 2, g, False, cur
    )
    assert out["argmax_epoch"] == 2 and out["deployed_epoch"] == 2
    assert out["macro_f1_at_argmax"] == pytest.approx(0.95) and out["passed"]
    bad = p4.guard_summary("a1", [[0.90, 0.93], [0.92, 0.93]], [[0.9] * 2] * 2, g, False, cur)
    assert not bad["passed"] and "guard" in bad["why"]
    tie = p4.guard_summary("a1", [[0.95, 0.95]], [[0.9, 0.9]], g, False, cur)
    assert tie["argmax_epoch"] == 1  # ties go to the earliest epoch
    cur_out = p4.guard_summary(
        "current", [[0.90, 0.95, 0.94]], [[0.9] * 3], g, False, {"deployed_epoch": 3}
    )
    assert cur_out["argmax_epoch"] == 2 and cur_out["deployed_epoch"] == 3
    assert cur_out["macro_f1_at_deployed"] == pytest.approx(0.94)


def _swap_frame(flip_ids: set[str], n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for sw in ("LD->REF", "PO->REF"):
            flipped = f"r{i}" in flip_ids
            rows.append((f"r{i}", sw, 1, 1, 2 if flipped else 1))
    return pd.DataFrame(rows, columns=["id", "swap", "gold", "clean_pred", "swap_pred"])


def test_a1_gain_detects_a_real_reduction_and_a_null() -> None:
    cur = _swap_frame({f"r{i}" for i in range(0, 200, 2)})  # 50% of ids flip
    cand = _swap_frame({f"r{i}" for i in range(0, 200, 10)})  # 10% flip
    g = p4.a1_gain(cur, cand, None, 2000, 42, 0.95)
    assert g["point"] == pytest.approx(0.40) and g["lo"] > 0 and g["ci_excludes_zero_positive"]
    assert g["flip_rate_current"] == pytest.approx(0.5) and g["n_instances"] == 400
    null = p4.a1_gain(cur, cur, None, 2000, 42, 0.95)
    assert (
        null["point"] == 0.0
        and null["lo"] <= 0 <= null["hi"]
        and not null["ci_excludes_zero_positive"]
    )
    with pytest.raises(ValueError, match="differ"):
        p4.a1_gain(cur, cand.iloc[:-2], None, 100, 42, 0.95)


def _mt_frame(agree: dict[str, float], n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        for lang in ("es", "fr", "de", "zh"):
            hit = rng.random() < agree[lang]
            rows.append((f"r{i}", lang, True, 1, 1, 1 if hit else 2))
    return pd.DataFrame(rows, columns=["id", "lang", "kept", "gold", "en_pred", "mt_pred"])


def test_a3_agreement_gains_zh_and_mean() -> None:
    cur = _mt_frame({"es": 0.8, "fr": 0.8, "de": 0.8, "zh": 0.5})
    cand = cur.copy()
    cand.loc[cand["lang"] == "zh", "mt_pred"] = 1  # zh fully agrees now; others unchanged
    g = p4.a3_agreement_gains(cur, cand, ["es", "fr", "de", "zh"], 1000, 42, 0.95)
    assert g["zh"]["point"] == pytest.approx(0.5, abs=0.1) and g["zh"]["lo"] > 0
    assert g["mean"]["point"] == pytest.approx(g["zh"]["point"] / 4)
    assert g["per_language"]["es"]["point"] == 0.0
    # the quality filter is respected: dropped rows do not count
    drop = cur["id"].str[1:].astype(int) < 60
    cur2, cand2 = cur.assign(kept=~drop), cand.assign(kept=~drop)
    g2 = p4.a3_agreement_gains(cur2, cand2, ["es", "fr", "de", "zh"], 200, 42, 0.95)
    assert g2["n_kept_per_language"] == {"es": 60, "fr": 60, "de": 60, "zh": 60}
    with pytest.raises(ValueError, match="no quality-filtered rows"):
        p4.a3_agreement_gains(
            cur.assign(kept=False), cand.assign(kept=False), ["es", "fr", "de", "zh"], 10, 42, 0.95
        )


def test_noise_gain_reduction_of_the_accuracy_drop() -> None:
    n = 300
    gold = np.zeros(n, dtype=int)
    cur = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(n)],
            "gold": gold,
            "clean_pred": gold,
            "noise_pred": np.where(np.arange(n) % 4 == 0, 1, 0),
        }
    )  # fmt: skip  (drop 0.25)
    cand = cur.assign(noise_pred=np.where(np.arange(n) % 20 == 0, 1, 0))  # drop 0.05
    g = p4.noise_gain(cur, cand, 2000, 42, 0.95)
    assert g["point"] == pytest.approx(0.20) and g["lo"] > 0
    assert g["drop_current"] == pytest.approx(0.25) and g["drop_candidate"] == pytest.approx(0.05)


def _dev_table(shift: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = rng.normal(0, 1, 100)
    ek = rng.normal(0, 1, 150)
    unk = rng.normal(-shift, 1, 80)
    rows = []
    for name, vals, is_unk in (("cal", cal, False), ("eval", ek, False), ("eval", unk, True)):
        for v in vals:
            rows.append((name, is_unk, "other" if is_unk and v < -5 else "x", float(v)))
    return pd.DataFrame(rows, columns=["set", "is_unknown", "pred", "maha_ft"])


def test_a2_gains_paired_and_sign() -> None:
    cur = {f"c{i}": _dev_table(0.5, i) for i in range(3)}
    cand = {f"c{i}": _dev_table(2.0, i) for i in range(3)}
    g = p4.a2_gains(cur, cand, "maha_ft", "maha_ft", ["other"], 500, 42, 0.95)
    assert g["auroc_gain"]["point"] > 0.1 and g["auroc_gain"]["lo"] > 0
    assert g["rej95_gain"]["point"] > 0 and g["rej95_gain"]["ci_excludes_zero_positive"]
    same = p4.a2_gains(cur, cur, "maha_ft", "maha_ft", ["other"], 500, 42, 0.95)
    assert same["auroc_gain"]["point"] == 0.0
    assert same["auroc_gain"]["lo"] == 0.0 == same["auroc_gain"]["hi"]  # identical draws: paired
    with pytest.raises(ValueError, match="DEV class sets differ"):
        p4.a2_gains(cur, {"zz": cand["c0"]}, "maha_ft", "maha_ft", ["other"], 10, 42, 0.95)


def test_dev_summary_and_rejection_threshold_rule() -> None:
    t = _dev_table(2.0, 0)
    s = p4.dev_summary({"c": t}, ["maha_ft"])["maha_ft"]
    cal = t[t["set"] == "cal"]["maha_ft"].to_numpy()
    thr = np.sort(cal)[len(cal) - math.ceil(0.95 * len(cal))]
    ev = t[t["set"] == "eval"]
    unk = ev[ev["is_unknown"]]["maha_ft"].to_numpy()
    assert s["rej95_mean"] == pytest.approx(float(np.mean(unk < thr)))
    assert 0.5 < s["auroc_mean"] < 1.0


def test_swap_noise_mt_summaries() -> None:
    sw = _swap_frame({"r0", "r1"}, n=10)
    s = p4.swap_summary(sw)
    assert s["pooled"]["n"] == 20 and s["pooled"]["n_flips"] == 4
    assert s["pooled"]["flip_rate"] == pytest.approx(0.2) and set(s["per_swap"]) == {
        "LD->REF",
        "PO->REF",
    }
    nz = pd.DataFrame({"id": list("abcd"), "gold": [1, 1, 1, 1], "clean_pred": [1, 1, 1, 0],
                       "noise_pred": [1, 0, 0, 0]})  # fmt: skip
    r = p4.noise_summary(nz)
    assert r["accuracy_clean"] == 0.75 and r["accuracy_noisy"] == 0.25 and r["drop"] == 0.5
    mt = _mt_frame({"es": 1.0, "fr": 1.0, "de": 1.0, "zh": 1.0}, n=4)
    mt.loc[0, "kept"] = False
    out = p4.mt_summary(mt, ["es", "fr", "de", "zh"])
    assert out["es"]["unfiltered"]["n"] == 4 and out["es"]["filtered"]["n"] == 3
    assert out["mean_filtered"]["agreement"] == 1.0


def test_ratio_samples_and_gain_record() -> None:
    idx = np.array([[0, 1], [1, 1]])
    r = p4.ratio_samples(np.array([1.0, 3.0]), np.array([2.0, 2.0]), idx)
    assert r.tolist() == [1.0, 1.5]
    z = p4.ratio_samples(np.array([1.0]), np.array([0.0]), np.array([[0]]))
    assert np.isnan(z[0])  # empty denominator is NaN, never a silent 0
    rec = p4.gain_record(0.3, np.array([0.1, 0.2, 0.3, np.nan, 0.4]), 0.95)
    assert rec["n_resamples"] == 4 and rec["lo"] > 0 and rec["ci_excludes_zero_positive"]


def test_validate_eval_mt_rejects_non_english_ids() -> None:
    mt = pd.DataFrame({"id": ["a", "b"], "lang": ["es", "zh"], "text": ["x", "y"], "system": "m",
                       "labse_cos": [0.9, 0.7], "kept": [True, False]})  # fmt: skip
    p4.validate_eval_mt(mt, ["a", "b"], ["es", "fr", "de", "zh"])
    with pytest.raises(AssertionError):
        p4.validate_eval_mt(mt, ["a"], ["es", "zh"])
    with pytest.raises(ValueError):
        p4.validate_eval_mt(mt.assign(kept=["True", "False"]), ["a", "b"], ["es", "zh"])


# ============================================================ defaults off: bitwise equivalence
def _word_id(w: str) -> int:
    return 2 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 90


class _StubTokenizer:
    """Whitespace tokenizer (tiny vocab) with the call + pad signatures train/predict use."""

    def __call__(self, texts: list[str], padding: bool = False, truncation: bool = True,
                 max_length: int = 64, return_tensors: str | None = None) -> Any:  # fmt: skip
        ids = [[_word_id(w) for w in t.split()][:max_length] or [2] for t in texts]
        feats = [{"input_ids": i, "attention_mask": [1] * len(i)} for i in ids]
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": [f["attention_mask"] for f in feats]}
        return self.pad(feats, return_tensors)

    def pad(self, feats: list[dict[str, list[int]]], return_tensors: str) -> Any:
        width = max(len(f["input_ids"]) for f in feats)
        return transformers.BatchEncoding({
            "input_ids": torch.tensor(
                [f["input_ids"] + [1] * (width - len(f["input_ids"])) for f in feats]
            ),
            "attention_mask": torch.tensor(
                [f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in feats]
            ),
        })  # fmt: skip


def _tiny_model(n_labels: int) -> Any:
    cfg = transformers.XLMRobertaConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=70, num_labels=n_labels, pad_token_id=1,
    )  # fmt: skip
    return transformers.XLMRobertaForSequenceClassification(cfg)


LABELS = ["a", "b", "c", "d"]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {"id": f"r{i}", "text": f"w{i} PO-{i} x y LD-{i}", "label": LABELS[i % 4], "split": s}
        for i, s in enumerate(["train"] * 24 + ["val"] * 8)
    ]
    df = pd.DataFrame(rows)
    return df[df["split"] == "train"], df[df["split"] == "val"]


def _fake_build(name: str, labels: list[str], attn: str | None) -> tuple[Any, Any]:
    return _StubTokenizer(), _tiny_model(len(labels))


def _run(mod: Any, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> tuple[Any, str]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU: never contends the GPU
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(mod, "build_model", _fake_build)
    train, ev = _frames()
    extras = kw.pop("extras", {})
    cfg = train_mod.TrainConfig(model_name="tiny", lr=1e-3, epochs=3, batch_size=8, max_len=12,
                                stop_epoch=3, wandb_group="x", **kw)  # fmt: skip
    # a plain dict: the pre-4c module has its own TrainConfig class (from_dict filters fields)
    res, model, _ = mod.train_model(
        asdict(cfg), train, ev, {"run_id": "t"}, labels=LABELS, **extras
    )
    return res, state_dict_sha256(model)


def _load_original_train(tmp_path: Path) -> Any:
    try:
        src = subprocess.run(  # noqa: S603
            ["git", "show", f"{BASE_SHA}:src/intent_router/train.py"],  # noqa: S607
            capture_output=True, check=True, text=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history for {BASE_SHA} unavailable")
    path = tmp_path / "train_orig.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("intent_router_train_orig", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["intent_router_train_orig"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_defaults_off_is_bitwise_identical_to_the_unmodified_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig = _load_original_train(tmp_path)
    assert not hasattr(orig, "randomize_ids")  # really the pre-4c module
    res_o, sha_o = _run(orig, monkeypatch)
    res_n, sha_n = _run(train_mod, monkeypatch)
    assert sha_o == sha_n  # identical final weights
    assert np.array_equal(res_o.probs, res_n.probs)  # bitwise-identical per-epoch eval probs
    assert [e["train_loss"] for e in res_o.epochs] == [e["train_loss"] for e in res_n.epochs]
    assert [e["macro_f1"] for e in res_o.epochs] == [e["macro_f1"] for e in res_n.epochs]


def test_each_factor_changes_training_and_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    base, sha0 = _run(train_mod, monkeypatch)
    oe = [f"alien request {i} about nothing" for i in range(20)]
    train_df, _ = _frames()
    extra = pd.DataFrame({
        "id": [f"{i}|mt" for i in train_df["id"]], "text": [t + " zz" for t in train_df["text"]],
        "label": train_df["label"], "src_id": train_df["id"],
    })  # fmt: skip
    variants = {
        "a1": {"id_randomize_p": 0.5},
        "a2": {"oe_lambda": 0.5, "extras": {"oe_texts": oe}},
        "a3": {"extras": {"extra_train_rows": extra}},
    }
    shas = {"base": sha0}
    for name, kw in variants.items():
        r1, s1 = _run(
            train_mod, monkeypatch, **{k: (dict(v) if k == "extras" else v) for k, v in kw.items()}
        )
        r2, s2 = _run(
            train_mod, monkeypatch, **{k: (dict(v) if k == "extras" else v) for k, v in kw.items()}
        )
        assert s1 == s2 and np.array_equal(r1.probs, r2.probs), f"{name} is not deterministic"
        assert not r1.nan_detected
        shas[name] = s1
    assert len(set(shas.values())) == 4  # every factor really alters the optimisation
    assert base.epochs[0]["train_loss"] > 0


def test_oe_requires_a_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="oe_texts"):
        _run(train_mod, monkeypatch, oe_lambda=0.5)


def test_defaults_off_bitwise_on_gpu_when_idle(tmp_path: Path) -> None:
    from _pytest.monkeypatch import MonkeyPatch

    from intent_router import gpu_lock

    if not torch.cuda.is_available() or gpu_lock.foreign_gpu_processes():
        pytest.skip("GPU absent or busy")
    orig = _load_original_train(tmp_path)

    results = []
    for mod in (orig, train_mod):
        with MonkeyPatch.context() as mp:
            mp.setenv("WANDB_MODE", "disabled")
            mp.setattr(mod, "build_model", _fake_build)
            train, ev = _frames()
            cfg = train_mod.TrainConfig(model_name="tiny", lr=1e-3, epochs=2, batch_size=8,
                                        max_len=12, stop_epoch=2, wandb_group="x",
                                        attn_implementation=None)  # fmt: skip
            res, model, _ = mod.train_model(asdict(cfg), train, ev, {"run_id": "t"}, labels=LABELS)
            results.append((res.probs.copy(), state_dict_sha256(model)))
            del model
    assert results[0][1] == results[1][1] and np.array_equal(results[0][0], results[1][0])


def test_read_text_csv_keeps_empty_text_as_empty_string(tmp_path: Path) -> None:
    """An empty translation must load as "" (str), not NaN, and the row must be preserved."""
    f = tmp_path / "mt.csv"
    f.write_text("id,lang,text,kept\na,de,hallo,True\nb,de,,False\n", encoding="utf-8")
    df = p4.read_text_csv(f)
    assert len(df) == 2
    assert df["text"].tolist() == ["hallo", ""]
    assert all(isinstance(t, str) for t in df["text"])
