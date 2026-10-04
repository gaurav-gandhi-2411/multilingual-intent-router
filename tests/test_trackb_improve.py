from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from intent_router import trackb_improve as ti  # noqa: E402
from intent_router.models import predict, predict_hidden  # noqa: E402
from intent_router.ood import auroc_unknown_positive, score_mahalanobis  # noqa: E402
from intent_router.train import _micro_slices, supcon_loss  # noqa: E402

DEV = ["d1", "d2", "d3", "d4", "d5"]


# ----------------------------------------------------------------------------- SupCon
def test_supcon_hand_example_two_classes() -> None:
    # z = e1, e1, e2, e2; labels 0,0,1,1; T = 1. Anchor 0: positive {1} (s=1), denominator
    # over a != 0 is e^1 + e^0 + e^0, so loss_i = log(e + 2) - 1 for every anchor.
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = torch.tensor([0, 0, 1, 1])
    assert float(supcon_loss(z, y, 1.0)) == pytest.approx(math.log(math.e + 2) - 1, abs=1e-6)
    # T = 0.5 doubles the similarities: log(e^2 + 2) - 2
    assert float(supcon_loss(z, y, 0.5)) == pytest.approx(math.log(math.e**2 + 2) - 2, abs=1e-6)


def test_supcon_skips_anchors_without_positive_and_has_finite_grad() -> None:
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    y = torch.tensor([0, 0, 1])  # row 2 has no positive: not an anchor
    loss = supcon_loss(z, y, 1.0)
    assert float(loss.detach()) == pytest.approx(math.log(math.e + 1) - 1, abs=1e-6)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    none = supcon_loss(torch.eye(3), torch.tensor([0, 1, 2]), 0.1)  # no anchor at all
    assert float(none) == 0.0


def test_micro_slices_cover_the_batch() -> None:
    assert _micro_slices(16, 1) == [None]
    sl = _micro_slices(16, 4)
    assert [(s.start, s.stop) for s in sl if s is not None] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    sl = _micro_slices(7, 2)
    assert [(s.start, s.stop) for s in sl if s is not None] == [(0, 4), (4, 7)]


# ------------------------------------------------------------- tiny model plumbing
def _word_id(w: str) -> int:
    return 2 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 90


class _StubTokenizer:
    """Whitespace tokenizer (tiny vocab) with the call + pad signatures train/predict use."""

    def _ids(self, texts: list[str], max_length: int) -> list[list[int]]:
        return [[_word_id(w) for w in t.split()][:max_length] or [2] for t in texts]

    def __call__(
        self,
        texts: list[str],
        padding: bool = False,
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str | None = None,
    ) -> Any:
        ids = self._ids(texts, max_length)
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}
        return self.pad(
            [{"input_ids": i, "attention_mask": [1] * len(i)} for i in ids], return_tensors
        )

    def pad(self, feats: list[dict[str, list[int]]], return_tensors: str) -> Any:
        width = max(len(f["input_ids"]) for f in feats)
        return transformers.BatchEncoding(
            {
                "input_ids": torch.tensor(
                    [f["input_ids"] + [1] * (width - len(f["input_ids"])) for f in feats]
                ),
                "attention_mask": torch.tensor(
                    [f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in feats]
                ),
            }
        )


def _tiny_model(n_labels: int = 5) -> transformers.XLMRobertaForSequenceClassification:
    cfg = transformers.XLMRobertaConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=70, num_labels=n_labels, pad_token_id=1,
    )  # fmt: skip
    torch.manual_seed(0)
    return transformers.XLMRobertaForSequenceClassification(cfg)


def test_predict_hidden_matches_predict_and_manual_pooling() -> None:
    model, tok = _tiny_model(), _StubTokenizer()
    texts = ["alpha beta", "gamma", "delta epsilon zeta eta"]
    layers = [0, -2, -1]
    lg, ft, pooled = predict_hidden(model, tok, texts, "tiny", 16, 2, layers)
    lg0, ft0 = predict(model, tok, texts, "tiny", 16, 2)
    assert np.array_equal(lg, lg0) and np.array_equal(ft, ft0)  # same forward, same numbers
    assert pooled.shape == (3, 3, 32) and pooled.dtype == np.float32
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(texts):  # batch of 1 => no padding at all
            enc = tok([t], padding=True, truncation=True, max_length=16, return_tensors="pt")
            hs = model(**enc, output_hidden_states=True).hidden_states
            for j, layer in enumerate(layers):
                np.testing.assert_allclose(
                    pooled[i, j], hs[layer][0].mean(0).numpy(), atol=1e-5
                )  # padding in the batched call must not leak into the mean
    assert predict_hidden(model, tok, [], "tiny", 16, 2, layers)[2].shape == (0, 3, 32)


def test_train_with_supcon_and_accumulation_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from intent_router import train as train_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def fake_build(name: str, labels: list[str], attn: str | None) -> tuple[Any, Any]:
        return _StubTokenizer(), _tiny_model(len(labels))

    monkeypatch.setattr(train_mod, "build_model", fake_build)
    labels = ["a", "b", "c", "d"]
    rows = [
        {"id": f"r{i}", "text": f"w{i} x y", "label": labels[i % 4], "split": s}
        for i, s in enumerate(["train"] * 24 + ["val"] * 8)
    ]
    df = pd.DataFrame(rows)
    out = {}
    for name, kw in {
        "plain": {},
        "supcon": {"supcon_lambda": 0.5},
        "accum": {"grad_accum": 2},
    }.items():
        cfg = train_mod.TrainConfig(
            model_name="tiny", lr=1e-3, epochs=2, batch_size=8, max_len=8, stop_epoch=2,
            wandb_group="x", **kw,
        )  # fmt: skip
        res, model, _ = train_mod.train_model(
            cfg, df[df["split"] == "train"], df[df["split"] == "val"], {"run_id": name},
            labels=labels,
        )  # fmt: skip
        assert not res.nan_detected and np.isfinite(res.probs).all()
        out[name] = res.epochs[-1]["train_loss"]
    assert out["supcon"] != out["plain"]  # the extra term changes the optimisation


# --------------------------------------------------------------- multi-layer Mahalanobis
def _toy_layers(seed: int, n: int, n_layers: int = 2, d: int = 4, k: int = 3) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    centres = rng.normal(0, 3, size=(n_layers, k, d))
    x = np.stack([centres[:, c] + rng.normal(0, 1, (n_layers, d)) for c in y])
    return x, y


def test_multilayer_maha_standardises_on_cal_known_only() -> None:
    tr_x, tr_y = _toy_layers(0, 60)
    cal_x, _ = _toy_layers(1, 30)
    ml = ti.fit_multilayer_maha(tr_x, tr_y, 3, cal_x)
    raw_cal = ml.layer_scores(cal_x)
    np.testing.assert_allclose(ml.mu, raw_cal.mean(axis=0))
    np.testing.assert_allclose(ml.sd, raw_cal.std(axis=0))  # population std of the cal rows
    z = (raw_cal - ml.mu) / ml.sd
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-9)
    assert ml.score(cal_x).mean() == pytest.approx(0.0, abs=1e-9)
    # Per-layer score is the plain Mahalanobis score of that layer; scoring other rows (train,
    # unknowns) never feeds back into mu / sd.
    np.testing.assert_allclose(
        ml.layer_scores(cal_x)[:, 0], score_mahalanobis(cal_x[:, 0], ml.gaussians[0])
    )
    ml.score(tr_x)
    np.testing.assert_allclose(ml.mu, raw_cal.mean(axis=0))
    # A different cal set gives different standardisation (so cal rows really are what is used).
    other = ti.fit_multilayer_maha(tr_x, tr_y, 3, _toy_layers(2, 30)[0])
    assert not np.allclose(other.mu, ml.mu)


# ---------------------------------------------------------------------- rank fusion
def test_rank_fusion_toy() -> None:
    f = ti.fit_rank_fusion([np.array([4.0, 1, 3, 2]), np.array([10.0, 20, 30, 40])])
    a, b = np.array([0.0, 2.5, 5.0]), np.array([10.0, 35.0, 100.0])
    # ECDF a: [0, .5, 1]; ECDF b: [.25, .75, 1]
    np.testing.assert_allclose(f.transform([a, b]), [0.125, 0.625, 1.0])
    with pytest.raises(ValueError):
        f.transform([a])


# ---------------------------------------------------------------------- selection rule
def _auroc_table(cur: list[float], delta: float) -> dict[str, dict[str, float]]:
    return {
        ti.CURRENT_KEY: dict(zip(DEV, cur, strict=True)),
        "base/maha_ml": {c: v + delta for c, v in zip(DEV, cur, strict=True)},
        "c2/other": {c: v - 0.1 for c, v in zip(DEV, cur, strict=True)},
    }


def test_selection_ship_no_ship_boundary() -> None:
    cur = [0.80, 0.90, 0.70, 0.85, 0.75]
    margin = float(np.std(cur, ddof=1) / math.sqrt(5))
    no = ti.select_on_dev(_auroc_table(cur, margin - 1e-6), ["base/maha_ml"], ti.CURRENT_KEY, DEV)
    yes = ti.select_on_dev(_auroc_table(cur, margin + 1e-6), ["base/maha_ml"], ti.CURRENT_KEY, DEV)
    assert no["margin_std_ddof1_over_sqrt_n"] == pytest.approx(margin)
    assert no["winner"] == "base/maha_ml" and not no["ship"] and no["shipped"] == ti.CURRENT_KEY
    assert yes["ship"] and yes["shipped"] == "base/maha_ml"
    # ineligible keys are never winners, even when they would be the best
    tab = _auroc_table(cur, 0.0)
    tab["b1/x"] = {c: 0.99 for c in DEV}
    out = ti.select_on_dev(tab, ["base/maha_ml"], ti.CURRENT_KEY, DEV)
    assert out["winner"] == "base/maha_ml" and not out["ship"]
    assert ti.select_on_dev(tab, [], ti.CURRENT_KEY, DEV)["winner"] is None


def test_selection_refuses_non_dev_classes() -> None:
    cur = [0.8, 0.9, 0.7, 0.85, 0.75]
    tab = _auroc_table(cur, 0.1)
    tab["base/maha_ml"]["confirm_class"] = 0.9  # a CONFIRM class leaks in
    with pytest.raises(ValueError, match="DEV classes"):
        ti.select_on_dev(tab, ["base/maha_ml"], ti.CURRENT_KEY, DEV)
    tab = _auroc_table(cur, 0.1)
    del tab["base/maha_ml"]["d5"]  # a DEV class is missing
    with pytest.raises(ValueError, match="missing"):
        ti.select_on_dev(tab, ["base/maha_ml"], ti.CURRENT_KEY, DEV)
    with pytest.raises(KeyError):
        ti.select_on_dev(_auroc_table(cur, 0.1), ["nope"], ti.CURRENT_KEY, DEV)


# ---------------------------------------------------------------------------- bootstrap
def _unit(known: list[float], unknown: list[float], thr: float = 0.0) -> ti.BootUnit:
    return ti.BootUnit(
        np.array(known, dtype=float),
        np.array(unknown, dtype=float),
        np.zeros(len(unknown), dtype=bool),
        {0.95: thr, 0.90: thr},
    )


def test_bootstrap_is_stratified_within_holdout() -> None:
    # Known rows are all above the threshold, unknown rows below: every stratified resample keeps
    # both groups, so AUROC, retention and rejection are exactly 1 in EVERY resample.
    u = _unit([10, 11, 12, 13, 14], [-1, -2, -3])
    b = ti.bootstrap_mean_ci([u], 500, 42)
    for m in ("auroc", "retention_95", "strict_recall_95", "retention_90", "lenient_recall_90"):
        assert b["lo"][m] == b["hi"][m] == 1.0 and b["point"][m] == 1.0
    # With a single mixed unknown group the unknown share is resampled but known retention
    # (known rows only) stays exactly 1: the groups are never mixed.
    u2 = _unit([5, 5, 5, 5], [-1, 10])
    b2 = ti.bootstrap_mean_ci([u2], 4000, 42)
    assert (b2["samples"]["retention_95"] == 1.0).all()
    assert set(np.unique(b2["samples"]["strict_recall_95"])) <= {0.0, 0.5, 1.0}
    assert b2["samples"]["strict_recall_95"].mean() == pytest.approx(0.5, abs=0.03)
    assert b2["point"]["strict_recall_95"] == 0.5


def test_bootstrap_point_matches_auroc_and_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    kn, un = rng.normal(1, 1, 40), rng.normal(0, 1, 25)
    u = _unit(kn.tolist(), un.tolist(), thr=0.2)
    a = ti.bootstrap_mean_ci([u, u], 300, 42, shared_draws=True)
    b = ti.bootstrap_mean_ci([u, u], 300, 42, shared_draws=True)
    assert a["point"]["auroc"] == pytest.approx(auroc_unknown_positive(kn, un))
    assert np.array_equal(a["samples"]["auroc"], b["samples"]["auroc"])
    # identical units + shared draws => the mean over units equals one unit's own samples
    one = ti.bootstrap_mean_ci([u], 300, 42)
    assert np.array_equal(a["samples"]["auroc"], one["samples"]["auroc"])
    d = ti.paired_delta_ci(a["samples"], one["samples"])
    assert d["auroc"]["mean_delta"] == 0.0
    with pytest.raises(ValueError):
        ti.bootstrap_mean_ci([u, _unit([1, 2], [0, 1])], 10, 1, shared_draws=True)


def test_rejection_retention_curve_uses_cal_threshold() -> None:
    cal = np.arange(1.0, 21.0)  # 20 cal rows
    known, unknown = np.array([5.0, 15.0, 25.0, 1.0]), np.array([0.0, 3.0, 12.0])
    c = ti.rejection_retention_curve(cal, known, unknown, [0.95, 0.90])
    # keep ceil(0.95 * 20) = 19 rows -> t = 2; keep 18 -> t = 3
    assert c["retention_known"] == [0.75, 0.75]
    assert c["strict_rejection_recall"] == [pytest.approx(1 / 3), pytest.approx(1 / 3)]
    c2 = ti.rejection_retention_curve(cal, known, unknown, [0.5])  # keep 10 -> t = 11
    assert c2["retention_known"] == [0.5] and c2["strict_rejection_recall"] == [
        pytest.approx(2 / 3)
    ]


# ---------------------------------------------------------------------------- business
def test_business_arithmetic() -> None:
    r = ti.business_row(1000, 0.05, 0.93, 0.40, 0.60)
    assert r["known_wrongly_abstained"] == pytest.approx(950 * 0.07)
    assert r["unknowns_caught_strict"] == pytest.approx(20.0)
    assert r["unknowns_misrouted_strict"] == pytest.approx(30.0)
    assert r["unknowns_caught_lenient"] == pytest.approx(30.0)
    assert r["unknowns_misrouted_lenient"] == pytest.approx(20.0)


# ------------------------------------------------------------------ reproduction compare
def _table(msp: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"id": ["a", "b", "c"], "set": ["cal", "eval", "eval"], "pred": ["x", "y", "x"], "msp": msp}
    )


def test_compare_scores_detects_any_difference() -> None:
    same = ti.compare_scores(_table([0.1, 0.2, 0.3]), _table([0.1, 0.2, 0.3]), ["msp"])
    assert same["methods"]["msp"]["array_equal"] and same["methods"]["msp"]["max_abs_diff"] == 0.0
    diff = ti.compare_scores(_table([0.1, 0.2, 0.3]), _table([0.1, 0.2, 0.30000001]), ["msp"])
    assert not diff["methods"]["msp"]["array_equal"]
    assert diff["methods"]["msp"]["max_abs_diff"] == pytest.approx(1e-8)
    assert not ti.compare_scores(
        _table([0.1, 0.2, 0.3]), _table([0.1, 0.2, 0.3]).iloc[:2], ["msp"]
    )["aligned"]


def test_log_test_call_counts_attempts(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    assert ti.log_test_call(p, "fp1", "trackb_reproduce", {"run_id": "r"}) == 1
    assert ti.log_test_call(p, "fp1", "trackb_reproduce", {"run_id": "r"}) == 2  # visible repeat
    assert ti.log_test_call(p, "fp2", "trackb_reproduce", {"run_id": "r2"}) == 1
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert [e["attempt"] for e in lines] == [1, 2, 1]
    assert all(e["call_type"] == "trackb_reproduce" and "git_sha" in e for e in lines)


# ------------------------------------------------------------ scoring pipeline (CPU, synthetic)
def _synthetic_ctx(tmp_path: Path) -> tuple[ti.Ctx, Any, dict[str, np.ndarray]]:
    from intent_router.ood import build_holdout_sets

    rng = np.random.default_rng(0)
    labels = ["a", "b", "c", "d"]
    rows = []
    for split, per in (("train", 12), ("val", 6), ("test", 6)):
        for lab in labels:
            rows += [{"id": f"{split}-{lab}-{i}", "text": "t", "label": lab, "split": split}
                     for i in range(per)]  # fmt: skip
    df = pd.DataFrame(rows)
    P = ti.Paths(tmp_path / "res", tmp_path / "out", tmp_path / "log.jsonl", False)
    cfg = {"c1": {"fusion_components": ["maha_ft", "neg_energy"]}}
    tb = {"ood": {"knn_k": [1, 5]}}
    ctx = ti.Ctx(cfg, {}, tb, P, df, False, False)
    ctx.fz = {i: rng.normal(size=8) for i in df["id"]}
    sets = build_holdout_sets(df, ["d"])
    k, hid, n_layers = 3, 8, 2
    arrays: dict[str, np.ndarray] = {}
    parts = (
        ("train", sets.train),
        ("cal", sets.cal),
        ("ek", sets.eval_known),
        ("unk", sets.eval_unknown),
    )
    for part, frame in parts:
        n = len(frame)
        arrays[f"{part}_ids"] = frame["id"].to_numpy(dtype=str)
        arrays[f"{part}_logits"] = rng.normal(size=(n, k)).astype(np.float32)
        arrays[f"{part}_features"] = rng.normal(size=(n, hid)).astype(np.float32)
        arrays[f"{part}_hidden"] = rng.normal(size=(n, n_layers, hid)).astype(np.float32)
    return ctx, sets, arrays


def test_score_run_layout_and_ensemble(tmp_path: Path) -> None:
    ctx, sets, arrays = _synthetic_ctx(tmp_path)
    table, info = ti.score_run(ctx, sets, arrays)
    assert list(table.columns) == [
        "id", "split", "set", "is_unknown", "gold", "pred", *ctx.all_methods
    ]  # fmt: skip
    assert set(table["set"]) == {"cal", "eval"} and table[ctx.all_methods].notna().all().all()
    cal = table[table["set"] == "cal"]
    assert len(cal) == len(sets.cal) and not cal["is_unknown"].any()
    # fusion = mean of cal-known ECDF ranks: on the cal rows themselves it lies in (0, 1]
    assert cal["fuse_maha_energy"].between(0, 1).all()
    assert len(info["multi_layer_cal_mu"]) == 2
    ens = ti.ensemble_table([table, table], [arrays, arrays], sets.labels)
    assert list(ens.columns) == ["id", "split", "set", "is_unknown", "gold", "pred",
                                 "ens_maha", "ens_entropy"]  # fmt: skip
    np.testing.assert_allclose(ens["ens_maha"], table["maha_ft"])  # mean of identical seeds
    assert (ens["ens_entropy"] <= 0).all()  # negated entropy


# ----------------------------------------------- select -> confirm -> business (synthetic CSVs)
CONFIRM = ["k1", "k2", "k3", "k4", "k5"]


def _fake_scores(ctx: ti.Ctx, holdout: tuple[str, ...], seed: int, gap: dict[str, float]) -> Any:
    rng = np.random.default_rng(seed)
    labels = ti.labels_for(ctx, holdout)
    parts = []
    for st, n, unk in (("cal", 40, False), ("eval", 40, False), ("eval", 30, True)):
        t = pd.DataFrame(
            {
                "id": [f"{st}{int(unk)}-{i}-{seed}" for i in range(n)],
                "split": "val" if st == "cal" else "test",
                "set": st,
                "is_unknown": unk,
                "gold": holdout[0] if unk else rng.choice(labels, n),
                "pred": rng.choice(labels, n),
            }
        )
        for m in ctx.all_methods:
            t[m] = rng.normal(0.0 if unk else gap.get(m, 1.0), 1.0, n)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def test_select_confirm_business_end_to_end(tmp_path: Path) -> None:
    labels = [*DEV, *CONFIRM, "other", "chitchat"]
    df = pd.DataFrame({"id": [f"x{i}" for i in range(len(labels))], "label": labels})
    cfg: dict[str, Any] = {
        "dev_classes": DEV, "confirm_classes": CONFIRM, "loco_seed": 42,
        "headline_seeds": [42, 43], "c1": {"fusion_components": ["maha_ft", "neg_energy"]},
        "c2": {"lambdas": [0.1, 0.5], "temperature": 0.1},
        "ood": {"op_retentions": [0.95, 0.90], "curve": {"lo": 0.8, "hi": 0.99, "step": 0.01},
                "safe_labels": ["other", "chitchat"]},
        "bootstrap": {"n_resamples": 200, "seed": 42, "level": 0.95},
        "business": {"messages": 1000, "prevalence": [0.01, 0.05, 0.10], "retention": [0.90, 0.95]},
        "wandb": {"group_prefix": "trackB-improve"},
    }  # fmt: skip
    fcfg = {"train": {"model_name": "m", "lr": 1e-4, "query_prefix": ""}}
    P = ti.Paths(tmp_path / "res", tmp_path / "out", tmp_path / "log.jsonl", False)
    ctx = ti.Ctx(
        cfg,
        fcfg,
        {"ood": {"knn_k": [1, 5]}, "headline": {"holdout": ["k4", "k5"]}},
        P,
        df,
        False,
        False,
    )
    base, c2a, c2b = ti.base_candidate(ctx), *ti.c2_candidates(ctx)
    ti.write_json(
        ti.guard_path(ctx, c2a.key),
        {"passed": False, "deployed_epoch": 7, "why": "F1 0.9 < 0.9341"},
    )
    ti.write_json(ti.guard_path(ctx, c2b.key), {"passed": True, "deployed_epoch": 5, "why": None})
    c2b = ti.c2_candidates(ctx)[1]  # now carries its deployed epoch
    assert c2b.stop_epoch == 5 and c2b.cid == "c2_l0p5_"
    strong = {m: 3.0 for m in ctx.all_methods}  # the C2 model separates known/unknown better
    n = 0
    for c in [*DEV, *CONFIRM]:
        for cand, gap in ((base, {"maha_ml": 1.2}), (c2b, strong)):
            n += 1
            tab = _fake_scores(ctx, (c,), n, gap)
            ti._save_csv(tab, P.scores / f"{ti.run_id_for(cand, 'loco', 42, c)}.csv")
    for cand, gap in ((base, {}), (c2b, strong)):
        for s in cfg["headline_seeds"]:
            n += 1
            tab = _fake_scores(ctx, ("k4", "k5"), n, gap)
            ti._save_csv(tab, P.scores / f"{ti.run_id_for(cand, 'headline', s)}.csv")
    ti.stage_select(ctx)
    sel = json.loads((P.results / "select.json").read_text())
    assert sel["winner"].startswith("c2_l0.5/") and sel["ship"]
    assert "c2_l0.1" in sel["excluded_by_guard"]
    assert set(sel["dev_per_class_auroc"][ti.CURRENT_KEY]) == set(DEV)  # DEV classes only
    ti.stage_confirm(ctx)
    conf = json.loads((P.results / "confirm.json").read_text())
    assert conf["shipped"] == sel["winner"] and set(conf["entries"]) == {
        ti.CURRENT_KEY,
        sel["winner"],
    }
    e = conf["entries"][sel["winner"]]["confirm"]
    assert set(e["per_unit"]) == set(CONFIRM)
    assert e["ci95"]["auroc"]["lo"] <= e["mean"]["auroc"] <= e["ci95"]["auroc"]["hi"]
    assert e["mean"]["op95"]["retention_known"] == pytest.approx(
        np.mean([u["op95"]["retention_known"] for u in e["per_unit"].values()])
    )
    assert len(e["curve_mean"]["target"]) == 20
    assert conf["paired_delta_winner_minus_current"]["confirm"]["auroc"]["mean_delta"] > 0
    assert (P.results / "confirm_curves.png").exists()
    ti.stage_business(ctx)
    biz = json.loads((P.results / "business.json").read_text())
    assert biz["label"] == "estimate from measured CONFIRM rates" and len(biz["rows"]) == 6
    r = biz["rows"][0]  # retention 0.90, pi 0.01, measured retention of the shipped method
    assert r["known_wrongly_abstained"] == pytest.approx(990 * (1 - r["retention_measured"]))


# ------------------------------------------------------------------ final_ood (stubbed model)
def test_final_ood_logs_one_call_and_checks_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    from intent_router import data as data_mod

    labels = ["a", "b", "c", "d"]
    monkeypatch.setattr(data_mod, "LABELS", labels)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, tok = _tiny_model(4), _StubTokenizer()
    monkeypatch.setattr(ti, "build_model", lambda *_a: (tok, model))
    monkeypatch.setattr(ti, "gpu_section", lambda *_a: contextlib.nullcontext({}))
    rows = [
        {"id": f"{sp}-{lab}-{i}", "text": f"w{lab}{i} x y z", "label": lab, "split": sp}
        for sp, per in (("train", 10), ("val", 5), ("test", 5))
        for lab in labels
        for i in range(per)
    ]
    df = pd.DataFrame(rows)
    layers, ml = [0, -1], 16

    def sorted_split(sp: str) -> pd.DataFrame:
        return df[df["split"] == sp].sort_values("id").reset_index(drop=True)

    out = {sp: predict_hidden(model, tok, sorted_split(sp)["text"].tolist(), "m", ml, 8, layers)
           for sp in ("train", "val", "test")}  # fmt: skip
    art = tmp_path / "art"
    art.mkdir()
    np.savez(art / "fl.npz", val_logits=out["val"][0], val_features=out["val"][1],
             train_features=out["train"][1])  # fmt: skip
    test = sorted_split("test")
    pd.DataFrame({"id": test["id"], "pred": out["test"][0].argmax(1)}).to_csv(
        art / "pred.csv", index=False
    )
    cfg = {
        "final_test_log": str(tmp_path / "final_log.jsonl"), "final_model_dir": "unused",
        "final_artifacts": {"features_logits": str(art / "fl.npz"),
                            "test_predictions": str(art / "pred.csv")},
        "c1": {"layers": layers, "fusion_components": ["maha_ft", "neg_energy"]},
        "gpu_lock_poll_s": 1,
    }  # fmt: skip
    fcfg = {"predict_batch_size": 8, "train": {"model_name": "m", "max_len": ml, "lr": 1e-4}}
    P = ti.Paths(tmp_path / "res", tmp_path / "out", tmp_path / "log.jsonl", False)
    ctx = ti.Ctx(cfg, fcfg, {"ood": {"retention": 0.95}}, P, df, False, False)
    ti.write_json(P.results / "select.json", {"shipped": "base/maha_ml"})
    ti.stage_final_ood(ctx)
    res = json.loads((P.results / "final_ood.json").read_text())
    assert res["status"] == "done" and res["method"] == "maha_ml"
    assert (
        res["test_predictions_match_saved_csv"] and res["max_abs_diff_vs_saved"]["val_logits"] == 0
    )
    assert 0.0 < res["val_retention_achieved"] <= 1.0 and res["n_test"] == len(test)
    log = [json.loads(x) for x in (tmp_path / "final_log.jsonl").read_text().splitlines()]
    assert [e["call_type"] for e in log] == ["ood_feature_extraction"]  # exactly one test call
    with pytest.raises(SystemExit, match="second"):  # a repeat is refused (and writes nothing)
        (P.results / "final_ood.json").unlink()
        ti.stage_final_ood(ctx)
    # shipped method that changes weights => TODO json, no extraction
    ti.write_json(P.results / "select.json", {"shipped": "c2_l0.5/maha_ft"})
    ti.stage_final_ood(ctx)
    assert json.loads((P.results / "final_ood.json").read_text())["status"] == "todo"
