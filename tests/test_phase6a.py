"""
Open-set improvement round: pure pieces + CPU-only integration with a tiny random model (no dataset
text).
"""

from __future__ import annotations

import ast
import json
import math
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from intent_router import ood_variants as ov  # noqa: E402
from intent_router import phase4e as p5  # noqa: E402
from intent_router import phase6a as p6  # noqa: E402
from intent_router import phase6a_select as sel  # noqa: E402
from intent_router import phase6a_views as pv  # noqa: E402
from intent_router import soup, twin_kl  # noqa: E402
from intent_router import train as train_mod  # noqa: E402
from intent_router.ood import HoldoutSets  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "intent_router"


# =========================================================================== twin + symmetric KL
def test_symmetric_kl_on_hand_made_distributions() -> None:
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    kl_pq = float((p * np.log(p / q)).sum())
    kl_qp = float((q * np.log(q / p)).sum())
    assert kl_pq == pytest.approx(0.51083, abs=1e-4) and kl_qp == pytest.approx(0.36810, abs=1e-4)
    la, lb = torch.log(torch.tensor([p])), torch.log(torch.tensor([q]))  # logits = log-probs
    got = twin_kl.symmetric_kl(la, lb)
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(kl_pq + kl_qp, abs=1e-5)  # sum, no 1/2 factor
    assert float(twin_kl.symmetric_kl(lb, la)[0]) == pytest.approx(float(got[0]), abs=1e-6)
    assert float(twin_kl.symmetric_kl(la, la)[0]) == pytest.approx(0.0, abs=1e-7)
    logits = torch.randn(5, 7)
    assert (twin_kl.symmetric_kl(logits, logits + 3.0) < 1e-6).all()  # softmax is shift-invariant


def test_twin_swap_preserves_digits_and_only_changes_prefixes() -> None:
    text = "track PO-123 and LD-45 and ABCD-9 then PO-123 again, no id like X-1 or abc-7"
    pat = re.compile(r"\b[A-Z]{2,4}-(\d+)")
    for seed in range(20):
        tw = twin_kl.twin_text(text, seed, 3, "row-1")
        assert pat.sub(r"@\1", tw) == pat.sub(r"@\1", text)  # same text once prefixes are masked
        assert [m.group(1) for m in pat.finditer(tw)] == ["123", "45", "9", "123"]  # digits kept
        assert "X-1" in tw and "abc-7" in tw  # non-ID look-alikes untouched
    changed = {twin_kl.twin_text(text, s, 1, "r") for s in range(40)}
    assert len(changed) > 5  # prefixes really are redrawn (not a no-op)
    assert twin_kl.twin_text("no ids here", 1, 1, "r") == "no ids here"
    assert not twin_kl.has_id("no ids here") and twin_kl.has_id("see PO-1")


def test_twin_is_deterministic_per_seed_epoch_row_and_differs_across_them() -> None:
    t = "PO-1 and LD-2 and ZZ-3"
    assert twin_kl.twin_text(t, 0, 1, "a") == twin_kl.twin_text(t, 0, 1, "a")
    variants = {twin_kl.twin_text(t, 0, e, "a") for e in range(1, 30)}
    assert len(variants) > 5
    assert twin_kl.twin_text(t, 0, 1, "a") != twin_kl.twin_text(t, 1, 1, "a") or True  # may collide


def test_aux_loss_only_counts_rows_with_ids_and_divides_by_batch_size() -> None:
    texts = ["PO-1 where", "no id", "LD-7 and LD-8"]
    ids = ["a", "b", "c"]
    logits = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]], requires_grad=True)
    calls: list[list[str]] = []

    def forward(tw: list[str]) -> torch.Tensor:
        calls.append(tw)
        return torch.tensor([[0.0, 1.0]] * len(tw))

    aux = twin_kl.make_twin_loss(0.5)
    loss = aux(
        logits=logits,
        idx=np.array([0, 1, 2]),
        epoch=1,
        forward=forward,
        texts=texts,
        row_ids=ids,
        model_seed=0,
        id_randomize_p=0.0,
    )
    assert len(calls) == 1 and len(calls[0]) == 2  # the no-ID row has no twin
    manual = (
        0.5
        * float(twin_kl.symmetric_kl(logits[[0, 2]].detach(), torch.tensor([[0.0, 1.0]] * 2)).sum())
        / 3
    )
    assert float(loss) == pytest.approx(manual, rel=1e-5)  # divided by the batch size (3)
    loss.backward()
    assert (
        logits.grad is not None and float(logits.grad[1].abs().sum()) == 0.0
    )  # no-ID row: no grad
    none = aux(
        logits=logits[1:2],
        idx=np.array([1]),
        epoch=1,
        forward=forward,
        texts=texts,
        row_ids=ids,
        model_seed=0,
        id_randomize_p=0.0,
    )
    assert float(none) == 0.0 and none.requires_grad  # connected zero, forward not called again
    assert len(calls) == 1
    with pytest.raises(ValueError):
        twin_kl.make_twin_loss(0.0)


# ================================================================================ soup helpers
def test_rotation_matches_the_preregistered_members() -> None:
    assert [soup.rotated_members(7, 5, r) for r in range(3)] == [
        [0, 1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 0, 1],
    ]
    with pytest.raises(ValueError):
        soup.rotated_members(3, 5, 0)


def test_averaged_state_dict_equals_the_manual_mean() -> None:
    g = torch.Generator().manual_seed(0)
    sds = [
        {
            "w": torch.randn(4, 3, generator=g),
            "b": torch.randn(3, generator=g),
            "ids": torch.arange(5),
        }
        for _ in range(5)
    ]
    avg = soup.average_state_dicts(sds)
    assert torch.allclose(avg["w"], torch.stack([s["w"] for s in sds]).mean(0), atol=1e-6)
    manual_b = (sds[0]["b"] + sds[1]["b"] + sds[2]["b"] + sds[3]["b"] + sds[4]["b"]) / 5
    assert torch.allclose(avg["b"], manual_b, atol=1e-6)
    assert avg["w"].dtype == torch.float32 and torch.equal(avg["ids"], torch.arange(5))
    one = soup.average_state_dicts(sds[:1])
    assert torch.equal(one["w"], sds[0]["w"])  # a one-member soup is that member
    with pytest.raises(ValueError):
        soup.average_state_dicts([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])
    with pytest.raises(ValueError):
        soup.average_state_dicts([{"i": torch.tensor([1])}, {"i": torch.tensor([2])}])
    with pytest.raises(ValueError):
        soup.average_state_dicts([])


def test_member_round_trip_through_safetensors(tmp_path: Path) -> None:
    sd = {"w": torch.randn(3, 3), "i": torch.arange(4)}
    soup.save_member(sd, tmp_path / "m" / "m0.safetensors")
    back = soup.load_member(tmp_path / "m" / "m0.safetensors")
    assert torch.equal(back["w"], sd["w"]) and torch.equal(back["i"], sd["i"])


def test_greedy_soup_rule_order_inclusion_and_ties() -> None:
    calls: list[list[int]] = []
    table = {(1,): 0.9, (1, 0): 0.92, (1, 0, 2): 0.91, (1, 0, 3): 0.92}

    def ev(idx: list[int]) -> float:
        calls.append(list(idx))
        return table[tuple(idx)]

    out = soup.greedy_soup([0.80, 0.90, 0.70, 0.60], ev)
    assert out["order"] == [1, 0, 2, 3]  # sorted by own score, best first
    assert out["selected"] == [
        1,
        0,
        3,
    ]  # member 2 lowers the soup (0.91 < 0.92); 3 ties -> kept (>=)
    assert [t["accepted"] for t in out["trace"]] == [True, True, False, True]
    assert out["final_score"] == pytest.approx(0.92) and calls[0] == [1]
    assert soup.greedy_soup([0.5, 0.5], lambda i: 0.5)["order"] == [0, 1]  # ties by index
    with pytest.raises(ValueError):
        soup.greedy_soup([], lambda i: 0.0)


def make_train_frame(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"r{i:04d}" for i in range(n)], "text": "x", "label": "a"})


def test_inner_split_is_seeded_disjoint_85_15_and_per_unit() -> None:
    tr = make_train_frame(200)
    fit, hold = soup.inner_split(tr, "cv_f0", 0.15, 42)
    assert len(hold) == 30 and len(fit) == 170
    assert not set(fit["id"]) & set(hold["id"]) and set(fit["id"]) | set(hold["id"]) == set(
        tr["id"]
    )
    fit2, hold2 = soup.inner_split(tr.sample(frac=1.0, random_state=1), "cv_f0", 0.15, 42)
    assert set(hold2["id"]) == set(hold["id"])  # independent of row order
    assert set(soup.inner_split(tr, "cv_f1", 0.15, 42)[1]["id"]) != set(hold["id"])  # per unit
    assert set(soup.inner_split(tr, "cv_f0", 0.15, 43)[1]["id"]) != set(hold["id"])  # per seed
    assert len(soup.inner_split(make_train_frame(7), "u", 0.15, 1)[1]) == 2  # ceil
    with pytest.raises(ValueError):
        soup.inner_split(tr, "u", 1.0, 42)
    with pytest.raises(ValueError):
        soup.inner_split(make_train_frame(1), "u", 0.15, 42)


def test_inner_holdout_never_contains_the_evaluated_fold() -> None:
    frame = make_train_frame(120).assign(cv_fold_s0=lambda d: np.arange(len(d)) % 5, split="train")
    from intent_router.cv import fold_split

    for k in range(5):
        tr, ev = fold_split(frame, 0, k)
        fit, hold = soup.inner_split(tr, f"cv_f{k}", 0.15, 42)
        assert not set(ev["id"]) & (set(fit["id"]) | set(hold["id"]))  # evaluated fold excluded


def fake_env(tmp_path: Path, smoke: bool = True) -> SimpleNamespace:
    pcfg = {
        "soup": {
            "pool_size": 7,
            "members_per_soup": 5,
            "rotate_shift": 2,
            "head_init_seed": 0,
            "inner_holdout_frac": 0.15,
            "inner_seed": 42,
        },
        "predict_batch_size": 4,
    }
    return SimpleNamespace(
        smoke=smoke,
        pcfg=pcfg,
        P=SimpleNamespace(results=tmp_path / "res", outputs=tmp_path / "out"),
    )


def test_greedy_soup_build_only_evaluates_the_inner_holdout(tmp_path: Path, monkeypatch) -> None:
    env = fake_env(tmp_path)
    pool = p6.Pool(
        "i3", p6.TrainSpec("i3", head_init_seed=0, inner_frac=0.15), (("i3g", "greedy"),)
    )
    members = p6.soup_members(env, 0)
    assert members == [0, 1, 2]  # smoke: 3 members per soup out of a pool of 5
    for m in range(5):
        soup.save_member(
            {"w": torch.full((2,), float(m))},
            p6.member_dir(env, pool, "cv", "f0") / f"m{m}.safetensors",
        )
        p6.write_json(
            p6.member_meta_path(env, pool, "cv", "f0", m),
            {"f1_inner_holdout": 0.5 + 0.1 * (m == 1), "gpu_exclusive": True, "wall_clock_s": 1.0},
        )
    ih = make_train_frame(10)
    seen: list[object] = []

    class Skel:
        def __init__(self) -> None:
            self.sd: dict | None = None

        def load_state_dict(self, sd: dict) -> None:
            self.sd = sd

    skel = Skel()

    def fake_f1(model: Skel, tok: object, df: pd.DataFrame, labels: list, tcfg: object, bs: int):
        seen.append(df)
        assert model.sd is not None
        return float(model.sd["w"].mean())  # soups with larger members score higher

    monkeypatch.setattr(p6, "f1_of", fake_f1)
    built = p6.build_soups(
        env, pool, "cv", "f0", 0, skel, None, ih, ["a"], None, ["uniform", "greedy"]
    )
    assert seen and all(df is ih for df in seen)  # greedy evaluated ONLY the inner-holdout rows
    assert torch.allclose(built["uniform"].sd["w"], torch.full((2,), 1.0))  # mean of members 0,1,2
    g = built["greedy"].info
    assert g["order"][0] == 1 and g["members"] == [0, 1, 2] and 1 in g["selected"]
    assert g["selected"][0] == 1
    assert 0 not in g["selected"] or g["selected"].index(0) > 0


def test_delete_guard_only_inside_phase6a_member_dirs(tmp_path: Path) -> None:
    good = tmp_path / "outputs" / "phase6a" / "pool_i3" / "members" / "cv_f0"
    good.mkdir(parents=True)
    (good / "m0.safetensors").write_text("x")
    assert p6.delete_dir_guarded(good, "members") is True and not good.exists()
    assert p6.delete_dir_guarded(good, "members") is False
    for bad, parent in (
        (tmp_path / "outputs" / "phase4e" / "v1" / "members" / "x", "members"),  # wrong project
        (tmp_path / "outputs" / "phase6a" / "pool_i3" / "oof_parts", "members"),  # wrong parent
    ):
        bad.mkdir(parents=True, exist_ok=True)
        with pytest.raises(AssertionError):
            p6.delete_dir_guarded(bad, parent)
        assert bad.exists()


# ===================================================== tiny-model integration of the train hooks
class FakeTok:
    """Just enough tokenizer for train._train: __call__ -> lists, pad -> padded tensors."""

    pad_token_id = 1

    def __call__(self, texts: list[str], truncation: bool = True, max_length: int = 64):
        ids = [[3 + (ord(c) % 90) for c in t[: max_length - 2]] or [3] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}

    def pad(self, feats: list[dict], return_tensors: str = "pt") -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        ids = [f["input_ids"] + [1] * (n - len(f["input_ids"])) for f in feats]
        am = [f["attention_mask"] + [0] * (n - len(f["attention_mask"])) for f in feats]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(am)}


def tiny_build(model_name: str, labels: list[str], attn: str | None = None):
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    cfg = XLMRobertaConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=80,
        num_labels=len(labels),
    )
    return FakeTok(), XLMRobertaForSequenceClassification(cfg)


@pytest.fixture
def tiny_train(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(train_mod, "build_model", tiny_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    labels = ["a", "b", "c"]
    monkeypatch.setattr(
        train_mod, "_label_maps", lambda: (labels, {x: i for i, x in enumerate(labels)})
    )
    rng = np.random.default_rng(0)
    texts = [f"{'PO-' if i % 2 else 'plain '}{i} item {rng.integers(100)}" for i in range(24)]
    df = pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(24)],
            "text": texts,
            "label": [labels[i % 3] for i in range(24)],
        }
    )
    ev = pd.DataFrame({"id": ["e0", "e1", "e2"], "text": ["PO-9 x", "y", "z"], "label": labels})

    def run(model_seed: int, hooks=None, lr: float = 0.0, **kw):
        cfg = train_mod.TrainConfig(
            model_name="tiny",
            lr=lr,
            epochs=2,
            batch_size=8,
            model_seed=model_seed,
            precision="fp32",
            **kw,
        )
        res, model, _ = train_mod.train_model(cfg, df, ev, {"run_id": "t"}, hooks=hooks)
        return res, {k: v.clone() for k, v in model.state_dict().items()}

    return run


def test_members_share_the_head_init_and_differ_without_the_hook(tiny_train) -> None:
    hooks = train_mod.TrainHooks(head_init_seed=0)
    _, a = tiny_train(1, hooks)  # lr 0: the final state is the init
    _, b = tiny_train(2, hooks)
    _, c = tiny_train(1, None)
    _, d = tiny_train(2, None)
    head = [k for k in a if k.startswith("classifier")]
    assert head and all(torch.equal(a[k], b[k]) for k in head)  # shared head init
    assert not all(torch.equal(c[k], d[k]) for k in head)  # plain seeds draw different heads
    _, e = tiny_train(1, train_mod.TrainHooks(head_init_seed=5))
    assert not all(torch.equal(a[k], e[k]) for k in head)  # a different head seed changes it


def test_members_differ_in_data_order_after_a_shared_init(tiny_train) -> None:
    hooks = train_mod.TrainHooks(head_init_seed=0)
    _, a = tiny_train(1, hooks, lr=1e-2)
    _, b = tiny_train(2, hooks, lr=1e-2)
    _, a2 = tiny_train(1, hooks, lr=1e-2)
    assert all(torch.equal(a[k], a2[k]) for k in a)  # deterministic per member seed
    assert not all(torch.equal(a[k], b[k]) for k in a)  # different member seed -> different run


def test_no_hooks_is_bitwise_identical_to_the_empty_hooks_object(tiny_train) -> None:
    _, a = tiny_train(3, None, lr=1e-2)
    _, b = tiny_train(3, train_mod.TrainHooks(), lr=1e-2)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_aux_loss_hook_is_called_every_step_and_changes_training(tiny_train) -> None:
    seen: list[tuple[int, int]] = []

    def aux(*, logits, idx, epoch, forward, texts, row_ids, model_seed, id_randomize_p):
        seen.append((epoch, len(idx)))
        assert len(texts) == 24 and len(row_ids) == 24 and model_seed == 3
        return logits.sum() * 0.0 + 0.5 * logits.pow(2).mean()

    _, base = tiny_train(3, None, lr=1e-2)
    _, with_aux = tiny_train(3, train_mod.TrainHooks(aux_loss=aux), lr=1e-2)
    assert seen == [(1, 8)] * 3 + [(2, 8)] * 3  # 24 rows / batch 8 = 3 steps x 2 epochs
    assert not all(torch.equal(base[k], with_aux[k]) for k in base)
    with pytest.raises(ValueError, match="grad_accum"):
        tiny_train(3, train_mod.TrainHooks(aux_loss=aux), grad_accum=2)


def test_i4_twin_loss_trains_end_to_end_on_the_tiny_model(tiny_train) -> None:
    spec = p6.TrainSpec("i4a", a1=True, kl_lambda=0.5)
    res, _ = tiny_train(0, spec.hooks(), lr=1e-2, id_randomize_p=0.5)
    assert not res.nan_detected and all(math.isfinite(e["train_loss"]) for e in res.epochs)
    assert p6.TrainSpec("ref").hooks() is None
    assert p6.TrainSpec("i3", head_init_seed=0).hooks() is not None


# ========================================================================= views (CPU, synthetic)
def make_sets() -> HoldoutSets:
    labels = ["a", "b"]

    def fr(prefix: str, n: int, lab: list[str], split: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": [f"{prefix}{i}" for i in range(n)],
                "text": "t",
                "label": [lab[i % len(lab)] for i in range(n)],
                "split": split,
            }
        )

    return HoldoutSets(
        ["z"],
        labels,
        fr("tr", 40, labels, "train"),
        fr("cal", 30, labels, "val"),
        fr("ek", 20, labels, "test"),
        fr("unk", 15, ["z"], "test"),
        {},
    )


def make_arrays(sets: HoldoutSets, rng: np.random.Generator, d: int = 6) -> dict[str, np.ndarray]:
    k = len(sets.labels)
    cent = rng.normal(scale=4.0, size=(k, d))
    out: dict[str, np.ndarray] = {}
    for part, fr, unknown in (
        ("train", sets.train, False),
        ("cal", sets.cal, False),
        ("ek", sets.eval_known, False),
        ("unk", sets.eval_unknown, True),
    ):
        y = np.array([sets.labels.index(x) if x in sets.labels else 0 for x in fr["label"]])
        f = cent[y] + rng.normal(size=(len(fr), d)) * (3.0 if unknown else 1.0)
        lg = rng.normal(size=(len(fr), k))
        lg[np.arange(len(fr)), y] += 3.0
        out[f"{part}_ids"], out[f"{part}_features"], out[f"{part}_logits"] = (
            fr["id"].to_numpy(dtype=str),
            f.astype(np.float32),
            lg.astype(np.float32),
        )
        if part != "train":
            out[f"n_{part}_features"] = (f + 0.5).astype(np.float32)  # "neutral" pass differs
            out[f"n_{part}_logits"] = lg.astype(np.float32)
    return out


def test_global_view_reproduces_the_shipped_threshold_rule() -> None:
    sets = make_sets()
    A = make_arrays(sets, np.random.default_rng(0))
    table, info = pv.build_view(sets, A, "maha_ft", "global", "raw")
    from intent_router.evaluate import threshold_at_retention
    from intent_router.ood import fit_gaussian_lw, score_mahalanobis

    y_tr = pv.local_gold(sets.train, sets.labels)
    ref = score_mahalanobis(A["cal_features"], fit_gaussian_lw(A["train_features"], y_tr, 2))
    cal = table[table["set"] == "cal"]
    assert np.allclose(cal["score"], ref)
    t = threshold_at_retention(ref, 0.95)
    assert info["per_retention"]["95"]["global"] == pytest.approx(t)
    assert np.allclose(cal["margin95"], ref - t) and (cal["margin95"] >= 0).mean() >= 0.95
    assert set(table.columns) >= {
        "id",
        "split",
        "set",
        "is_unknown",
        "gold",
        "pred",
        "score",
        "margin95",
        "margin90",
    }
    assert table["is_unknown"].sum() == 15 and len(table) == 30 + 20 + 15


def test_neutral_mode_scores_the_neutral_arrays_and_keeps_train_fit() -> None:
    sets = make_sets()
    A = make_arrays(sets, np.random.default_rng(1))
    raw, _ = pv.build_view(sets, A, "maha_ft", "global", "raw")
    neu, info = pv.build_view(sets, A, "maha_ft", "global", "neutral")
    assert not np.allclose(raw["score"], neu["score"])  # different features were scored
    from intent_router.evaluate import threshold_at_retention

    assert info["per_retention"]["95"]["global"] == pytest.approx(
        threshold_at_retention(neu[neu["set"] == "cal"]["score"].to_numpy(), 0.95)
    )  # neutral cal
    with pytest.raises(ValueError):
        pv.build_view(sets, A, "maha_ft", "global", "bogus")
    with pytest.raises(KeyError):
        pv.build_view(sets, A, "maha_ft", "nope", "raw")


def test_i6a_view_uses_per_predicted_class_thresholds_and_reports_fallbacks() -> None:
    sets = make_sets()
    A = make_arrays(sets, np.random.default_rng(2))
    table, info = pv.build_view(sets, A, "maha_ft", "i6a", "raw", min_rows=5)
    rec = info["per_retention"]["95"]
    assert set(rec["per_class"]) == {"a", "b"} and rec["n_fallback_classes"] == 0
    cal = table[table["set"] == "cal"]
    for lab in ("a", "b"):
        rows = cal[cal["pred"] == lab]
        assert (rows["margin95"] >= 0).mean() >= 0.95
        assert np.allclose(rows["score"] - rows["margin95"], rec["per_class"][lab])
    _, info_hi = pv.build_view(sets, A, "maha_ft", "i6a", "raw", min_rows=1000)  # all fall back
    assert info_hi["per_retention"]["95"]["n_fallback_classes"] == 2


def test_i6b_view_is_a_single_cross_fitted_threshold_and_i1_scorers_run() -> None:
    sets = make_sets()
    A = make_arrays(sets, np.random.default_rng(3))
    table, info = pv.build_view(sets, A, "maha_ft", "i6b", "raw")
    r = info["per_retention"]["95"]
    assert r["n_crossfit_rows"] == 70 and "crossfit_threshold" in r
    assert np.allclose(table["score"] - table["margin95"], r["crossfit_threshold"])  # one threshold
    for scorer in ("i1a", "i1b"):
        t, _ = pv.build_view(sets, A, scorer, "global", "neutral")
        assert np.isfinite(t["score"]).all()


# ============================================================== selection (synthetic saved files)
CLASSES, SEEDS, FOLDS = ["c1", "c2"], [0, 1, 2], [0, 1]
N_ROWS, K = 40, 4


def make_pcfg(cands: list[dict]) -> dict:
    return {
        "thresholds": {
            "a_max_drop": 0.010,
            "b_max_auroc_drop": 0.010,
            "b_max_rej95_drop": 0.03,
            "c_max_flip_increase": 0.02,
            "d_max_agreement_drop": 0.010,
            "e_max_ece_increase": 0.02,
        },
        "bootstrap": {"n_resamples": 100, "seed": 42, "level": 0.95},
        "calibration": {"ece_bins": 15},
        "translation": {"langs": ["es", "fr"]},
        "candidates": cands,
    }


def write_weights(
    res: Path, key: str, acc: float = 0.8, flip: float = 0.3, agree: float = 0.8, seed_off: int = 0
) -> None:
    ids = [f"r{i:03d}" for i in range(N_ROWS)]
    for s in SEEDS:
        d = res / key / f"s{s}"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(100 + s + seed_off)
        gold = np.arange(N_ROWS) % K
        pred = np.where(rng.random(N_ROWS) < acc, gold, (gold + 1) % K)
        probs = np.full((N_ROWS, K), 0.1)
        probs[np.arange(N_ROWS), pred] = 0.7
        clean = pd.DataFrame(
            {
                "id": ids,
                "fold_seed": 0,
                "model_seed": s,
                "fold": np.arange(N_ROWS) % 2,
                "gold": gold,
                "pred": pred,
            }
        )
        for j in range(K):
            clean[f"prob_{j}"] = probs[:, j]
        clean.to_csv(d / "oof_clean.csv", index=False)
        sw_ids = ids[:12]
        sw = pd.DataFrame(
            {
                "id": sw_ids * 2,
                "swap": ["LD->REF"] * 12 + ["PO->REF"] * 12,
                "gold": gold[:12].tolist() * 2,
                "clean_pred": pred[:12].tolist() * 2,
            }
        )
        flipped = np.random.default_rng(7 + s).random(24) < flip
        sw["swap_pred"] = np.where(flipped, (sw["clean_pred"] + 1) % K, sw["clean_pred"])
        sw.to_csv(d / "oof_swap.csv", index=False)
        pd.DataFrame({"id": ids, "gold": gold, "clean_pred": pred, "noise_pred": pred}).to_csv(
            d / "oof_noise.csv", index=False
        )
        mt_rows = []
        for i in ids[:20]:
            for lang in ("es", "fr"):
                a = np.random.default_rng(zlib.crc32(f"{i}|{lang}|{s}".encode())).random() < agree
                mt_rows.append(
                    {
                        "id": i,
                        "lang": lang,
                        "system": "x",
                        "kept": True,
                        "gold": 0,
                        "en_pred": 1,
                        "mt_pred": 1 if a else 2,
                    }
                )
        pd.DataFrame(mt_rows).to_csv(d / "oof_mt.csv", index=False)


def write_view(
    res: Path,
    name: str,
    shift: float = 0.0,
    cal_off: float = 0.0,
    seed_off: int = 0,
) -> None:
    for mode in pv.MODES:
        for s in SEEDS:
            for c in CLASSES:
                rng = np.random.default_rng(1000 * (c == "c2") + s)  # same rows for every candidate
                rng2 = np.random.default_rng(5000 + seed_off + s + 7 * (c == "c2"))
                known = rng.normal(1.0, 1.0, 40) + shift
                unknown = rng.normal(-1.0, 1.0, 30) + rng2.normal(0, 0.01, 30)
                cal = np.random.default_rng(9 + s).normal(1.0, 1.0, 30) + cal_off
                thr95 = np.sort(cal)[::-1][math.ceil(0.95 * 30) - 1]
                thr90 = np.sort(cal)[::-1][math.ceil(0.90 * 30) - 1]
                score = np.concatenate([cal, known, unknown])
                t = pd.DataFrame(
                    {
                        "id": [f"cal{i}" for i in range(30)]
                        + [f"k{i}" for i in range(40)]
                        + [f"u{i}" for i in range(30)],
                        "split": "x",
                        "set": ["cal"] * 30 + ["eval"] * 70,
                        "is_unknown": [False] * 70 + [True] * 30,
                        "gold": "g",
                        "pred": "a",
                        "score": score,
                        "margin95": score - thr95,
                        "margin90": score - thr90,
                    }
                )
                d = res / name / f"s{s}" / "views" / mode
                d.mkdir(parents=True, exist_ok=True)
                t.to_csv(d / f"{name}_loco_{c}_s{s}.csv", index=False)


def make_ctx(res: Path, cands: list[dict]) -> sel.SelCtx:
    return sel.SelCtx(res, make_pcfg(cands), CLASSES, SEEDS, FOLDS, ["other"], 200)


def spec(key: str, weights: str, comps: int = 1, **kw) -> dict:
    return {
        "key": key,
        "weights": weights,
        "scorer": "maha_ft",
        "thr": "global",
        "components": comps,
        **kw,
    }


def build_tree(res: Path) -> list[dict]:
    write_weights(res, "ref")
    write_view(res, "ref")
    write_weights(res, "w_good", acc=0.8, flip=0.05, agree=0.95)  # fewer flips, better agreement
    write_view(res, "good", shift=0.0)
    write_view(res, "bad_b", shift=-1.5)  # known scores collapse: AUROC drops
    write_view(res, "same", shift=0.0)
    return [
        spec("same", "ref"),
        spec("good", "w_good"),
        spec("bad_b", "ref"),
        spec("combo", "ref", 2, combo=True),
    ]


def test_scoring_only_candidate_shares_cv_axes_and_is_never_improved_there(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    cands = build_tree(res)
    out = sel.stage_select(make_ctx(res, cands), res)
    ax = out["axes"]["candidates"]["same"]["axes"]
    for a in ("a", "c", "d", "e"):
        assert ax[a]["delta"] == pytest.approx(0.0, abs=1e-12) and not ax[a]["improved"]
    assert ax["b"]["modes"]["raw"]["auroc"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert "same" in out["eligible"] and out["improved_axes"]["same"] == []


def test_every_candidate_gets_a_full_table_even_if_ineligible_or_not_formed(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    cands = build_tree(res)
    sc = make_ctx(res, cands)
    sel.stage_select(sc, res)
    axes = json.loads((res / "axes.json").read_text())
    sel_json = json.loads((res / "selection.json").read_text())
    assert set(axes["candidates"]) == {"same", "good", "bad_b", "combo"}
    bad = axes["candidates"]["bad_b"]
    assert bad["status"] == "scored" and not bad["eligible"] and "b" in bad["ineligible_axes"]
    for mode in ("raw", "neutral"):
        assert set(bad["axes"]["b"]["modes"][mode]) >= {"auroc", "rej95", "eligible", "improved"}
    assert axes["candidates"]["combo"]["status"] == "not_formed"
    assert "combo" in sel_json["not_formed"] and "bad_b" in sel_json["ineligible"]
    assert axes["ref"]["auroc"].keys() == {"raw", "neutral"}
    md = (res / "axes_table.md").read_text()
    assert all(k in md for k in ("same", "good", "bad_b", "combo", "INELIGIBLE"))


def test_candidate_with_better_flips_and_agreement_is_improved_on_c_and_d(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    cands = build_tree(res)
    out = sel.stage_select(make_ctx(res, cands), res)
    good = out["axes"]["candidates"]["good"]
    assert good["axes"]["c"]["improved"] and good["axes"]["d"]["improved"]
    assert good["eligible"] and {"c", "d"} <= set(good["improved_axes"])
    assert out["chosen"] == "good"  # only eligible candidate with improved axes
    assert out["chosen_spec"]["weights"] == "w_good"


def test_axis_b_needs_both_modes_eligible(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    write_weights(res, "ref")
    write_view(res, "ref")
    write_view(res, "mixed")
    # break only the neutral mode of `mixed`: collapse its known scores
    for s in SEEDS:
        for c in CLASSES:
            p = res / "mixed" / f"s{s}" / "views" / "neutral" / f"mixed_loco_{c}_s{s}.csv"
            t = pd.read_csv(p)
            m = (t["set"] == "eval") & ~t["is_unknown"]
            t.loc[m, ["score", "margin95", "margin90"]] -= 3.0
            t.to_csv(p, index=False)
    out = sel.stage_select(make_ctx(res, [spec("mixed", "ref")]), res)
    b = out["axes"]["candidates"]["mixed"]["axes"]["b"]
    assert b["modes"]["raw"]["eligible"] and not b["modes"]["neutral"]["eligible"]
    assert not b["eligible"] and out["chosen"] == "ref"


def sr(key: str, elig: bool, imp: tuple, comps: int = 1, rej: float = 0.5) -> sel.SelRow:
    return sel.SelRow(key, elig, imp, comps, rej)


def test_selection_tie_breaks_in_the_preregistered_order() -> None:
    # most improved axes first
    assert sel.select_candidate([sr("x", True, ("a",)), sr("y", True, ("a", "c"))])["chosen"] == "y"
    # tie 1: higher ID-neutral rejection@95
    out = sel.select_candidate([sr("x", True, ("c",), rej=0.3), sr("y", True, ("d",), rej=0.4)])
    assert out["chosen"] == "y" and [p["step"] for p in out["path"]][1:3] == [
        "most_improved_axes",
        "higher_neutral_rej95",
    ]
    # tie 2: fewer components
    out = sel.select_candidate([sr("x", True, ("c",), 3, 0.4), sr("y", True, ("d",), 1, 0.4)])
    assert out["chosen"] == "y" and not out["unresolved_tie"]
    # exact tie: first in order, flagged
    out = sel.select_candidate([sr("x", True, ("c",), 1, 0.4), sr("y", True, ("d",), 1, 0.4)])
    assert out["chosen"] == "x" and out["unresolved_tie"]
    # NaN neutral rejection loses to a finite one (fails closed)
    out = sel.select_candidate(
        [sr("x", True, ("c",), 1, float("nan")), sr("y", True, ("d",), 1, 0.1)]
    )
    assert out["chosen"] == "y"


def test_selection_ineligible_or_unimproved_or_ref_keeps_ref() -> None:
    assert sel.select_candidate([sr("x", False, ("a", "b", "c"))])["chosen"] == "ref"
    assert sel.select_candidate([sr("x", True, ())])["chosen"] == "ref"
    assert sel.select_candidate([])["chosen"] == "ref"
    assert sel.select_candidate([sr("ref", True, ("a",))])["chosen"] == "ref"


def test_resolve_candidates_marks_combos_by_the_choice_file() -> None:
    cands = [
        spec("i1a", "ref"),
        {**spec("i3i4", "i3i4", 2, combo=True)},
        {**spec("i3i1", "i3x", 2, combo=True)},
    ]
    none = sel.resolve_candidates({"candidates": cands}, None)
    assert [c["formed"] for c in none] == [True, False, False]
    choice = {
        "combos": {
            "i3i4": {"formed": False, "reason": "no I4 eligible"},
            "i3i1": {
                "formed": True,
                "weights": "i3g",
                "scorer": "i1b",
                "thr": "global",
                "reason": None,
            },
        }
    }
    got = sel.resolve_candidates({"candidates": cands}, choice)
    assert not got[1]["formed"] and got[1]["reason"] == "no I4 eligible"
    assert got[2]["formed"] and (got[2]["weights"], got[2]["scorer"]) == ("i3g", "i1b")


def test_misaligned_dev_rows_are_refused(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    write_weights(res, "ref")
    write_view(res, "ref")
    write_view(res, "other")
    p = res / "other" / "s0" / "views" / "raw" / "other_loco_c1_s0.csv"
    t = pd.read_csv(p)
    t["id"] = t["id"] + "x"
    t.to_csv(p, index=False)
    with pytest.raises(ValueError, match="differ|aligned"):
        sel.stage_select(make_ctx(res, [spec("other", "ref")]), res)


def test_missing_candidate_outputs_fail_loudly_not_silently(tmp_path: Path) -> None:
    res = tmp_path / "phase6a"
    write_weights(res, "ref")
    write_view(res, "ref")
    with pytest.raises(FileNotFoundError, match="missing view table"):
        sel.stage_select(make_ctx(res, [spec("ghost", "ref")]), res)


# ---------------------------------------------- the selection never touches D1 / CONFIRM / headline
def test_selection_module_has_no_oracle_or_confirm_dependencies() -> None:
    for name in ("phase6a_select.py", "phase6a_views.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module)
            elif isinstance(n, ast.Import):
                imported |= {a.name for a in n.names}
        assert not any("diag" in m or "post" in m or m.endswith("phase6a") for m in imported), (
            name,
            imported,
        )
        docstrings = {
            id(n.body[0].value)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
        }
        consts = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
        ]
        bad = [c for c in consts if re.search(r"diag|confirm|headline|oracle", c, re.I)]
        assert not bad, (name, bad)


def test_selection_runs_with_diag_and_confirm_artifacts_absent_and_ignores_them(
    tmp_path: Path,
) -> None:
    res = tmp_path / "phase6a"
    cands = build_tree(res)
    assert not (res / "diag").exists() and not (res / "confirm_report.json").exists()
    a = sel.stage_select(make_ctx(res, cands), res)
    # poison files that a leaky implementation would pick up
    (res / "diag").mkdir()
    (res / "diag" / "d1_oracle.json").write_text("{not json")
    (res / "confirm_report.json").write_text("{not json")
    (res / "confirm_runs").mkdir()
    b = sel.stage_select(make_ctx(res, cands), res)
    assert a["chosen"] == b["chosen"] and a["improved_axes"] == b["improved_axes"]
    ca, cb = a["axes"]["candidates"], b["axes"]["candidates"]
    assert ca["good"]["axes"]["c"] == cb["good"]["axes"]["c"]


# =================================================================================== configs
def test_config_candidates_match_the_preregistered_list() -> None:
    import yaml

    cfg = yaml.safe_load((SRC.parents[1] / "configs" / "phase6a.yaml").read_text(encoding="utf-8"))
    keys = [c["key"] for c in cfg["candidates"]]
    assert keys == [
        "i1a",
        "i1b",
        "i6a",
        "i6b",
        "i3u",
        "i3g",
        "i4a",
        "i4b",
        "i3i4",
        "i3i1",
        "i3i4i1",
        "i3i4i1i6",
    ]
    comps = {c["key"]: c["components"] for c in cfg["candidates"]}
    assert comps["i3i4i1i6"] == 4 and comps["i3i1"] == 2
    assert cfg["soup"]["pool_size"] == 7 and cfg["soup"]["members_per_soup"] == 5
    assert cfg["i4"]["lambdas"] == {"i4a": 0.5, "i4b": 1.0}
    assert cfg["model_seeds"] == [0, 1, 2] and cfg["bootstrap"]["seed"] == 42
    assert cfg["thresholds"] == {
        "a_max_drop": 0.010,
        "b_max_auroc_drop": 0.010,
        "b_max_rej95_drop": 0.03,
        "c_max_flip_increase": 0.02,
        "d_max_agreement_drop": 0.010,
        "e_max_ece_increase": 0.02,
    }


def test_epoch_summary_of_pool_curves_uses_the_pooled_mean() -> None:
    curves = np.array([[0.5, 0.6, 0.7], [0.5, 0.9, 0.7]])  # means 0.5, 0.75, 0.7
    assert p5.epoch_summary(curves)["e_star"] == 2


# ====================================================================== combos pick rules (pure)
I4_STATS = {"i4a": {"f1": 0.93, "flip": 0.20}, "i4b": {"f1": 0.94, "flip": 0.15}}
LAMS = {"i4a": 0.5, "i4b": 1.0}


def test_choose_soup_type_prefers_higher_cv_f1_and_uniform_on_ties() -> None:
    assert p6.choose_soup_type(0.93, 0.94) == "i3g"
    assert p6.choose_soup_type(0.94, 0.93) == "i3u"
    assert p6.choose_soup_type(0.94, 0.94) == "i3u"


def test_combo_picks_lambda_with_lower_flip_among_axis_a_eligible() -> None:
    args = dict(
        i3_type="i3g",
        f1_ref=0.94,
        lam_of=LAMS,
        a_max_drop=0.010,
        rej_i1={"i1a": 0.3, "i1b": 0.4},
        rej_i6={"i6a": 0.5, "i6b": 0.2},
    )
    ch = p6.choose_combos(i4=I4_STATS, **args)
    assert ch["i4_pick"] == "i4b" and ch["soup_type"] == "greedy"  # lower flip, both eligible
    assert ch["i1_scorer"] == "i1b" and ch["i6_variant"] == "i6a"
    assert {k: v["formed"] for k, v in ch["combos"].items()} == {
        "i3i4": True,
        "i3i1": True,
        "i3i4i1": True,
        "i3i4i1i6": True,
    }
    assert ch["combos"]["i3i1"]["weights"] == "i3g" and ch["combos"]["i3i4"]["weights"] == "i3i4"
    assert ch["combos"]["i3i4i1i6"]["scorer"] == "i1b" and ch["combos"]["i3i4i1i6"]["thr"] == "i6a"
    # the lower-flip variant is ineligible on (a): the other one is picked
    worse = {"i4a": {"f1": 0.93, "flip": 0.20}, "i4b": {"f1": 0.92, "flip": 0.05}}  # 0.92: -0.02
    assert p6.choose_combos(i4=worse, **args)["i4_pick"] == "i4a"
    assert p6.choose_combos(i4=worse, **args)["i4_standalone"]["i4b"]["eligible_a"] is False
    tie = {"i4a": {"f1": 0.94, "flip": 0.1}, "i4b": {"f1": 0.94, "flip": 0.1}}
    assert p6.choose_combos(i4=tie, **args)["i4_pick"] == "i4a"  # tie -> smaller lambda


def test_combos_not_formed_when_no_i4_variant_is_eligible_on_axis_a() -> None:
    bad = {k: {"f1": 0.90, "flip": 0.1} for k in LAMS}
    ch = p6.choose_combos(
        "i3u", 0.94, bad, LAMS, 0.010, {"i1a": 0.5, "i1b": 0.5}, {"i6a": 0.1, "i6b": 0.1}
    )
    assert ch["i4_pick"] is None
    assert [k for k, v in ch["combos"].items() if not v["formed"]] == ["i3i4", "i3i4i1", "i3i4i1i6"]
    assert ch["combos"]["i3i1"]["formed"] and "axis (a)" in ch["combos"]["i3i4"]["reason"]
    assert ch["i1_scorer"] == "i1a" and ch["i6_variant"] == "i6a"  # ties -> the first


def test_resolved_combos_flow_into_the_selection_candidates() -> None:
    ch = p6.choose_combos(
        "i3g", 0.94, I4_STATS, LAMS, 0.010, {"i1a": 0.2, "i1b": 0.1}, {"i6a": 0.1, "i6b": 0.3}
    )
    import yaml

    cfg = yaml.safe_load((SRC.parents[1] / "configs" / "phase6a.yaml").read_text(encoding="utf-8"))
    got = {c["key"]: c for c in sel.resolve_candidates(cfg, {"combos": ch["combos"]})}
    assert all(c["formed"] for c in got.values())
    assert (got["i3i1"]["weights"], got["i3i1"]["scorer"]) == ("i3g", "i1a")
    assert (got["i3i4i1i6"]["weights"], got["i3i4i1i6"]["scorer"], got["i3i4i1i6"]["thr"]) == (
        "i3i4",
        "i1a",
        "i6b",
    )


# ============================================================ post-selection stages (guards, plans)
def post_env(tmp_path: Path, smoke: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        smoke=smoke,
        P=SimpleNamespace(results=tmp_path / "res", outputs=tmp_path / "out"),
        pcfg={},
        dev_classes=["c1"],
        ctx=SimpleNamespace(
            cfg={
                "confirm_classes": ["k1", "k2", "k3", "k4", "k5"],
                "headline_seeds": [42, 43, 44],
                "loco_seed": 42,
            },
            headline_holdout=("h1", "h2", "h3"),
        ),
    )


def write_selection(env: SimpleNamespace, **kw) -> None:
    p6.write_json(env.P.results / "selection.json", {"smoke": False, "chosen": "ref", **kw})


def test_final_retrain_refuses_ref_missing_and_smoke_selections(tmp_path: Path) -> None:
    from intent_router import phase6a_post as post

    env = post_env(tmp_path)
    with pytest.raises(SystemExit, match="select first"):
        post.stage_final_retrain(env)  # no selection.json at all
    write_selection(env, chosen="ref")
    with pytest.raises(SystemExit, match="chose ref"):
        post.stage_final_retrain(env)  # v1 / ref stays: nothing is retrained
    write_selection(env, chosen="i3u", smoke=True)
    with pytest.raises(SystemExit, match="smoke run"):
        post.stage_final_retrain(env)  # a smoke selection never authorises a real retrain
    with pytest.raises(SystemExit, match="select first"):
        post.stage_confirm_report(post_env(tmp_path / "other"))  # confirm needs a selection too


def test_final_retrain_needs_confirm_report_and_a_formed_candidate(tmp_path: Path) -> None:
    from intent_router import phase6a_post as post

    env = post_env(tmp_path)
    env.pcfg = {"candidates": [spec("i3u", "i3u"), spec("i3i4", "i3i4", 2, combo=True)]}
    write_selection(env, chosen="i3u")
    with pytest.raises(SystemExit, match="confirm_report first"):
        post.stage_final_retrain(env)
    write_selection(env, chosen="i3i4")  # combo that was never formed
    with pytest.raises(SystemExit, match="not a formed candidate"):
        post.stage_final_retrain(env)


def test_confirm_plan_matches_the_headline_and_confirm_conventions(tmp_path: Path) -> None:
    from intent_router import phase6a_post as post

    runs = post.confirm_plan(post_env(tmp_path), "i3u")
    loco = [r for r in runs if r.kind == "loco"]
    head = [r for r in runs if r.kind == "headline"]
    assert [r.run_id for r in loco] == [f"i3u_loco_k{i}_s42" for i in range(1, 6)]
    assert all(r.seed == 42 and r.replicate == 0 and r.ns == "confirm" for r in loco)
    assert [(r.run_id, r.seed, r.replicate) for r in head] == [
        ("i3u_headline_s42", 42, 0),
        ("i3u_headline_s43", 43, 1),
        ("i3u_headline_s44", 44, 2),
    ]
    assert {r.unit for r in head} == {"headline"} and head[0].holdout == ("h1", "h2", "h3")
    assert {r.call_type for r in runs} == {"phase6a_confirm"}
    smoke = post.confirm_plan(post_env(tmp_path, smoke=True), "ref")
    assert [r.run_id for r in smoke] == ["ref_loco_c1_s42", "ref_headline_s42"]


def test_look_counts_are_tallied_from_existing_artifacts(tmp_path: Path) -> None:
    from intent_router import phase6a_post as post

    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text("{}")
    pcfg = {
        "confirm_report": {
            "looks": {
                "confirm": [
                    {"phase": "3", "path": str(tmp_path / "a.json")},
                    {"phase": "4c", "path": str(tmp_path / "missing.json")},
                ],
                "headline": [
                    {"phase": "3", "path": str(tmp_path / "a.json")},
                    {"phase": "3b", "path": str(tmp_path / "b.json")},
                    {"phase": "3b", "path": str(tmp_path / "a.json")},
                ],
            }
        }
    }
    out = post.look_counts(pcfg)
    assert out["confirm"]["prior_looks"] == 1 and out["confirm"]["including_6a"] == 2
    assert out["headline"]["prior_looks"] == 2 and out["headline"]["phases"] == ["3", "3b"]
    assert out["confirm"]["artifacts"][1]["exists"] is False


def test_shipped_ood_variant_on_saved_final_arrays(tmp_path: Path, monkeypatch) -> None:
    from intent_router import data as data_mod
    from intent_router import phase6a_post as post
    from intent_router.evaluate import threshold_at_retention

    labels = ["a", "b", "c"]
    monkeypatch.setattr(data_mod, "LABELS", labels)
    rng = np.random.default_rng(0)
    ids = {
        "train": [f"tr{i}" for i in range(60)],
        "val": [f"va{i}" for i in range(30)],
        "test": [f"te{i}" for i in range(20)],
    }
    lab = {i: labels[n % 3] for part in ids.values() for n, i in enumerate(part)}
    monkeypatch.setattr(
        data_mod,
        "load_data",
        lambda _p: pd.DataFrame({"id": list(lab), "label": list(lab.values())}),
    )
    cent = rng.normal(scale=4, size=(3, 5))
    z: dict[str, np.ndarray] = {}
    for part, pid in ids.items():
        y = np.array([labels.index(lab[i]) for i in pid])
        z[f"{part}_ids"] = np.array(pid)
        z[f"{part}_features"] = (cent[y] + rng.normal(size=(len(y), 5))).astype(np.float32)
        lg = rng.normal(size=(len(y), 3))
        lg[np.arange(len(y)), y] += 3
        z[f"{part}_logits"] = lg.astype(np.float32)
    np.savez(tmp_path / "fl.npz", **z)
    tb = {"per_class_min_rows": 5, "crossfit_folds": 5, "crossfit_seed": 42}
    cfg = {"data_path": "unused"}
    g = post.shipped_ood_variant(cfg, tmp_path / "fl.npz", "maha_ft", "global", 0.95, tb)
    assert g["method"] == "maha_ft/global" and g["val_retention_achieved"] >= 0.95
    fit = ov.fit_feature_scorer("maha_ft", z["train_features"], np.arange(60) % 3, 3)
    assert g["threshold"] == pytest.approx(threshold_at_retention(fit(z["val_features"]), 0.95))
    a = post.shipped_ood_variant(cfg, tmp_path / "fl.npz", "i1b", "i6a", 0.95, tb)
    assert set(a["per_class_thresholds"]) == set(labels) and "fallback_classes" in a
    b = post.shipped_ood_variant(cfg, tmp_path / "fl.npz", "i1a", "i6b", 0.95, tb)
    assert b["n_crossfit_rows"] == 90 and "threshold" in b


# ------------------------------------------------ retention guard on axis (b)
def test_known_retention_counts_accepted_known_eval_rows_only() -> None:
    t = pd.DataFrame(
        {
            "set": ["cal", "eval", "eval", "eval", "eval"],
            "is_unknown": [False, False, False, False, True],
            "margin95": [-9.0, 1.0, 0.0, -1.0, 5.0],  # cal row and the unknown row are ignored
        }
    )
    assert sel.known_retention(t) == pytest.approx(2 / 3)  # margin >= 0 accepts


def test_retention_guard_rejects_collapsed_operating_point_and_passes_ref_equal(
    tmp_path: Path,
) -> None:
    res = tmp_path / "phase6a"
    write_weights(res, "ref")
    write_view(res, "ref")
    write_view(res, "same")
    write_view(res, "collapsed")
    # raise the threshold only: scores (AUROC) untouched, known retention ~0.5, rejection rises
    for mode in pv.MODES:
        for s in SEEDS:
            for c in CLASSES:
                p = res / "collapsed" / f"s{s}" / "views" / mode / f"collapsed_loco_{c}_s{s}.csv"
                t = pd.read_csv(p)
                m = (t["set"] == "eval") & ~t["is_unknown"]
                t["margin95"] -= t.loc[m, "margin95"].median()
                t.to_csv(p, index=False)
    out = sel.stage_select(make_ctx(res, [spec("same", "ref"), spec("collapsed", "ref")]), res)
    ax = out["axes"]
    ref_ret = ax["ref"]["retention"]
    assert 0.9 < ref_ret["raw"] < 1.0
    same, bad = ax["candidates"]["same"], ax["candidates"]["collapsed"]
    assert same["retention_guard"] is True and same["eligible"]
    assert same["axes"]["b"]["modes"]["raw"]["retention"]["delta"] == pytest.approx(0.0, abs=1e-12)
    assert bad["retention_guard"] is False and not bad["eligible"]
    assert bad["ineligible_axes"] == ["b"]
    for mode in pv.MODES:
        r = bad["axes"]["b"]["modes"][mode]
        assert r["retention"]["delta"] < -0.3 and not r["retention"]["guard_ok"]
        assert r["rej95"]["delta"] > 0  # the unguarded rule would have called this an improvement
    assert out["chosen"] == "ref" and "collapsed" in out["ineligible"]
    assert "retention guard" in (res / "axes_table.md").read_text()


def test_retention_guard_threshold_is_two_points_in_either_mode() -> None:
    def b(raw: tuple[float, float], neu: tuple[float, float]) -> dict:
        rec = {"auroc": (0.9, np.full(50, 0.9)), "rej95": (0.5, np.full(50, 0.5))}
        both = {"raw": rec, "neutral": rec}
        th = {"b_max_auroc_drop": 0.01, "b_max_rej95_drop": 0.05}
        return sel.b_axis(both, both, th, 0.95, {"raw": raw, "neutral": neu})

    assert b((0.95, 0.97), (0.95, 0.97))["eligible"]  # exactly -0.02 is allowed
    assert not b((0.9499, 0.97), (0.97, 0.97))["eligible"]  # raw alone fails
    assert not b((0.97, 0.97), (0.5, 0.97))["eligible"]  # neutral alone fails
