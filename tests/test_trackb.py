from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from intent_router import trackb  # noqa: E402
from intent_router.models import prepare_text  # noqa: E402

E5 = "intfloat/multilingual-e5-base"


def _word_id(w: str) -> int:
    return 2 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 90


class _StubTokenizer:
    """Whitespace tokenizer with a tiny vocab; records every text it is asked to encode."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _ids(self, texts: list[str], max_length: int) -> list[list[int]]:
        self.seen += texts
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


def _tiny_encoder() -> Any:
    cfg = transformers.XLMRobertaConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=70, pad_token_id=1,
    )  # fmt: skip
    torch.manual_seed(0)
    return transformers.XLMRobertaModel(cfg).eval()


# ----------------------------------------------------------- frozen-embedding path (CPU)
def test_frozen_embeddings_prefix_pooling_and_cache(tmp_path: Path) -> None:
    enc, tok = _tiny_encoder(), _StubTokenizer()
    texts = ["alpha beta", "gamma", "delta epsilon zeta eta"]
    emb = trackb.embed_frozen_rows(texts, enc, tok, E5, 16, torch.device("cpu"))
    assert emb.shape == (3, 32) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)  # L2-normalised
    assert all(t.startswith("query: ") for t in tok.seen)  # the e5 prefix is applied

    # masked mean pooling reproduced by hand for the unpadded single row
    single = prepare_text("gamma", E5)
    ids = torch.tensor([[_word_id(w) for w in single.split()]])
    with torch.no_grad():
        h = enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0].mean(0)
    manual = (h / h.norm()).numpy()
    assert np.allclose(emb[1], manual, atol=1e-5)

    # batch composition (padding) must not change a row's embedding
    solo = trackb.embed_frozen_rows(texts[2:], enc, _StubTokenizer(), E5, 16, torch.device("cpu"))
    assert np.allclose(emb[2], solo[0], atol=1e-5)

    cache = tmp_path / "frozen.npz"
    ids = ["id3", "id1", "id2"]
    trackb.save_frozen(cache, ids, emb, E5, 16)
    got = trackb.frozen_lookup(cache, ["id1", "id2", "id3"], E5, 16)
    assert got is not None
    assert np.allclose(got, emb[[1, 2, 0]], atol=1e-6)  # re-ordered by id
    assert trackb.frozen_lookup(cache, ["id1", "missing"], E5, 16) is None
    assert trackb.frozen_lookup(cache, ["id1"], E5, 32) is None  # max_len mismatch => stale
    assert trackb.frozen_lookup(tmp_path / "none.npz", ["id1"], E5, 16) is None


# ------------------------------------------------------------------- run plan + config
def _cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "trackb.yaml").read_text())
    fcfg = yaml.safe_load((root / "configs" / "final.yaml").read_text())
    return cfg, fcfg


LABELS = [
    "ai_agent_performance", "appointment_manager", "chitchat", "customer_support",
    "document_processing", "knowledge_base", "orders", "other",
    "shipment_information.analytics", "shipment_information.disruptions",
    "shipment_information.realtime_query", "yard_management",
]  # fmt: skip


def test_plan_runs_counts_and_ids() -> None:
    cfg, _ = _cfgs()
    head = trackb.plan_runs(cfg, LABELS, "headline")
    loco = trackb.plan_runs(cfg, LABELS, "loco")
    ls = trackb.plan_runs(cfg, LABELS, "ls")
    assert [r.seed for r in head] == [42, 43, 44]
    assert all(r.holdout == ("document_processing", "yard_management") for r in head + ls)
    assert len(loco) == 10 and {r.holdout[0] for r in loco} == set(LABELS) - {"other", "chitchat"}
    assert all(r.seed == 42 and r.label_smoothing == 0.0 for r in loco)
    assert [r.label_smoothing for r in ls] == [0.1] * 3
    ids = [r.run_id for r in head + loco + ls]
    assert len(ids) == len(set(ids)) == 16
    assert {r.group for r in head} == {"trackB-headline"} and {r.group for r in ls} == {"trackB-ls"}


def test_trackb_models_use_the_final_config() -> None:
    cfg, fcfg = _cfgs()
    spec = trackb.plan_runs(cfg, LABELS, "headline")[1]
    t = trackb.train_dict(fcfg, spec, smoke=False, wandb_project="p")
    ft = fcfg["train"]
    assert (t.model_name, t.lr, t.epochs, t.stop_epoch) == (ft["model_name"], ft["lr"], 20, 9)
    assert (t.batch_size, t.max_len, t.attn_implementation) == (16, 64, "eager")
    assert (t.warmup_ratio, t.weight_decay, t.precision) == (0.1, 0.01, "fp16")
    assert t.model_seed == 43 and t.label_smoothing == 0.0 and not t.class_weighted
    ls = trackb.train_dict(fcfg, trackb.plan_runs(cfg, LABELS, "ls")[0], False, "p")
    assert ls.label_smoothing == 0.1 and ls.model_seed == 42 and ls.lr == ft["lr"]
    assert trackb.train_dict(fcfg, spec, smoke=True, wandb_project="p").stop_epoch == 1


# ----------------------------------------- train_model with a reduced label space (CPU)
def test_train_model_label_space_matches_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    from intent_router import train as train_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    sizes: list[int] = []

    def fake_build(name: str, labels: list[str], attn: str | None) -> tuple[Any, Any]:
        sizes.append(len(labels))
        cfg = transformers.XLMRobertaConfig(
            vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, max_position_embeddings=70, num_labels=len(labels),
            pad_token_id=1,
        )  # fmt: skip
        return _StubTokenizer(), transformers.XLMRobertaForSequenceClassification(cfg)

    monkeypatch.setattr(train_mod, "build_model", fake_build)
    labels = ["b", "chitchat", "d", "other"]
    rows = [
        {"id": f"r{i}", "text": f"w{i} x", "label": labels[i % 4], "split": s}
        for i, s in enumerate(["train"] * 16 + ["val"] * 8)
    ]
    df = pd.DataFrame(rows)
    tcfg = train_mod.TrainConfig(
        model_name="tiny", lr=1e-3, epochs=2, batch_size=8, max_len=8, stop_epoch=1,
        wandb_group="x",
    )  # fmt: skip
    res, model, _ = train_mod.train_model(
        tcfg, df[df["split"] == "train"], df[df["split"] == "val"], {"run_id": "t"}, labels=labels
    )
    assert sizes == [4] and model.config.num_labels == 4
    assert res.probs.shape == (1, 8, 4) and set(res.gold.tolist()) <= {0, 1, 2, 3}


# ------------------------------------------------------ aggregation from score tables only
def _fake_run(seed: int, holdout: list[str], gap: float, best: str | None = None) -> dict[str, Any]:
    """A run dict as load_run() returns it, built from a synthetic score table."""
    from intent_router import ood

    labels = ["a", "other", "b"]
    rng = np.random.default_rng(seed)
    methods = ood.method_names((1, 5))
    parts = []
    for st, n, unk in (("cal", 60, False), ("eval", 60, False), ("eval", 40, True)):
        gold = rng.choice(["a", "b"], n) if not unk else np.array([holdout[0]] * n)
        t = pd.DataFrame({"set": st, "is_unknown": unk, "gold": gold,
                          "pred": rng.choice(labels, n)})  # fmt: skip
        for m in methods:
            t[m] = rng.normal(0.0 if unk else gap + (3.0 if m == best else 0.0), 1.0, n)
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    meta = {
        "smoke": False, "holdout": holdout, "seed": seed, "labels": labels,
        "audit": {"temperature": 1.0}, "training": {"wall_clock_s": 1.0, "gpu_exclusive": True,
                                                     "wandb_url": None},
    }  # fmt: skip
    return {"meta": meta, "table": table, "methods": methods,
            "metrics": ood.run_metrics(table, methods, labels)}  # fmt: skip


def test_headline_ls_and_loco_aggregation() -> None:
    hold = ["z"]
    base = {f"h{s}": _fake_run(s, hold, 1.0) for s in (42, 43, 44)}
    ls = {f"l{s}": _fake_run(s, hold, 2.0) for s in (42, 43, 44)}
    h = trackb.headline_report(base, [42, 43, 44])
    a = [r["metrics"]["methods"]["msp"]["auroc"] for r in base.values()]
    assert h["methods"]["msp"]["auroc"]["mean"] == pytest.approx(np.mean(a))
    assert h["methods"]["msp"]["auroc"]["std"] == pytest.approx(np.std(a, ddof=1))
    d = trackb.ls_report(base, ls)["methods"]["msp"]["auroc"]
    assert d["delta"]["mean"] == pytest.approx(d["ls0p1"]["mean"] - d["ls0"]["mean"])
    assert d["delta"]["mean"] > 0  # the wider known/unknown gap in the "ls" runs shows up
    loco = {f"c{i}": _fake_run(42, [f"c{i}"], 0.5 + 0.1 * i) for i in range(4)}
    rep = trackb.loco_report(loco, 4)
    sel = rep["selection"]
    assert sel["shipped_method"] == max(
        sel["mean_auroc_by_method"], key=sel["mean_auroc_by_method"].get
    )
    assert "design choice" in sel["note"] and sorted(rep["per_class"]) == ["c0", "c1", "c2", "c3"]
    assert (
        trackb.loco_report(dict(list(loco.items())[:2]), 4).get("selection") is None
    )  # incomplete


@pytest.mark.parametrize("best", ["knn5_ft", "knn1_frozen"])
def test_shipped_stage_uses_saved_arrays_only(
    best: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from intent_router import data as data_mod

    cfg, fcfg = _cfgs()
    monkeypatch.setattr(data_mod, "LABELS", list(LABELS))
    P = trackb.Paths(tmp_path / "res", tmp_path / "fig", tmp_path / "out",
                     tmp_path / "frozen.npz", tmp_path / "log.jsonl", False)  # fmt: skip
    (P.results / "scores").mkdir(parents=True)
    (P.results / "runs").mkdir(parents=True)
    specs = trackb.plan_runs(cfg, LABELS, "loco")
    for sp in specs[:-1]:  # incomplete LOCO => graceful skip, nothing written
        run = _fake_run(sp.seed, list(sp.holdout), 1.0, best)
        run["table"].to_csv(P.results / "scores" / f"{sp.run_id}.csv", index=False)
        (P.results / "runs" / f"{sp.run_id}.json").write_text(json.dumps(run["meta"]))
    rng = np.random.default_rng(0)
    rows, parts = [], {}
    for part, per_class in (("train", 10), ("val", 4), ("test", 4)):
        ids = [f"{part}{i}_{c}" for c in range(12) for i in range(per_class)]
        y = np.repeat(np.arange(12), per_class)
        parts[part] = (ids, y)
        rows += [{"id": i, "label": LABELS[c], "split": part} for i, c in zip(ids, y, strict=True)]
    df_all = pd.DataFrame(rows)
    arrays: dict[str, np.ndarray] = {}
    for part, (ids, y) in parts.items():
        arrays[f"{part}_ids"] = np.array(ids)
        arrays[f"{part}_logits"] = (4 * np.eye(12)[y] + rng.normal(size=(len(y), 12))).astype("f4")
        arrays[f"{part}_features"] = (3 * np.eye(12, 16)[y] + rng.normal(size=(len(y), 16))).astype(
            "f4"
        )
    npz = tmp_path / "fl.npz"
    np.savez(npz, **arrays)
    cfg["final_artifacts"] = {
        "features_logits": str(npz),
        "test_predictions": str(tmp_path / "x.csv"),
    }
    cfg["ood"]["knn_k"] = [1, 5]
    all_ids = sum((parts[p][0] for p in parts), [])
    emb = rng.normal(size=(len(all_ids), 16)).astype("f4")
    tr = fcfg["train"]
    trackb.save_frozen(P.frozen_cache, all_ids, emb, tr["model_name"], int(tr["max_len"]))

    trackb.stage_shipped(cfg, fcfg, P, df_all)
    assert not (P.results / "shipped.json").exists()  # LOCO incomplete: skipped, not guessed
    sp = specs[-1]
    run = _fake_run(sp.seed, list(sp.holdout), 1.0, best)
    run["table"].to_csv(P.results / "scores" / f"{sp.run_id}.csv", index=False)
    (P.results / "runs" / f"{sp.run_id}.json").write_text(json.dumps(run["meta"]))
    trackb.stage_shipped(cfg, fcfg, P, df_all)
    out = json.loads((P.results / "shipped.json").read_text())
    assert out["method"] == best and "design choice" in out["selection"]["note"]
    assert out["k"] == (5 if best == "knn5_ft" else 1)
    assert out["val_retention_achieved"] >= 0.95 and out["n_val"] == 48 and out["n_test"] == 48
    assert 0.0 <= out["test_coverage_at_threshold"] <= 1.0 and out["temperature"] > 0
    assert "none" in out["sources"]["test_inference"]
    cfg["final_artifacts"]["features_logits"] = str(tmp_path / "missing.npz")
    (P.results / "shipped.json").unlink()
    trackb.stage_shipped(cfg, fcfg, P, df_all)  # missing final artifacts: skip gracefully
    assert not (P.results / "shipped.json").exists()
