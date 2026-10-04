from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intent_router import llm_baseline as lb
from intent_router.llm_audit import LABEL_DESCRIPTIONS

LABELS = [f"L{i}" for i in range(12)]
REAL = list(LABEL_DESCRIPTIONS)
HOLDOUT = ["L10", "L11"]


def _train() -> pd.DataFrame:
    n = 120
    return pd.DataFrame(
        {
            "id": [f"r{i:03d}" for i in range(n)],
            "text": [f"text {i}" for i in range(n)],
            "label": [LABELS[i % 12] for i in range(n)],
            "split": "train",
        }
    )


@pytest.mark.parametrize(
    ("raw", "want"),
    [("L3", "L3"), ("  l3\n", "L3"), ("UNKNOWN", "unknown"), (" Unknown ", "unknown")],
)
def test_parse_valid(raw: str, want: str) -> None:
    assert lb.parse_label(raw, LABELS) == want


@pytest.mark.parametrize(
    "raw",
    ["", "L3 because", "The answer is L3", "L3.", '"L3"', "ambiguous: L1 | L2", "L99", "L3\nL4"],
)
def test_parse_junk(raw: str) -> None:
    assert lb.parse_label(raw, LABELS) is None


def test_parse_respects_offered_labels() -> None:
    assert lb.parse_label("L11", [x for x in LABELS if x not in HOLDOUT]) is None


def test_fewshot_distinct_train_deterministic() -> None:
    train = _train()
    a = lb.sample_fewshot(train, 42, 5)
    assert a == lb.sample_fewshot(train, 42, 5)
    assert len({s["label"] for s in a}) == 5
    assert set(s["id"] for s in a) <= set(train["id"])
    for s in a:
        assert train.loc[train["id"] == s["id"], "label"].iloc[0] == s["label"]
    assert a != lb.sample_fewshot(train, 7, 5)


def test_fewshot_excludes_heldout_and_rejects_non_train() -> None:
    train = _train()
    for seed in range(20):
        shots = lb.sample_fewshot(train, seed, 5, HOLDOUT)
        assert not {s["label"] for s in shots} & set(HOLDOUT)
        assert len({s["label"] for s in shots}) == 5
    bad = train.copy()
    bad.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train split"):
        lb.sample_fewshot(bad, 42, 5)


def test_prompt_structure() -> None:
    labels = list(LABEL_DESCRIPTIONS)[:3]
    shots = [{"id": "a", "text": "hello", "label": labels[0]}]
    msgs = lb.build_prompt("query text", labels, shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "- unknown: fits none of the listed intents" in msgs[0]["content"]
    assert LABEL_DESCRIPTIONS[labels[1]] in msgs[0]["content"]
    assert msgs[2]["content"] == labels[0]
    assert "query text" in msgs[-1]["content"]
    assert len(lb.build_prompt("q", labels, [])) == 2


class _FakeJudge:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.calls = replies, 0

    def chat(self, messages: list[dict[str, str]], keep_alive: object = None) -> str:
        self.calls += 1
        return self.replies.pop(0)


def test_classify_retry_and_failure() -> None:
    j = _FakeJudge(["nope", REAL[2]])
    r = lb.classify(j, "x", REAL, [])  # type: ignore[arg-type]
    assert (r["pred"], r["parse_ok"], j.calls) == (REAL[2], True, 2)
    j = _FakeJudge(["nope", "still nope"])
    r = lb.classify(j, "x", REAL, [])  # type: ignore[arg-type]
    assert (r["pred"], r["parse_ok"], j.calls) == (lb.PARSE_FAIL, False, 2)
    j = _FakeJudge(["unknown"])
    assert lb.classify(j, "x", REAL, [])["pred"] == "unknown"  # type: ignore[arg-type]
    assert j.calls == 1


def test_cost_arithmetic() -> None:
    # 0.36 s/msg at $0.35/hr: 1000 msgs = 360 s = 0.1 h -> $0.035
    assert lb.cost_per_1k(0.36, 0.35) == pytest.approx(0.035)
    assert lb.cost_per_1k(0.0, 0.35) == 0.0
    s = lb.latency_stats([0.1, 0.2, 0.3, 0.4])
    assert s["p50_s"] == pytest.approx(0.25)
    assert s["throughput_msg_per_s"] == pytest.approx(4.0)


def test_metrics_track_a_unknown_and_failure_are_wrong() -> None:
    y = np.arange(12)
    preds = [LABELS[i] for i in range(12)]
    ok = [True] * 12
    m = lb.metrics_track_a(y, preds, ok, LABELS, 200, 42)
    assert m["macro_f1"]["point"] == 1.0 and m["accuracy"]["point"] == 1.0
    preds[0], preds[1] = lb.UNKNOWN, lb.PARSE_FAIL
    ok[1] = False
    m = lb.metrics_track_a(y, preds, ok, LABELS, 200, 42)
    assert m["accuracy"]["point"] == pytest.approx(10 / 12)
    assert m["macro_f1"]["point"] == pytest.approx(10 / 12)  # F1 0 for two classes, mean over 12
    assert m["parse_failure_rate"] == pytest.approx(1 / 12)
    assert m["unknown_rate"] == pytest.approx(1 / 12)


def test_paired_delta_and_mcnemar() -> None:
    y = np.tile(np.arange(12), 3)
    good = y.copy()
    bad = y.copy()
    bad[:12] = 12  # first block all 'unknown'
    d = lb.paired_delta_macro_f1(y, good, bad, 12, 500, 42)
    assert d["delta"] > 0 and d["lo"] <= d["delta"] <= d["hi"]
    m = lb.metrics_track_a(
        y,
        [LABELS[i] if i < 12 else "x" for i in bad],
        [True] * 36,
        LABELS,
        200,
        42,
        final_pred=good,
    )
    assert m["mcnemar_exact_on_correctness_a_is_llm"]["b_only_correct"] == 12


def test_metrics_track_b() -> None:
    known = [x for x in LABELS if x not in HOLDOUT]
    gold = ["L0", "L1", "L2", "L3", "L10", "L10", "L11", "L11"]
    unk = [False] * 4 + [True] * 4
    preds = ["L0", lb.UNKNOWN, "L5", lb.PARSE_FAIL, lb.UNKNOWN, lb.PARSE_FAIL, "L0", lb.UNKNOWN]
    m = lb.metrics_track_b(gold, unk, preds, LABELS)
    assert m["retention_known"] == pytest.approx(3 / 4)  # parse failure on known = accepted
    assert m["strict_rejection_recall"] == pytest.approx(2 / 4)  # parse failure != rejection
    assert m["n_known_accepted"] == 3
    assert m["known_accepted_accuracy"] == pytest.approx(1 / 3)
    assert m["parse_failures_known"] == 1 and m["parse_failures_unknown"] == 1
    assert len(known) == 10


def test_loopback_guard_is_reused() -> None:
    from intent_router import llm_audit

    assert lb.validate_loopback_url is llm_audit.validate_loopback_url
    assert lb._http_json is llm_audit._http_json
    assert lb.validate_loopback_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    with pytest.raises(ValueError, match="non-loopback"):
        lb.validate_loopback_url("http://example.com:11434")
    with pytest.raises(ValueError, match="non-http"):
        lb.validate_loopback_url("https://127.0.0.1:11434")


def _ft_cfg(threads: int = 8) -> tuple[dict, dict, dict]:
    cfg = {"cost": {"gpu_usd_per_hr": 0.35, "cpu_usd_per_hr": 0.05, "llm_note": "n"}}
    llm = {"configs": {"m|zero": {"p50_s": 1.0, "p95_s": 2.0}}}
    lat = {"p50_s": 0.1, "threads": 1}
    ft = {
        "gpu_batch1": {"p50_s": 0.01},
        "gpu_batch32_throughput_msg_per_s": 100.0,
        "cpu_batch1_1thread": lat,
        "cpu_batch1_default_threads": {"p50_s": 0.1, "threads": threads},
    }
    return cfg, llm, ft


def test_build_costs_prices_threads_as_vcpus() -> None:
    cfg, llm, ft = _ft_cfg(8)
    c = lb.build_costs(cfg, llm, ft)
    one, eight = c["rows"]["ft|cpu_batch1_1thread"], c["rows"]["ft|cpu_batch1_default_threads"]
    assert (one["n_vcpu"], eight["n_vcpu"]) == (1, 8)
    assert eight["usd_per_hr"] == pytest.approx(8 * one["usd_per_hr"]) == pytest.approx(0.40)
    # same latency -> 8x the cost per 1k messages
    assert eight["usd_per_1k_messages"] == pytest.approx(8 * one["usd_per_1k_messages"])
    assert eight["usd_per_hr_basis"] == "8 vCPU x $0.05/hr"
    for row in c["rows"].values():
        assert row["usd_per_hr"] > 0 and row["usd_per_hr_basis"]
    assert c["rows"]["ft|gpu_batch1"]["usd_per_hr_basis"] == "1 T4-class GPU at $0.35/hr"
    assert c["rows"]["llm|m|zero"]["usd_per_hr_basis"] == "1 T4-class GPU at $0.35/hr"
    a = c["assumptions"]
    assert a["gpu_usd_per_hr"] == 0.35 and a["cpu_usd_per_vcpu_per_hr"] == 0.05
    assert "T4-class" in a["gpu_note"] and "vCPUs" in a["cpu_note"]


def _rescore_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    labs = LABELS
    ids = [f"t{i:02d}" for i in range(24)]
    gold = [labs[i % 12] for i in range(24)]
    final = pd.DataFrame({"id": ids, "gold_label": gold, "pred": [i % 12 for i in range(24)]})
    rows = []
    for mode, wrong in (("zero", 0), ("five", 6)):
        for i, (rid, g) in enumerate(zip(ids, gold, strict=True)):
            p = g if i >= wrong else lb.UNKNOWN
            rows.append(
                {"id": rid, "model": "m", "mode": mode, "gold": g, "pred": p, "parse_ok": True}
            )
    preds = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)  # order must not matter
    return preds, final


def test_rescore_metrics_matches_direct_computation_and_checks_gold() -> None:
    preds, final = _rescore_frames()
    labs = LABELS
    m = lb.rescore_metrics(preds, final, labs, 200, 42)
    assert set(m) == {"m|zero", "m|five"}
    assert m["m|zero"]["accuracy"]["point"] == 1.0
    assert m["m|five"]["accuracy"]["point"] == pytest.approx(18 / 24)
    assert m["m|five"]["unknown_rate"] == pytest.approx(6 / 24)
    assert m["m|zero"]["delta_macro_f1_llm_minus_final"]["delta"] == pytest.approx(0.0)
    bad = final.copy()
    bad.loc[0, "gold_label"] = "other"
    with pytest.raises(AssertionError, match="differ"):
        lb.rescore_metrics(preds, bad, labs, 200, 42)


def test_stage_rescore_track_a_updates_only_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from intent_router import data as data_mod

    monkeypatch.setattr(data_mod, "LABELS", list(LABELS))  # restored after the test
    monkeypatch.setattr(data_mod, "load_data", lambda path: pd.DataFrame())

    preds, final = _rescore_frames()
    res = tmp_path / "res"
    fin = tmp_path / "final"
    res.mkdir()
    fin.mkdir()
    preds.to_csv(res / "predictions_track_a.csv", index=False)
    final.to_csv(fin / "test_predictions.csv", index=False)
    (fin / "model_version.json").write_text(json.dumps({"model_fingerprint": "abc123"}))
    (res / "track_a.json").write_text(
        json.dumps({"label": "keep", "fewshot_ids": {"zero": []}, "metrics": {"old": 1}})
    )
    cfg = {
        "data_path": "unused",
        "results_dir": str(res),
        "test_predictions_path": str(fin / "test_predictions.csv"),
        "bootstrap": {"n_resamples": 200, "seed": 42},
    }
    lb.stage_rescore_track_a(cfg)
    out = json.loads((res / "track_a.json").read_text())
    assert out["label"] == "keep" and out["fewshot_ids"] == {"zero": []}
    assert set(out["metrics"]) == {"m|zero", "m|five"}
    assert out["rescored_against_final_fingerprint"] == "abc123"
    assert "rescored_git_sha" in out
    assert "rescore_track_a" in lb.STAGES
