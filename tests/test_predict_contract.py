"""Serving contract tests: hermetic (tiny random model + tokenizer built in tmp), no dataset.

Runs in `.venv-serve` (torch + transformers + fastapi + onnxruntime), not in the training venv.
The integration test at the bottom needs a packaged v1 serve dir + the dataset and is skipped
when they are absent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import (  # noqa: E402
    PreTrainedTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)

from intent_router import package, serve, serve_bench  # noqa: E402
from intent_router.predict import (  # noqa: E402
    MahalanobisBank,
    Router,
    mahalanobis_score,
    softmax,
)

LABELS = ["alpha", "beta", "gamma", "delta", "epsilon"]
WORDS = ["query:", "where", "is", "my", "order", "book", "a", "dock", "hello", "refund", "please"]
PREFIX = "query: "
MAX_LEN = 16
HIDDEN = 32
T = 1.7  # deliberately != 1 so a missing temperature shows up in confidence
ROOT = Path(__file__).resolve().parents[1]


def _tiny_model_dir(path: Path) -> Path:
    """Random 2-layer XLM-R classifier + WordLevel tokenizer saved like outputs/final_model."""
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({w: 4 + i for i, w in enumerate(WORDS)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
    )
    torch.manual_seed(0)
    cfg = XLMRobertaConfig(
        vocab_size=len(vocab),
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    model = XLMRobertaForSequenceClassification(cfg).eval()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    fast.save_pretrained(path)
    c = json.loads((path / "config.json").read_text())
    c["query_prefix"], c["max_length"] = PREFIX, MAX_LEN  # what final.save_final_model adds
    (path / "config.json").write_text(json.dumps(c))
    return path


def _bank() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    means = rng.normal(size=(len(LABELS), HIDDEN))
    a = rng.normal(size=(HIDDEN, HIDDEN))
    return means, a @ a.T / HIDDEN + np.eye(HIDDEN)  # symmetric positive definite precision


@pytest.fixture(scope="module")
def serve_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Packaged fake serve dir with a threshold at the median OOD score of the sample texts."""
    base = tmp_path_factory.mktemp("serve")
    model_dir = _tiny_model_dir(base / "model")
    means, prec = _bank()
    out = package.write_serve_dir(
        model_dir,
        base / "serve",
        labels=LABELS,
        prefix=PREFIX,
        max_len=MAX_LEN,
        temperature=T,
        ood_threshold=0.0,
        retention=0.95,
        model_fingerprint="f" * 64,
        version="vTEST",
        means=means,
        precision=prec,
        note="synthetic",
    )
    router = Router.load(out, "torch")
    _, feats = router.infer(SAMPLES)
    thr = float(np.median(router.ood_scores(feats)))  # mixes abstained / not abstained
    cfg = json.loads((out / "router_config.json").read_text())
    cfg["ood_threshold"] = thr
    (out / "router_config.json").write_text(json.dumps(cfg))
    return out


SAMPLES = [
    "where is my order",
    "book a dock please",
    "hello",
    "refund my order please where is a dock",
    "zzz unknown words only",
    "a",
]


@pytest.fixture(scope="module")
def router(serve_dir: Path) -> Router:
    return Router.load(serve_dir, "torch")


@pytest.fixture
def client(serve_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("ROUTER_MODEL_DIR", str(serve_dir))
    monkeypatch.setenv("ROUTER_BACKEND", "torch")
    monkeypatch.setenv("ROUTER_THREADS", "1")
    with TestClient(serve.create_app()) as c:
        yield c


# ------------------------------------------------------------------ predict() contract
def test_predict_keys_and_types(router: Router) -> None:
    for text in SAMPLES:
        r = router.predict(text)
        assert set(r) == {"label", "confidence", "ood_score", "abstained", "top3"}
        assert isinstance(r["label"], str) and r["label"] in LABELS
        assert isinstance(r["confidence"], float) and 0.0 < r["confidence"] <= 1.0
        assert isinstance(r["ood_score"], float) and r["ood_score"] <= 0.0
        assert isinstance(r["abstained"], bool)
        assert all(set(t) == {"label", "prob"} for t in r["top3"])


def test_top3_sorted_sum_and_confidence(router: Router) -> None:
    for r in router.predict_batch(SAMPLES):
        probs = [t["prob"] for t in r["top3"]]
        assert len(r["top3"]) == 3
        assert probs == sorted(probs, reverse=True)
        assert len({t["label"] for t in r["top3"]}) == 3
        assert sum(probs) <= 1.0 + 1e-12
        assert r["confidence"] == r["top3"][0]["prob"]
        assert r["label"] == r["top3"][0]["label"]


def test_temperature_is_applied(router: Router) -> None:
    logits, feats = router.infer(SAMPLES)
    expected = softmax(logits, T)
    for i, r in enumerate(router.postprocess(logits, feats)):
        assert r["confidence"] == pytest.approx(float(expected[i].max()), abs=1e-12)
    raw = softmax(logits, 1.0).max(axis=1)
    assert not np.allclose(raw, expected.max(axis=1))  # T != 1 really changes the output


def test_abstained_matches_threshold(router: Router) -> None:
    preds = router.predict_batch(SAMPLES)
    thr = router.cfg.ood_threshold
    assert all(p["abstained"] == (p["ood_score"] < thr) for p in preds)
    assert {p["abstained"] for p in preds} == {True, False}  # fixture threshold is the median


def test_ood_score_is_mahalanobis_of_penultimate_features(router: Router) -> None:
    _, feats = router.infer(SAMPLES)
    means, prec = _bank()
    naive = []
    for x in feats.astype(np.float64):
        d = [float(np.sqrt((x - m) @ prec @ (x - m))) for m in means]  # per-class loop
        naive.append(-min(d))
    assert router.ood_scores(feats) == pytest.approx(naive, rel=1e-9)


def test_query_prefix_is_applied(router: Router) -> None:
    ids = router.encode(["hello"])["input_ids"][0].tolist()
    vocab = router.tokenizer.get_vocab()
    assert ids == [vocab["<s>"], vocab["query:"], vocab["hello"], vocab["</s>"]]


def test_truncation_to_max_len(router: Router) -> None:
    enc = router.encode(["hello " * 100])
    assert enc["input_ids"].shape[1] == MAX_LEN


def test_batch_equals_single(router: Router) -> None:
    batch = router.predict_batch(SAMPLES)
    for text, b in zip(SAMPLES, batch, strict=True):
        s = router.predict(text)
        assert s["label"] == b["label"]
        assert s["confidence"] == pytest.approx(b["confidence"], abs=1e-5)
        assert s["ood_score"] == pytest.approx(b["ood_score"], abs=1e-3)


def test_non_str_input_rejected(router: Router) -> None:
    with pytest.raises(TypeError):
        router.predict(123)  # type: ignore[arg-type]
    assert router.predict_batch([]) == []


def test_bad_backend_and_missing_onnx(serve_dir: Path) -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        Router.load(serve_dir, "tensorrt")
    with pytest.raises(FileNotFoundError, match="onnx_export"):
        Router.load(serve_dir, "onnx_int8")


# ----------------------------------------------------------------------------- ONNX
def test_onnx_backends_match_torch(serve_dir: Path, router: Router, tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from intent_router import onnx_export

    d = tmp_path / "s"
    d.mkdir()
    for p in serve_dir.iterdir():
        (d / p.name).write_bytes(p.read_bytes())
    onnx_export.export_fp32(d, d / "onnx/model_fp32.onnx")
    onnx_export.quantize_int8(d / "onnx/model_fp32.onnx", d / "onnx/model_int8.onnx")
    lt, ft = router.infer(SAMPLES)
    fp32 = Router.load(d, "onnx_fp32")
    lo, fo = fp32.infer(SAMPLES)
    assert np.abs(lt - lo).max() < 1e-4 and np.abs(ft - fo).max() < 1e-4  # features too
    # a different batch size / length mix than the export example: dynamic axes really work
    assert np.abs(router.infer(SAMPLES[:1])[0] - fp32.infer(SAMPLES[:1])[0]).max() < 1e-4
    int8 = Router.load(d, "onnx_int8")
    li, fi = int8.infer(SAMPLES)  # quantised: only sanity (shapes, finite), quality is benchmarked
    assert li.shape == lt.shape and fi.shape == ft.shape
    assert np.isfinite(li).all() and np.isfinite(fi).all()
    assert set(int8.predict(SAMPLES[0])) == {
        "label",
        "confidence",
        "ood_score",
        "abstained",
        "top3",
    }


# ------------------------------------------------------------------------------- API
def test_health_ready(client: Any) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True and body["status"] == "ok"
    assert body["model_version"] == "vTEST" and body["model_fingerprint"] == "f" * 64
    assert body["backend"] == "torch"


def test_not_ready_without_lifespan(serve_dir: Path) -> None:
    c = TestClient(serve.create_app())  # no `with`: lifespan never runs, model never loaded
    r = c.get("/health")
    assert r.status_code == 503 and set(r.json()) == {"code", "message"}
    r = c.post("/predict", json={"text": "hello"})
    assert r.status_code == 503 and r.json()["code"] == "not_ready"


def test_predict_endpoint_matches_router(client: Any, router: Router) -> None:
    r = client.post("/predict", json={"text": "where is my order"})
    assert r.status_code == 200
    body = r.json()
    local = router.predict("where is my order")
    assert body["label"] == local["label"]
    assert body["confidence"] == pytest.approx(local["confidence"], abs=1e-9)
    assert body["abstained"] == local["abstained"]
    assert [t["label"] for t in body["top3"]] == [t["label"] for t in local["top3"]]
    assert body["model_version"] == "vTEST" and body["model_fingerprint"] == "f" * 64
    assert body["confidence"] == body["top3"][0]["prob"] and len(body["top3"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   \n\t "},
        {"text": "x" * 2001},
        {"text": 123},
        {"text": None},
        {"text": ["a"]},
        {},
        {"txt": "hello"},
    ],
)
def test_validation_errors_have_code_message_shape(client: Any, payload: dict[str, Any]) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert set(body) == {"code", "message"} and body["code"] == "validation_error"
    assert isinstance(body["message"], str) and body["message"]
    for v in payload.values():  # request text is never echoed back
        if isinstance(v, str) and len(v) > 3:
            assert v not in body["message"]


def test_max_length_text_is_accepted(client: Any) -> None:
    assert client.post("/predict", json={"text": "a" * 2000}).status_code == 200


def test_malformed_json_and_unknown_route_shapes(client: Any) -> None:
    r = client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and set(r.json()) == {"code", "message"}
    r = client.get("/nope")
    assert r.status_code == 404 and set(r.json()) == {"code", "message"}
    assert client.get("/predict").status_code == 405


# --------------------------------------------------- numerics + bench helpers (hand-checked)
def test_mahalanobis_hand_checked() -> None:
    means, prec = np.array([[0.0, 0.0], [10.0, 0.0]]), np.eye(2)
    x = np.array([[3.0, 4.0], [10.0, 1.0]])
    assert mahalanobis_score(x, means, prec) == pytest.approx([-5.0, -1.0])
    bank = MahalanobisBank(means, prec)
    assert bank.score(x) == pytest.approx([-5.0, -1.0])


def test_spearman_hand_checked() -> None:
    # reference values from scipy.stats.spearmanr (ties in both vectors / in x only)
    a, b = np.array([1, 2, 2, 3, 5]), np.array([1, 3, 2, 4, 4])
    assert serve_bench.rank_average(a).tolist() == [1.0, 2.5, 2.5, 4.0, 5.0]
    assert serve_bench.rank_average(b).tolist() == [1.0, 3.0, 2.0, 4.5, 4.5]
    assert serve_bench.spearman(a, b) == pytest.approx(0.9473684210526317, abs=1e-12)
    x, y = np.array([0.1, -3, 2.5, 2.5, 7, -3]), np.array([5, 4, 3, 2, 1, 0])
    assert serve_bench.spearman(x, y) == pytest.approx(-0.17654696590094993, abs=1e-12)
    assert serve_bench.spearman(a, a) == pytest.approx(1.0)
    assert serve_bench.spearman(a, -a) == pytest.approx(-1.0)
    assert serve_bench.spearman(a, np.ones(5)) is None  # constant side: undefined


def test_macro_f1_hand_checked() -> None:
    y, p = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0])
    # class0: tp1 pred2 true2 -> .5; class1: tp2 pred3 true2 -> .8; class2: 0; class3 absent: 0
    assert serve_bench.macro_f1(y, p, 4) == pytest.approx((0.5 + 0.8 + 0.0 + 0.0) / 4)


def test_latency_stats_percentiles() -> None:
    s = serve_bench.latency_stats([float(i) for i in range(1, 101)])
    assert s["p50_ms"] == pytest.approx(50.5) and s["p95_ms"] == pytest.approx(95.05)
    assert s["n_calls"] == 100


def test_backend_quality_agreement_fields() -> None:
    gold = np.array([0, 1, 2, 1])
    ref = {"logits": np.eye(3)[[0, 1, 2, 0]] * 5.0, "scores": np.array([-1.0, -2.0, -9.0, -3.0])}
    q = serve_bench.backend_quality(
        gold,
        np.eye(3)[[0, 1, 1, 0]] * 5.0,  # differs from ref at row 2
        np.array([-1.0, -2.5, -3.0, -9.0]),  # abstention (thr -5) differs at rows 2 and 3
        1.0,
        -5.0,
        ref,
        ["a", "b", "c", "d"],
    )
    assert q["accuracy"] == 0.5 and q["pred_agreement_vs_torch"] == 0.75
    assert q["pred_disagreement_ids"] == ["c"]
    assert q["abstention_disagreement_ids"] == ["c", "d"]
    assert q["abstention_agreement_vs_torch"] == 0.5 and q["n_abstained"] == 1


# --------------------------------------------- bench: test-split gate + logged call types
def test_val_split_needs_no_log(tmp_path: Path) -> None:
    gate = serve_bench.TestUse("val", False, None, False)
    gate.before(serve_bench.CALL_TYPE_QUALITY, "fp", "torch", 5, "serve_quality")  # no-op


def test_test_split_refused_without_flags(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        serve_bench.TestUse("test", False, tmp_path / "log.jsonl", False)
    with pytest.raises(SystemExit):
        serve_bench.TestUse("test", True, None, False)


def test_test_stage_logs_one_line_per_variant_before_inference(
    serve_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "test_eval_log.jsonl"
    gate = serve_bench.TestUse("test", True, log, False)
    seen_before_infer: list[int] = []
    real = serve_bench.run_backend

    def spy(r: Router, texts: list[str], batch: int) -> dict[str, np.ndarray]:
        seen_before_infer.append(len(serve_bench.read_log(log)))  # lines present at inference
        return real(r, texts, batch)

    monkeypatch.setattr(serve_bench, "run_backend", spy)
    ids = [f"t{i}" for i in range(len(SAMPLES))]
    res = serve_bench.stage_quality(
        serve_dir, ids, SAMPLES, ["alpha"] * len(SAMPLES), ["torch"], 1, gate, "fp123"
    )
    lines = serve_bench.read_log(log)
    assert seen_before_infer == [1]  # the line was already on disk when inference ran
    assert len(lines) == 1 and lines[0]["call_type"] == "quantization_inference"
    assert lines[0]["model_fingerprint"] == "fp123" and lines[0]["variant"] == "torch"
    assert set(res["torch"]) >= {"macro_f1", "accuracy", "abstention_rate"}
    with pytest.raises(SystemExit, match="already logged"):  # no second quality run on test
        serve_bench.stage_quality(
            serve_dir, ids, SAMPLES, ["alpha"] * len(SAMPLES), ["torch"], 1, gate, "fp123"
        )
    assert len(serve_bench.read_log(log)) == 1  # refusal logs nothing


def test_latency_on_test_logs_latency_inference(serve_dir: Path, tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    gate = serve_bench.TestUse("test", True, log, False)
    res = serve_bench.stage_latency(serve_dir, SAMPLES, ["torch"], [1], 2, 1, gate, "fp9")
    lines = serve_bench.read_log(log)
    assert [e["call_type"] for e in lines] == ["latency_inference"]
    assert res["torch"]["threads_1"]["n_calls"] == len(SAMPLES)


# ------------------------------------------------------ integration on the real v1 model
V1_SERVE = ROOT / "serve_model"
V1_NPZ = ROOT / "outputs" / "final_v1" / "features_logits.npz"
V1_VAL_CSV = ROOT / "results" / "final_v1" / "val_predictions.csv"
V1_DATA = ROOT / "data" / "dataset.csv"


@pytest.mark.skipif(
    not (V1_SERVE / "router_config.json").exists()
    or not (V1_SERVE / "model.safetensors").exists()
    or not V1_NPZ.exists()
    or not V1_VAL_CSV.exists()
    or not V1_DATA.exists(),
    reason="packaged v1 serve_model / saved arrays / dataset not present",
)
def test_v1_torch_backend_reproduces_saved_val_arrays() -> None:
    z = np.load(V1_NPZ)
    ids, texts, _ = serve_bench.load_split(
        str(V1_DATA),
        str(ROOT / "splits" / "splits.csv"),
        "val",  # VAL only, never test
    )
    r = Router.load(V1_SERVE, "torch")
    logits, feats = r.infer(texts)
    row = {i: k for k, i in enumerate(z["val_ids"].tolist())}
    order = [row[i] for i in ids]
    # probabilities as saved by final.py (T = 1, float32) from the saved logits
    with V1_VAL_CSV.open(newline="") as fh:
        saved = {x["id"]: [float(x[f"prob_{k}"]) for k in range(12)] for x in csv.DictReader(fh)}
    p_saved = np.array([saved[i] for i in ids])
    p_now = softmax(logits, 1.0)
    assert np.abs(p_now - p_saved).max() < 1e-4
    assert np.abs(logits - z["val_logits"][order]).max() < 1e-3
    s_saved = mahalanobis_score(z["val_features"][order], r.bank.means, r.bank.precision)
    s_now = r.ood_scores(feats)
    assert np.abs(s_now - s_saved).max() < 1e-3  # scores are O(100); observed 1.9e-4
    assert r.cfg.version == "v1" and r.cfg.model_fingerprint.startswith("7ca22e16d2ac")
