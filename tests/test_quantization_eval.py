"""Pure-logic tests for the v1 quantisation study (no model files needed)."""

from __future__ import annotations

import pytest

from intent_router.onnx_export import select_encoder_linear_exclusions
from intent_router.quantization_eval import (
    MARGIN,
    adoption_decision,
    f1_delta,
    passes_bar,
    select_variant,
)

NODES = [
    ("/model/roberta/encoder/layer.0/attention/self/query/MatMul", "MatMul", True),
    ("/model/roberta/encoder/layer.0/attention/self/MatMul", "MatMul", False),  # QK^T
    ("/model/roberta/encoder/layer.0/attention/self/MatMul_1", "MatMul", False),  # softmax.V
    ("/model/roberta/encoder/layer.0/output/dense/MatMul", "MatMul", True),
    ("/model/classifier/dense/MatMul", "MatMul", True),  # head, in case it exports as MatMul
    (
        "/model/classifier/out_proj/Gemm",
        "Gemm",
        True,
    ),  # head Gemm: quantize_dynamic would rewrite it
    ("/model/roberta/pooler/Gemm", "Gemm", True),  # a non-head Gemm is left alone by the selector
    ("/model/roberta/embeddings/word_embeddings/Gather", "Gather", True),
]


def test_exclusions_keep_only_encoder_linear() -> None:
    ex = select_encoder_linear_exclusions(NODES)
    assert ex == [
        "/model/roberta/encoder/layer.0/attention/self/MatMul",
        "/model/roberta/encoder/layer.0/attention/self/MatMul_1",
        "/model/classifier/dense/MatMul",
        "/model/classifier/out_proj/Gemm",
        "/model/classifier/out_proj/Gemm_MatMul",  # the quantiser's name for that Gemm
    ]
    matmuls = [n for n, op, _ in NODES if op == "MatMul"]
    kept = [n for n in matmuls if n not in ex]
    assert kept == [
        "/model/roberta/encoder/layer.0/attention/self/query/MatMul",
        "/model/roberta/encoder/layer.0/output/dense/MatMul",
    ]


def test_exclusions_empty_graph() -> None:
    assert select_encoder_linear_exclusions([]) == []


def test_passes_bar_boundaries() -> None:
    assert passes_bar(0.0) and passes_bar(MARGIN) and passes_bar(0.01)
    assert not passes_bar(MARGIN - 0.001)
    assert f1_delta(0.88, 0.89) == pytest.approx(-0.01)


def test_adoption_requires_both_splits() -> None:
    assert adoption_decision({"val": -0.004, "test": 0.0})["decision"] == "adopted_as_option"
    d = adoption_decision({"val": 0.0, "test": -0.02})
    assert d["decision"] == "rejected" and d["passes"] == {"val": True, "test": False}
    assert adoption_decision({"val": -0.0527, "test": -0.05})["decision"] == "rejected"
    with pytest.raises(ValueError):
        adoption_decision({"val": 0.0})


def test_select_variant_first_pass_else_best() -> None:
    order = ["V1", "V2", "V3"]
    assert select_variant({"V1": -0.03, "V2": -0.001}, {"V1": 0.8, "V2": 0.9}, order) == "V2"
    # none passes -> best val macro-F1 among evaluated, ties resolved to the earliest
    deltas = {"V1": -0.03, "V2": -0.02, "V3": -0.02}
    f1 = {"V1": 0.80, "V2": 0.85, "V3": 0.85}
    assert select_variant(deltas, f1, order) == "V2"
    with pytest.raises(ValueError):
        select_variant({}, {}, order)


def test_default_backend_is_onnx_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    from intent_router import predict

    monkeypatch.delenv("ROUTER_BACKEND", raising=False)
    assert predict.env_settings()[1] == "onnx_fp32" == predict.DEFAULT_BACKEND
    assert {"torch", "onnx_int8", "onnx_int8_encoder"} <= set(predict.BACKENDS)
