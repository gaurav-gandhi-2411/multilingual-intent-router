"""Export a serve dir's model to ONNX (fp32 + dynamic int8) with penultimate features.

    python -m intent_router.onnx_export --serve-dir serve_model

Writes <serve-dir>/onnx/{model_fp32.onnx, model_int8.onnx, export_report.json}.

Graph: inputs input_ids / attention_mask (int64, dynamic batch + sequence axes); outputs
`logits` [batch, K] and `features` [batch, hidden]. `features` is the classification head's
dense+tanh output (input of classifier.out_proj), captured with the same forward pre-hook the
training side used for features_logits.npz, so Mahalanobis scoring works on ONNX backends.

Quantisation: onnxruntime.quantization.quantize_dynamic (weights int8, QInt8; activations
quantised dynamically per batch at run time). optimum's ORTQuantizer is a thin wrapper over the
same call and would add a dependency, so the primitive is used directly. Settings are CLI flags
(defaults: per_channel=True, reduce_range=True: on val the four per_channel x reduce_range
combos were within noise of each other (n=74), this one had the smallest max prob drift;
reduce_range also guards the u8*s8 saturation of AVX2-without-VNNI CPUs, e.g. the Zen 3 dev box).

`--int8-mode encoder_linear_only` (builds <serve-dir>/onnx/model_int8_encoder.onnx from the EXISTING
fp32 graph, nothing else is rewritten): only the encoder's weight MatMuls (the Linear layers) are
quantised. Embedding tables (Gather), the classification head (dense + out_proj) and the
activation x activation attention MatMuls stay fp32. The default `full` mode quantises the whole
graph (incl. the 250k-row word-embedding table) and is what hurt v1 (results/serving/).
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from intent_router.predict import (
    ONNX_FILES,
    ONNX_OUTPUTS,
    Router,
    softmax,
)

INT8_MODES = ("full", "encoder_linear_only")
GEMM_MATMUL_SUFFIX = "_MatMul"  # onnxruntime names the MatMul it rewrites a Gemm into
HEAD_MARKER = "classifier"  # node-name fragment of the classification head (dense + out_proj)
OPSET = 17  # LayerNorm is a native op from 17; Gelu (20) is exported as erf and fused by ORT
# Synthetic strings only (never dataset text): different lengths exercise the dynamic axes.
CHECK_TEXTS = (
    "where is my shipment right now",
    "hi",
    "please book a dock appointment for tomorrow morning at the north yard and notify the carrier",
    "wo ist meine lieferung",
    "cancel order number 12345 and refund the deposit please, this is urgent and long " * 3,
)


def _wrapper(model: Any) -> Any:
    """nn.Module returning (logits, features); features via the out_proj forward pre-hook."""
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.captured: list[Any] = []
            model.classifier.out_proj.register_forward_pre_hook(
                lambda _m, args: self.captured.append(args[0])
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
            self.captured.clear()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            return logits, self.captured[-1]

    return Wrapper().eval()


def export_fp32(serve_dir: Path, out_path: Path, opset: int = OPSET) -> None:
    """torch.onnx.export (TorchScript exporter) of the eager fp32 model to out_path."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from intent_router.predict import CONFIG_NAME, RouterConfig

    cfg = RouterConfig.from_file(serve_dir / CONFIG_NAME)
    tok = AutoTokenizer.from_pretrained(str(serve_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(serve_dir), dtype=torch.float32, attn_implementation="eager"
    ).eval()
    wrapper = _wrapper(model)
    enc = tok(
        [cfg.prefix + t for t in CHECK_TEXTS[:2]],
        padding=True,
        truncation=True,
        max_length=cfg.max_len,
        return_tensors="pt",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (enc["input_ids"], enc["attention_mask"]),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=list(ONNX_OUTPUTS),
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
                "features": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # the TorchScript exporter: no onnxscript dependency, stable graphs
        )


def quantize_int8(
    fp32_path: Path, int8_path: Path, per_channel: bool = True, reduce_range: bool = True
) -> None:
    """Dynamic weight int8 (QInt8) quantisation of the MatMul weights of the fp32 graph."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )


def select_encoder_linear_exclusions(nodes: Iterable[tuple[str, str, bool]]) -> list[str]:
    """Names of nodes to EXCLUDE so only the encoder Linear layers are quantised.

    `nodes` = (name, op_type, weight_is_initializer). A MatMul is kept for quantisation iff its
    second input is a constant weight (a Linear layer) and it is not part of the classification
    head. Excluded: head MatMuls, activation x activation MatMuls (attention QK^T, softmax.V) and
    the head Gemms (the exporter emits the 2-D head Linears as Gemm). quantize_dynamic first
    rewrites every Gemm to a MatMul named `<gemm name>_MatMul` and then quantises it even with
    op_types_to_quantize=['MatMul'] (onnxruntime 1.30), so both names are excluded.
    """
    out: list[str] = []
    for name, op, w_const in nodes:
        if op == "MatMul" and (HEAD_MARKER in name or not w_const):
            out.append(name)
        elif op == "Gemm" and HEAD_MARKER in name:
            out += [name, name + GEMM_MATMUL_SUFFIX]
    return out


def graph_op_counts(model: Any) -> dict[str, int]:
    """Op-type histogram of an onnx ModelProto's top-level graph."""
    return dict(collections.Counter(n.op_type for n in model.graph.node))


def quantize_int8_encoder_linear(
    fp32_path: Path, int8_path: Path, per_channel: bool = True, reduce_range: bool = True
) -> dict[str, Any]:
    """Dynamic int8 of the encoder Linear MatMuls only; returns the node-quantisation counts.

    Asserts by inspecting the quantised graph: the MatMulInteger nodes are exactly the selected
    encoder Linear MatMuls, every excluded node is still an fp32 MatMul (head Gemms are rewritten
    to MatMul + Add by the quantiser but keep float32 weights), and no Gather table is quantised.
    """
    import onnx
    from onnx import TensorProto
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = onnx.load(str(fp32_path))
    inits = {i.name for i in src.graph.initializer}
    mm = [(n.name, n.op_type, len(n.input) > 1 and n.input[1] in inits) for n in src.graph.node]
    exclude = select_encoder_linear_exclusions(mm)
    matmuls = {name for name, op, _ in mm if op == "MatMul"}
    expected_q = matmuls - set(exclude)  # the encoder Linear MatMuls
    n_head_gemm = sum(op == "Gemm" and HEAD_MARKER in name for name, op, _ in mm)
    fp32_ops = graph_op_counts(src)
    del src
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=exclude,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    q = onnx.load(str(int8_path))
    q_ops = graph_op_counts(q)
    init_names = {i.name for i in q.graph.initializer}
    float_inits = {i.name for i in q.graph.initializer if i.data_type == TensorProto.FLOAT}
    quantized = {
        n.name.removesuffix("_quant") for n in q.graph.node if n.op_type == "MatMulInteger"
    }
    still_matmul = {n.name: n for n in q.graph.node if n.op_type == "MatMul"}
    head_mm = [n for name, n in still_matmul.items() if name.endswith(GEMM_MATMUL_SUFFIX)]
    gather_w = {
        n.input[0] for n in q.graph.node if n.op_type == "Gather" and n.input[0] in init_names
    }
    counts = {
        "fp32_graph_ops": {k: fp32_ops.get(k, 0) for k in ("MatMul", "Gemm", "Gather")},
        "quantized_graph_ops": {k: q_ops.get(k, 0) for k in ("MatMul", "MatMulInteger", "Gather")},
        "n_matmul_fp32_graph": len(matmuls),
        "n_head_gemm_fp32_graph": n_head_gemm,
        "n_quantized_encoder_linear_expected": len(expected_q),
        "n_matmul_integer_in_quantized": len(quantized),
        "n_matmul_left_fp32_act_act": len(still_matmul) - len(head_mm),
        "n_head_linear_left_fp32": len(head_mm),
        "n_head_linear_with_float_weights": sum(n.input[1] in float_inits for n in head_mm),
        "n_gather_tables": len(gather_w),
        "n_gather_tables_fp32": sum(w in float_inits for w in gather_w),
        "exclude_list_size": len(exclude),
    }
    assert quantized == expected_q, "quantised set != encoder Linear MatMuls"
    assert len(head_mm) == n_head_gemm == counts["n_head_linear_with_float_weights"], counts
    assert counts["n_gather_tables_fp32"] == counts["n_gather_tables"], "a Gather was quantised"
    return counts


def compare_to_torch(serve_dir: Path, backend: str, torch_router: Router) -> dict[str, Any]:
    """Max abs diffs of logits / features / temperature-scaled probs vs the torch backend."""
    router = Router.load(serve_dir, backend)
    texts = list(CHECK_TEXTS)
    lt, ft = torch_router.infer(texts)
    lo, fo = router.infer(texts)
    pt, po = softmax(lt, torch_router.cfg.temperature), softmax(lo, router.cfg.temperature)
    return {
        "backend": backend,
        "n_texts": len(texts),
        "max_abs_diff_logits": float(np.abs(lt - lo).max()),
        "max_abs_diff_features": float(np.abs(ft - fo).max()),
        "max_abs_diff_probs": float(np.abs(pt - po).max()),
        "argmax_equal": bool((lt.argmax(1) == lo.argmax(1)).all()),
    }


def export_all(
    serve_dir: Path, per_channel: bool = True, reduce_range: bool = True, opset: int = OPSET
) -> dict[str, Any]:
    """Export fp32, quantise to int8, check both against torch, write export_report.json."""
    import onnx
    import onnxruntime
    import torch

    fp32 = serve_dir / ONNX_FILES["onnx_fp32"]
    int8 = serve_dir / ONNX_FILES["onnx_int8"]
    export_fp32(serve_dir, fp32, opset)
    onnx.checker.check_model(str(fp32))  # path form: also valid for >2 GB external-data models
    quantize_int8(fp32, int8, per_channel, reduce_range)
    torch_router = Router.load(serve_dir, "torch")
    report: dict[str, Any] = {
        "opset": opset,
        "exporter": "torch.onnx.export (TorchScript, dynamo=False)",
        "quantization": {
            "tool": "onnxruntime.quantization.quantize_dynamic",
            "weight_type": "QInt8",
            "activations": "dynamic (uint8, computed per batch at run time)",
            "per_channel": per_channel,
            "reduce_range": reduce_range,
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "size_bytes": {"onnx_fp32": fp32.stat().st_size, "onnx_int8": int8.stat().st_size},
        "checks_vs_torch": [
            compare_to_torch(serve_dir, "onnx_fp32", torch_router),
            compare_to_torch(serve_dir, "onnx_int8", torch_router),
        ],
    }
    (fp32.parent / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve-dir", type=Path, required=True)
    ap.add_argument("--no-per-channel", action="store_true")
    ap.add_argument("--no-reduce-range", action="store_true")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--int8-mode", choices=INT8_MODES, default="full")
    a = ap.parse_args(argv)
    if a.int8_mode == "encoder_linear_only":
        fp32 = a.serve_dir / ONNX_FILES["onnx_fp32"]
        out = a.serve_dir / ONNX_FILES["onnx_int8_encoder"]
        counts = quantize_int8_encoder_linear(
            fp32, out, not a.no_per_channel, not a.no_reduce_range
        )
        print(json.dumps({"out": str(out), "size_bytes": out.stat().st_size, **counts}, indent=2))
        return
    rep = export_all(a.serve_dir, not a.no_per_channel, not a.no_reduce_range, a.opset)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
