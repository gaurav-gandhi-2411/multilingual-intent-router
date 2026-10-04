"""v1 quantisation study: torch fp32 vs ONNX fp32 vs ONNX int8 (encoder Linear layers only).

    python -m intent_router.quantization_eval --serve-dir serve_model \
        --out-val results/serving/quantization_v1_val.json \
        --out-test results/serving/quantization_v1_test.json \
        --allow-test --test-log results/final/test_eval_log.jsonl

PRE-REGISTERED PROTOCOL (fixed before any number was seen; not to be tuned afterwards)
  Variants (at most three, tried in this order, each fully logged):
    V1 per_channel=True,  reduce_range=True    (dev CPU: AVX2 without VNNI, u8*s8 guard)
    V2 per_channel=True,  reduce_range=False
    V3 per_channel=False, reduce_range=True
  Selection on VAL only: the first variant whose val macro-F1 delta vs torch fp32 is >= -0.005;
  if none qualifies, the variant with the best val macro-F1 (ties: earliest) is carried to test.
  TEST is read once for the carried variant (plus torch + onnx_fp32), as an inference-only
  quantisation comparison logged as call_type `quantization_inference`: NOT a Track A evaluation.
  Adoption rule: adopt int8 as an OPTION only if macro-F1 delta (int8 - torch fp32) >= -0.005 on
  BOTH val and test; otherwise the decision is `rejected` and the numbers are recorded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from intent_router import serve_bench as sb
from intent_router.onnx_export import quantize_int8_encoder_linear
from intent_router.predict import ONNX_FILES, Router

MARGIN = -0.005  # max tolerated macro-F1 drop vs torch fp32 (n=74: one flipped row ~ 0.01-0.02)
VARIANTS: tuple[tuple[str, bool, bool], ...] = (
    ("V1", True, True),  # (name, per_channel, reduce_range)
    ("V2", True, False),
    ("V3", False, True),
)
INT8 = "onnx_int8_encoder"
ADOPTION_RULE = (
    "adopt int8 (encoder_linear_only) as an OPTION only if macro-F1 delta (int8 - torch fp32) >= "
    f"{MARGIN} on BOTH val and test; else rejected. Variants (max 3) selected on val only."
)


# ----------------------------------------------------------------------- pure decision logic
def f1_delta(int8_f1: float, ref_f1: float) -> float:
    """macro-F1 of the int8 variant minus the fp32 reference."""
    return int8_f1 - ref_f1


def passes_bar(delta: float, margin: float = MARGIN) -> bool:
    """True iff the delta is no worse than `margin` (tiny epsilon: -0.005 itself passes)."""
    return delta >= margin - 1e-12


def select_variant(val_deltas: dict[str, float], val_f1: dict[str, float], order: list[str]) -> str:
    """First variant (in `order`, among those evaluated) passing the val bar, else best val F1."""
    evaluated = [v for v in order if v in val_deltas]
    if not evaluated:
        raise ValueError("no variant was evaluated")
    for v in evaluated:
        if passes_bar(val_deltas[v]):
            return v
    return max(evaluated, key=lambda v: (val_f1[v], -evaluated.index(v)))


def adoption_decision(deltas: dict[str, float], margin: float = MARGIN) -> dict[str, Any]:
    """Pre-registered adoption rule over {'val': delta, 'test': delta}; both splits required."""
    if set(deltas) != {"val", "test"}:
        raise ValueError("adoption needs exactly the val and test deltas")
    ok = {k: passes_bar(v, margin) for k, v in deltas.items()}
    adopted = all(ok.values())
    return {
        "rule": ADOPTION_RULE,
        "margin": margin,
        "macro_f1_delta_vs_torch": deltas,
        "passes": ok,
        "decision": "adopted_as_option" if adopted else "rejected",
    }


# ------------------------------------------------------------------------------ evaluation
def evaluate_split(
    serve_dir: Path,
    split: str,
    data: str,
    splits: str,
    backends: list[str],
    gate: sb.TestUse,
    fp: str,
    threads: int | None,
) -> dict[str, Any]:
    """Per-backend metrics vs torch on one split (torch first = reference)."""
    ids, texts, gold_lab = sb.load_split(data, splits, split)
    res: dict[str, Any] = {}
    ref: dict[str, np.ndarray] | None = None
    gold: np.ndarray | None = None
    for be in backends:
        router = Router.load(serve_dir, be, threads)
        if gold is None:
            gold = np.array([router.cfg.labels.index(x) for x in gold_lab])
        gate.before(sb.CALL_TYPE_QUALITY, fp, be, len(texts), "quantization_v1")
        arr = sb.run_backend(router, texts, sb.QUALITY_BATCH)
        ref = arr if be == "torch" else ref
        res[be] = sb.backend_quality(
            gold,
            arr["logits"],
            arr["scores"],
            router.cfg.temperature,
            router.cfg.ood_threshold,
            ref if be != "torch" else None,
            ids,
        )
        print(f"[{split}] {be}: f1={res[be]['macro_f1']:.4f} acc={res[be]['accuracy']:.4f}")
        del router
        gc.collect()
    res["_n_rows"] = len(ids)
    res["_ids_sha256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return res


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serve-dir", type=Path, required=True)
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--out-val", type=Path, required=True)
    ap.add_argument("--out-test", type=Path, required=True)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--test-log", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = json.loads((a.serve_dir / "router_config.json").read_text(encoding="utf-8"))
    fp = cfg["model_fingerprint"]
    int8_path = a.serve_dir / ONNX_FILES[INT8]
    fp32_path = a.serve_dir / ONNX_FILES["onnx_fp32"]
    val_gate = sb.TestUse("val", False, None, False)
    base: dict[str, Any] = {
        "model_fingerprint": fp,
        "model_version": cfg["version"],
        "ood_threshold": cfg["ood_threshold"],
        "temperature": cfg["temperature"],
        "environment": sb.environment(),
        "git_sha": sb._git_sha(),
        "sizes_bytes": sb.variant_sizes(a.serve_dir),
        "adoption_rule": ADOPTION_RULE,
        "protocol": __doc__,
        "settings": {
            "quality_batch": sb.QUALITY_BATCH,
            "threads": a.threads,
            "quantize_dynamic": {
                "op_types_to_quantize": ["MatMul"],
                "weight_type": "QInt8",
                "mode": "encoder_linear_only",
            },
        },
    }

    # ---- val: torch + onnx_fp32 once, then each int8 variant in pre-registered order
    val = evaluate_split(
        a.serve_dir, "val", a.data, a.splits, ["torch", "onnx_fp32"], val_gate, fp, a.threads
    )
    ref_f1 = val["torch"]["macro_f1"]
    variants: list[dict[str, Any]] = []
    val_deltas: dict[str, float] = {}
    val_f1: dict[str, float] = {}
    last_built: str | None = None
    for name, pc, rr in VARIANTS:
        counts = quantize_int8_encoder_linear(fp32_path, int8_path, pc, rr)
        last_built = name
        per = evaluate_split(
            a.serve_dir, "val", a.data, a.splits, ["torch", INT8], val_gate, fp, a.threads
        )
        m = per[INT8]
        val_f1[name] = m["macro_f1"]
        val_deltas[name] = f1_delta(m["macro_f1"], ref_f1)
        variants.append(
            {
                "name": name,
                "per_channel": pc,
                "reduce_range": rr,
                "size_bytes": int8_path.stat().st_size,
                "node_counts": counts,
                "val": m,
                "val_macro_f1_delta_vs_torch": val_deltas[name],
                "val_passes_bar": passes_bar(val_deltas[name]),
            }
        )
        print(f"[variant {name}] pc={pc} rr={rr} val delta={val_deltas[name]:+.4f}")
        if passes_bar(val_deltas[name]):
            break
    selected = select_variant(val_deltas, val_f1, [v[0] for v in VARIANTS])
    sel = next(v for v in VARIANTS if v[0] == selected)
    if last_built != selected:  # file on disk must be the carried variant
        quantize_int8_encoder_linear(fp32_path, int8_path, sel[1], sel[2])
    base["variants"] = variants
    base["selected_variant"] = selected
    base["selected_settings"] = {"per_channel": sel[1], "reduce_range": sel[2]}
    base["sizes_bytes"] = sb.variant_sizes(a.serve_dir)
    val_out = {**base, "split": "val", "n_rows": val["_n_rows"], "ids_sha256": val["_ids_sha256"]}
    val_out["quality"] = {
        "torch": val["torch"],
        "onnx_fp32": val["onnx_fp32"],
        INT8: next(v for v in variants if v["name"] == selected)["val"],
    }
    a.out_val.parent.mkdir(parents=True, exist_ok=True)
    a.out_val.write_text(json.dumps(val_out, indent=2), encoding="utf-8")

    # ---- test: one logged inference-only pass for the carried variant (+ torch, onnx_fp32)
    gate = sb.TestUse("test", a.allow_test, a.test_log, False)
    test = evaluate_split(
        a.serve_dir, "test", a.data, a.splits, ["torch", "onnx_fp32", INT8], gate, fp, a.threads
    )
    test_delta = f1_delta(test[INT8]["macro_f1"], test["torch"]["macro_f1"])
    decision = adoption_decision({"val": val_deltas[selected], "test": test_delta})
    decision["onnx_fp32_macro_f1_delta_vs_torch"] = {
        "val": f1_delta(val["onnx_fp32"]["macro_f1"], val["torch"]["macro_f1"]),
        "test": f1_delta(test["onnx_fp32"]["macro_f1"], test["torch"]["macro_f1"]),
    }
    note = (
        "inference-only quantisation comparison on test (call_type quantization_inference); "
        "NOT a Track A evaluation"
    )
    test_out = {
        **base,
        "split": "test",
        "note": note,
        "n_rows": test["_n_rows"],
        "ids_sha256": test["_ids_sha256"],
        "quality": {k: v for k, v in test.items() if not k.startswith("_")},
        "decision": decision,
    }
    a.out_test.parent.mkdir(parents=True, exist_ok=True)
    a.out_test.write_text(json.dumps(test_out, indent=2), encoding="utf-8")
    val_out["decision"] = decision
    a.out_val.write_text(json.dumps(val_out, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
