"""CPU batch-1 latency of the freshly trained v1 on the Colab runtime (the canonical latency run).

Called by the Colab notebook after final training + Track A:

    python -m intent_router.latency_colab --model-dir outputs/final_model \
        --features outputs/final/features_logits.npz --results-dir results_colab/final \
        --config /content/colab_cfg/final.yaml --out results_colab/serving/latency_colab.json

It (1) packages the fresh model into a serve dir (same files as `intent_router.package`, but fitted
from the fresh features: the committed shipped threshold belongs to the RTX 3070 model, not to this
one), (2) exports the ONNX fp32 graph with `intent_router.onnx_export`, (3) times PyTorch fp32 and
ONNX fp32 on the CPU with `intent_router.serve_bench.stage_latency` (batch 1, warm-up, repeats,
1 and `os.cpu_count()` threads) over the VAL texts only (never test rows), and (4) writes the
same JSON
schema as `results/serving/latency_v1.json`, plus the runtime description the report prints.
GPU is never used: the accelerator, if any, is only recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from intent_router import serve_bench

DEFAULT_OUT = "results_colab/serving/latency_colab.json"
BACKENDS = ["torch", "onnx_fp32"]
NOTE = (
    "freshly trained v1 on the stated runtime; PyTorch fp32 and ONNX fp32 on CPU, batch 1, val "
    "texts only; the accelerator (if any) is unused; shared cloud vCPUs, not an idle-machine run"
)


# ------------------------------------------------------------------------------ environment
def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str:
    """CPU model: first `model name` line of /proc/cpuinfo (Linux), else serve_bench's fallback."""
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return serve_bench.cpu_model()


def gpu_name(run: Callable[..., Any] = subprocess.run) -> str | None:
    """GPU name from nvidia-smi, or None when there is no GPU / no nvidia-smi."""
    try:
        res = run(  # noqa: S603 - fixed argv, no shell
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = [ln.strip() for ln in str(res.stdout).splitlines() if ln.strip()]
    return names[0] if names else None


def runtime_info(
    environ: Mapping[str, str] | None = None, gpu: Callable[[], str | None] = gpu_name
) -> dict[str, Any]:
    """Runtime description (Colab or not, accelerator or 'CPU-only') and a one-line label."""
    env = os.environ if environ is None else environ
    tag = env.get("COLAB_RELEASE_TAG") or None
    colab = tag is not None or "COLAB_GPU" in env
    accel = gpu() or "CPU-only"
    label = f"{'Colab' if colab else 'non-Colab'}, {accel}"
    if accel != "CPU-only":
        label += " (accelerator unused: CPU benchmark)"
    return {
        "colab": colab,
        "colab_release_tag": tag,
        "colab_gpu_env": env.get("COLAB_GPU") or None,
        "accelerator": accel,
        "label": label,
    }


def collect_environment(
    environ: Mapping[str, str] | None = None,
    gpu: Callable[[], str | None] = gpu_name,
    cpuinfo: Path = Path("/proc/cpuinfo"),
) -> dict[str, Any]:
    """serve_bench.environment() (library versions, cores) with the CPU model and runtime added."""
    env = serve_bench.environment()
    env["cpu_model"] = cpu_model(cpuinfo)
    env["platform"] = platform.platform()
    rt = runtime_info(environ, gpu)
    env["runtime"] = rt
    env["runtime_type"] = rt["label"]
    return env


def thread_counts(cpu_count: int | None) -> list[int]:
    """[1, os.cpu_count()]; just [1] on a single-vCPU machine."""
    n = int(cpu_count or 1)
    return [1] if n <= 1 else [1, n]


# --------------------------------------------------------------------------------- result
def build_result(
    *,
    ids: list[str],
    fingerprint: str,
    cfg: Mapping[str, Any],
    environment: dict[str, Any],
    sizes_bytes: dict[str, Any],
    latency: dict[str, Any],
    threads: list[int],
    warmup: int,
    repeats: int,
    export_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The latency JSON: same keys as results/serving/latency_v1.json (minus idle/load blocks)."""
    result: dict[str, Any] = {
        "stage": "latency",
        "source": "colab",
        "split": "val",
        "n_rows": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "model_fingerprint": fingerprint,
        "model_version": cfg["version"],
        "environment": environment,
        "sizes_bytes": sizes_bytes,
        "settings": {
            "warmup": warmup,
            "repeats": repeats,
            "threads": threads,
            "backends": list(latency),
        },
        "note": NOTE,
        "latency": latency,
    }
    if export_check is not None:
        result["onnx_export_check"] = export_check
    return result


def write_result(result: dict[str, Any], out: Path) -> None:
    """Write the JSON (numbers and ids only; no dataset text)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------ packaging
def package_fresh_model(
    *,
    model_dir: Path,
    features_npz: Path,
    results_dir: Path,
    config_path: Path,
    data: str,
    splits: str,
    retention: float,
    out: Path,
) -> Path:
    """Serve dir for the freshly trained model (bank + threshold fitted from its own features).

    Reads only train/val arrays and train ids/labels; temperature from the fresh Track A file.
    """
    import torch
    import yaml
    from transformers import AutoModelForSequenceClassification

    from intent_router.evaluate import threshold_at_retention
    from intent_router.models import state_dict_sha256
    from intent_router.ood import fit_gaussian_lw
    from intent_router.package import write_serve_dir
    from intent_router.predict import mahalanobis_score

    mcfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    labels = [mcfg["id2label"][str(i)] for i in range(len(mcfg["id2label"]))]
    train_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["train"]
    track_a = json.loads((results_dir / "track_a.json").read_text(encoding="utf-8"))
    temperature = float(track_a["calibration"]["temperature"])
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), dtype=torch.float32)
    fingerprint = state_dict_sha256(model)
    del model

    z = np.load(features_npz)
    tr_ids, _, tr_labels = serve_bench.load_split(data, splits, "train")
    lab_by_id = dict(zip(tr_ids, tr_labels, strict=True))
    l2i = {lab: i for i, lab in enumerate(labels)}
    y_tr = np.array([l2i[lab_by_id[i]] for i in z["train_ids"]])
    g = fit_gaussian_lw(z["train_features"], y_tr, len(labels))
    thr = threshold_at_retention(
        mahalanobis_score(z["val_features"], g.means, g.precision), retention
    )
    return write_serve_dir(
        model_dir,
        out,
        labels=labels,
        prefix=str(train_cfg["query_prefix"]),
        max_len=int(train_cfg["max_len"]),
        temperature=temperature,
        ood_threshold=float(thr),
        retention=retention,
        model_fingerprint=fingerprint,
        version="v1",
        means=g.means,
        precision=g.precision,
        shrinkage=g.shrinkage,
        note="Colab-trained v1 (latency benchmark only)",
    )


def run(a: argparse.Namespace) -> dict[str, Any]:
    """Package, export, benchmark, write; returns the result dict."""
    from intent_router import onnx_export
    from intent_router.predict import ONNX_FILES, Router

    retention = float(
        json.loads(Path(a.retention_from).read_text(encoding="utf-8"))["retention_target"]
    )
    serve = package_fresh_model(
        model_dir=a.model_dir,
        features_npz=a.features,
        results_dir=a.results_dir,
        config_path=a.config,
        data=a.data,
        splits=a.splits,
        retention=retention,
        out=a.serve_dir,
    )
    onnx_export.export_fp32(serve, serve / ONNX_FILES["onnx_fp32"])
    cfg = json.loads((serve / "router_config.json").read_text(encoding="utf-8"))
    check = onnx_export.compare_to_torch(serve, "onnx_fp32", Router.load(serve, "torch"))
    ids, texts, _ = serve_bench.load_split(a.data, a.splits, "val")  # val only, never test
    threads = thread_counts(os.cpu_count())
    gate = serve_bench.TestUse("val", False, None, False)
    latency = serve_bench.stage_latency(
        serve, texts, BACKENDS, threads, a.warmup, a.repeats, gate, cfg["model_fingerprint"]
    )
    result = build_result(
        ids=ids,
        fingerprint=cfg["model_fingerprint"],
        cfg=cfg,
        environment=collect_environment(),
        sizes_bytes=serve_bench.variant_sizes(serve),
        latency=latency,
        threads=threads,
        warmup=a.warmup,
        repeats=a.repeats,
        export_check=check,
    )
    write_result(result, a.out)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, default=Path("outputs/final_model"))
    ap.add_argument("--features", type=Path, default=Path("outputs/final/features_logits.npz"))
    ap.add_argument("--results-dir", type=Path, default=Path("results_colab/final"))
    ap.add_argument("--config", type=Path, required=True, help="the (redirected) final.yaml")
    ap.add_argument("--serve-dir", type=Path, default=Path("serve_model_colab"))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--retention-from", default="results/final/ood_shipped.json")
    ap.add_argument("--warmup", type=int, default=serve_bench.WARMUP)
    ap.add_argument("--repeats", type=int, default=serve_bench.REPEATS)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU benchmark: no CUDA call can happen by accident
    result = run(a)
    print(json.dumps(result, indent=2))
    print(f"[latency_colab] wrote {a.out}")


if __name__ == "__main__":
    main()
