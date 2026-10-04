"""Pure parts of the Colab latency benchmark (no model, no benchmark is run here)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from intent_router import latency_colab as lc


def test_cpu_model_reads_proc_cpuinfo(tmp_path: Path) -> None:
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nmodel name\t: Fake Xeon CPU @ 2.20GHz\nflags\t: x\n")
    assert lc.cpu_model(f) == "Fake Xeon CPU @ 2.20GHz"


def test_cpu_model_falls_back_when_cpuinfo_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lc.serve_bench, "cpu_model", lambda: "fallback-cpu")
    assert lc.cpu_model(tmp_path / "missing") == "fallback-cpu"


def test_gpu_name_none_without_nvidia_smi() -> None:
    def no_smi(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("nvidia-smi")

    def failing(*_a: Any, **_k: Any) -> Any:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    assert lc.gpu_name(no_smi) is None
    assert lc.gpu_name(failing) is None
    assert lc.gpu_name(lambda *_a, **_k: SimpleNamespace(stdout="\n")) is None


def test_gpu_name_parses_first_line() -> None:
    fake = lambda *_a, **_k: SimpleNamespace(stdout="Tesla T4\n")  # noqa: E731
    assert lc.gpu_name(fake) == "Tesla T4"


def test_runtime_info_colab_gpu() -> None:
    rt = lc.runtime_info(
        {"COLAB_RELEASE_TAG": "release-colab-20260101", "COLAB_GPU": "1"}, lambda: "Tesla T4"
    )
    assert rt["colab"] is True and rt["accelerator"] == "Tesla T4"
    assert rt["colab_release_tag"] == "release-colab-20260101" and rt["colab_gpu_env"] == "1"
    assert rt["label"] == "Colab, Tesla T4 (accelerator unused: CPU benchmark)"


def test_runtime_info_no_gpu_no_colab_never_calls_cuda() -> None:
    rt = lc.runtime_info({}, lambda: None)
    assert rt == {
        "colab": False,
        "colab_release_tag": None,
        "colab_gpu_env": None,
        "accelerator": "CPU-only",
        "label": "non-Colab, CPU-only",
    }
    cpu_colab = lc.runtime_info({"COLAB_RELEASE_TAG": "r"}, lambda: None)
    assert cpu_colab["label"] == "Colab, CPU-only"


def test_thread_counts() -> None:
    assert lc.thread_counts(2) == [1, 2]
    assert lc.thread_counts(8) == [1, 8]
    assert lc.thread_counts(1) == [1]
    assert lc.thread_counts(None) == [1]


def _lat() -> dict[str, Any]:
    st = {"threads": 1, "n_calls": 222, "p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.5}
    return {
        "torch": {"threads_1": st, "threads_2": {**st, "threads": 2}},
        "onnx_fp32": {"threads_1": st, "threads_2": {**st, "threads": 2}},
    }


def test_collect_environment_adds_cpu_model_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "cpuinfo"
    f.write_text("model name\t: Fake CPU\n")
    monkeypatch.setattr(
        lc.serve_bench, "environment", lambda: {"torch": "2.x", "os_cpu_count": 2, "cpu_model": "x"}
    )
    env = lc.collect_environment({"COLAB_RELEASE_TAG": "r"}, lambda: None, f)
    assert env["cpu_model"] == "Fake CPU" and env["os_cpu_count"] == 2
    assert env["runtime_type"] == "Colab, CPU-only" and env["runtime"]["colab"] is True


def test_build_and_write_result_schema_matches_what_the_report_reads(tmp_path: Path) -> None:
    res = lc.build_result(
        ids=["a", "b"],
        fingerprint="f" * 64,
        cfg={"version": "v1"},
        environment={"cpu_model": "c", "os_cpu_count": 2, "runtime_type": "Colab, CPU-only"},
        sizes_bytes={"onnx_fp32": 5},
        latency=_lat(),
        threads=[1, 2],
        warmup=10,
        repeats=3,
    )
    out = tmp_path / "serving" / "latency_colab.json"
    lc.write_result(res, out)
    d = json.loads(out.read_text(encoding="utf-8"))
    # the keys report/data.py:_serving reads (shared with results/serving/latency_v1.json)
    assert d["split"] == "val" and d["n_rows"] == 2 and d["stage"] == "latency"
    assert d["settings"] == {
        "warmup": 10,
        "repeats": 3,
        "threads": [1, 2],
        "backends": ["torch", "onnx_fp32"],
    }
    assert d["environment"]["os_cpu_count"] == 2 and d["environment"]["cpu_model"] == "c"
    assert d["latency"]["onnx_fp32"]["threads_2"]["p95_ms"] == 2.0
    assert len(d["ids_sha256"]) == 64 and "onnx_export_check" not in d
    assert "val" in d["note"] and "test" not in d["split"]


def test_run_uses_val_split_only_and_no_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole orchestration with the heavy parts stubbed: val split, [1, cpu_count] threads."""
    from intent_router import onnx_export

    serve = tmp_path / "serve"
    serve.mkdir()
    (serve / "router_config.json").write_text(
        json.dumps({"version": "v1", "model_fingerprint": "ab" * 32}), encoding="utf-8"
    )
    ret = tmp_path / "ood_shipped.json"
    ret.write_text(json.dumps({"retention_target": 0.95}), encoding="utf-8")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(lc, "package_fresh_model", lambda **kw: seen.update(pkg=kw) or serve)
    monkeypatch.setattr(onnx_export, "export_fp32", lambda *a, **k: seen.update(export=a))
    monkeypatch.setattr(onnx_export, "compare_to_torch", lambda *a, **k: {"argmax_equal": True})
    monkeypatch.setattr("intent_router.predict.Router.load", lambda *a, **k: object())
    monkeypatch.setattr(
        lc.serve_bench,
        "load_split",
        lambda d, s, split: seen.update(split=split) or (["i1", "i2"], ["t1", "t2"], ["x", "y"]),
    )

    def stage(
        serve_dir: Path, texts: list[str], backends: list[str], threads: list[int], *a: Any
    ) -> Any:
        seen.update(threads=threads, backends=backends, texts=texts)
        return _lat()

    monkeypatch.setattr(lc.serve_bench, "stage_latency", stage)
    monkeypatch.setattr(lc.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(lc, "gpu_name", lambda *a, **k: None)
    monkeypatch.setattr(
        lc,
        "collect_environment",
        lambda: {"cpu_model": "c", "os_cpu_count": 2, "runtime_type": "Colab, CPU-only"},
    )
    monkeypatch.setattr(lc.serve_bench, "variant_sizes", lambda d: {"onnx_fp32": 1})
    a = argparse.Namespace(
        model_dir=tmp_path,
        features=tmp_path / "f.npz",
        results_dir=tmp_path,
        config=tmp_path / "c.yaml",
        serve_dir=serve,
        out=tmp_path / "out" / "latency_colab.json",
        data="d.csv",
        splits="s.csv",
        retention_from=str(ret),
        warmup=10,
        repeats=3,
    )
    res = lc.run(a)
    assert seen["split"] == "val" and seen["threads"] == [1, 2]
    assert seen["backends"] == ["torch", "onnx_fp32"] and seen["texts"] == ["t1", "t2"]
    assert seen["pkg"]["retention"] == 0.95
    assert res["onnx_export_check"] == {"argmax_equal": True}
    assert json.loads(a.out.read_text(encoding="utf-8"))["settings"]["threads"] == [1, 2]
