"""Serving latency + quantisation-quality benchmark (torch fp32 vs onnx fp32 vs onnx int8).

Lean on purpose (numpy + stdlib + the serving stack; no pandas / scipy / sklearn) so it runs in
`.venv-serve`. Writes numbers and ids only; dataset text is read in memory and never saved.

    python -m intent_router.serve_bench --stage latency --serve-dir serve_model --split val \
        --out results/serving/latency.json
    python -m intent_router.serve_bench --stage quality --serve-dir serve_model --split val \
        --out results/serving/quality_val.json

Stages
  latency  CPU batch-1 predict() latency (tokenise + forward + post-process): p50 / p95 ms for
           every backend x thread count (default 1 and 8), warm-up 10 calls, all texts, 3 repeats;
           plus size on disk per variant, library versions, CPU model and os.cpu_count().
  idle     60 s of 1 Hz system-CPU samples (psutil); exit 0 iff the machine is idle (see
           idle_summary); used to gate the latency run.
  quality  per backend on one split: macro-F1 (all K labels, absent = 0), accuracy, prediction
           agreement with torch fp32, OOD-score Spearman vs torch fp32, abstention rate and
           abstention agreement (the fp32 val-derived threshold from router_config.json applied
           to every backend's scores).

TEST SPLIT: only through the logged mechanism. `--split test` needs --allow-test and --test-log
(results/<dir>/test_eval_log.jsonl). One line per variant is appended BEFORE that variant sees a
test row: call_type `quantization_inference` (quality) or `latency_inference` (latency). A second
quality run for the same (fingerprint, variant) is refused unless --allow-repeat-test.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from intent_router.predict import BACKENDS, ONNX_FILES, Router, softmax

IDLE_CPU_PCT = 15.0  # a 1 s system-CPU sample below this counts as idle
IDLE_WINDOW_S = 60
IDLE_MIN_FRACTION = 0.95  # "sustained": mean < IDLE_CPU_PCT and >= 95% of samples below it
CALL_TYPE_QUALITY = "quantization_inference"
CALL_TYPE_LATENCY = "latency_inference"
WARMUP = 10
REPEATS = 3
THREAD_COUNTS = (1, 8)
QUALITY_BATCH = 16


# --------------------------------------------------------------------------- data (stdlib)
def load_split(
    data_path: str, splits_path: str, split: str
) -> tuple[list[str], list[str], list[str]]:
    """(ids, texts, labels) of one split, sorted by id; csv module only (no pandas)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    with open(splits_path, newline="", encoding="utf-8") as fh:
        in_split = {r["id"] for r in csv.DictReader(fh) if r["split"] == split}
    with open(data_path, newline="", encoding="utf-8") as fh:
        rows = sorted((r for r in csv.DictReader(fh) if r["id"] in in_split), key=lambda r: r["id"])
    if len(rows) != len(in_split):
        raise ValueError(f"{data_path} does not cover all {split} ids in {splits_path}")
    return [r["id"] for r in rows], [r["text"] for r in rows], [r["label"] for r in rows]


# ------------------------------------------------------------------------------ numerics
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default method), q in [0, 100]."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_stats(ms: list[float]) -> dict[str, float | int]:
    """p50 / p95 / mean / min / max in milliseconds over all timed calls."""
    return {
        "n_calls": len(ms),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "mean_ms": float(np.mean(ms)),
        "min_ms": float(np.min(ms)),
        "max_ms": float(np.max(ms)),
    }


def rank_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (== scipy.stats.rankdata 'average')."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    sx = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    for end in range(1, len(x) + 1):
        if end == len(x) or sx[end] != sx[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2.0  # mean of ranks start+1..end
            start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation (Pearson of average ranks); None if either side is constant."""
    ra, rb = rank_average(a), rank_average(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return None if denom == 0.0 else float((ra * rb).sum() / denom)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over all n_classes labels, absent/undefined classes score 0 (stats.macro_f1)."""
    f1 = np.zeros(n_classes)
    for k in range(n_classes):
        tp = float(np.sum((y_true == k) & (y_pred == k)))
        denom = float(np.sum(y_true == k) + np.sum(y_pred == k))
        f1[k] = 2.0 * tp / denom if denom > 0 else 0.0
    return float(f1.mean())


def backend_quality(
    gold: np.ndarray,
    logits: np.ndarray,
    scores: np.ndarray,
    cfg_temperature: float,
    threshold: float,
    ref: dict[str, np.ndarray] | None,
    ids: list[str],
) -> dict[str, Any]:
    """Metrics for one backend; `ref` (the torch fp32 arrays) adds the agreement fields."""
    pred = logits.argmax(axis=1)
    probs = softmax(logits, cfg_temperature)
    abstained = scores < threshold
    out: dict[str, Any] = {
        "macro_f1": macro_f1(gold, pred, int(logits.shape[1])),
        "accuracy": float(np.mean(gold == pred)),
        "abstention_rate": float(np.mean(abstained)),
        "n_abstained": int(abstained.sum()),
    }
    if ref is not None:
        r_pred = ref["logits"].argmax(axis=1)
        r_abs = ref["scores"] < threshold
        r_probs = softmax(ref["logits"], cfg_temperature)
        out.update(
            {
                "pred_agreement_vs_torch": float(np.mean(pred == r_pred)),
                "ood_spearman_vs_torch": spearman(scores, ref["scores"]),
                "abstention_agreement_vs_torch": float(np.mean(abstained == r_abs)),
                "max_abs_diff_prob_vs_torch": float(np.abs(probs - r_probs).max()),
                "max_abs_diff_ood_vs_torch": float(np.abs(scores - ref["scores"]).max()),
                "pred_disagreement_ids": [i for i, d in zip(ids, pred != r_pred, strict=True) if d],
                "abstention_disagreement_ids": [
                    i for i, d in zip(ids, abstained != r_abs, strict=True) if d
                ],
            }
        )
    return out


# ------------------------------------------------------------------------- CPU load (psutil)
def idle_summary(samples: list[float], threshold: float = IDLE_CPU_PCT) -> dict[str, Any]:
    """Mean / max / fraction-below-threshold of 1 Hz system-CPU samples + the idle verdict."""
    if not samples:
        raise ValueError("no CPU samples")
    below = float(np.mean(np.asarray(samples) < threshold))
    mean = float(np.mean(samples))
    return {
        "n_samples": len(samples),
        "mean_pct": mean,
        "max_pct": float(np.max(samples)),
        "fraction_below_threshold": below,
        "threshold_pct": threshold,
        "idle": bool(mean < threshold and below >= IDLE_MIN_FRACTION),
        "rule": f"mean < {threshold}% and >= {IDLE_MIN_FRACTION:.0%} of 1 s samples < {threshold}%",
    }


def sample_cpu(seconds: int) -> list[float]:
    """System-wide CPU percent, one blocking 1 s sample per second (psutil)."""
    import psutil

    psutil.cpu_percent(interval=None)  # prime
    return [float(psutil.cpu_percent(interval=1.0)) for _ in range(seconds)]


class LoadSampler:
    """Background 1 Hz sampler of system CPU % and this process's share of the machine."""

    def __init__(self) -> None:
        import psutil

        self._proc = psutil.Process()
        self._ncpu = psutil.cpu_count(logical=True) or 1
        self.system: list[float] = []
        self.own: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.system.append(float(psutil.cpu_percent(interval=1.0)))
            self.own.append(float(self._proc.cpu_percent(interval=None)) / self._ncpu)

    def __enter__(self) -> LoadSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        """System CPU stats during the benchmark; `own_*` = this process (of the whole machine)."""
        n = min(len(self.system), len(self.own))
        if n == 0:
            return {"n_samples": 0}
        sys_a, own_a = np.asarray(self.system[:n]), np.asarray(self.own[:n])
        other = np.maximum(sys_a - own_a, 0.0)
        return {
            "n_samples": n,
            "system_mean_pct": float(sys_a.mean()),
            "system_max_pct": float(sys_a.max()),
            "own_process_mean_pct_of_machine": float(own_a.mean()),
            "other_processes_mean_pct_approx": float(other.mean()),
            "other_processes_max_pct_approx": float(other.max()),
        }


# ------------------------------------------------------------------- environment + logging
def cpu_model() -> str:
    """Best-effort CPU model string (PROCESSOR_IDENTIFIER on Windows, /proc/cpuinfo on Linux)."""
    if os.environ.get("PROCESSOR_IDENTIFIER"):
        return os.environ["PROCESSOR_IDENTIFIER"]
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> dict[str, Any]:
    """Library versions, CPU model, logical cores, OS."""
    import onnxruntime
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "torch_default_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def read_log(path: Path) -> list[dict[str, Any]]:
    """Parse the jsonl test log (missing file => empty)."""
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def log_test_call(
    path: Path, call_type: str, fingerprint: str, variant: str, n_rows: int, stage: str
) -> None:
    """Append one line to the shared append-only test log (same fields as the other modules)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "call_type": call_type,
        "model_fingerprint": fingerprint,
        "variant": variant,
        "n_rows": n_rows,
        "stage": stage,
        "smoke": False,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


class TestUse:
    """Gate for every call that reads test rows; a no-op for the val/train splits."""

    __test__ = False  # not a pytest class despite the Test* name

    def __init__(self, split: str, allow: bool, log_path: Path | None, repeat_ok: bool) -> None:
        self.on = split == "test"
        if self.on and not (allow and log_path):
            raise SystemExit("--split test needs --allow-test and --test-log (logged path only)")
        self.log_path = log_path
        self.repeat_ok = repeat_ok

    def before(self, call_type: str, fp: str, variant: str, n: int, stage: str) -> None:
        """Refuse a repeated quality run, then log the use BEFORE any test row is read."""
        if not self.on:
            return
        assert self.log_path is not None
        if call_type == CALL_TYPE_QUALITY and not self.repeat_ok:
            for e in read_log(self.log_path):
                if (e.get("call_type"), e.get("model_fingerprint"), e.get("variant")) == (
                    call_type,
                    fp,
                    variant,
                ):
                    raise SystemExit(
                        f"test quality run already logged for {fp[:12]} {variant} at "
                        f"{e['timestamp']}; refusing (--allow-repeat-test overrides)"
                    )
        log_test_call(self.log_path, call_type, fp, variant, n, stage)


# ------------------------------------------------------------------------------ stages
def variant_sizes(serve_dir: Path) -> dict[str, int | None]:
    """Bytes on disk per variant (None when the file is absent)."""
    files = {"torch_safetensors": "model.safetensors", **ONNX_FILES}
    return {k: (serve_dir / v).stat().st_size if (serve_dir / v).exists() else None
            for k, v in files.items()}  # fmt: skip


def _time_batch1(router: Router, texts: list[str], warmup: int, repeats: int) -> list[float]:
    """Per-call predict() wall time in ms: `warmup` untimed calls, then repeats x all texts."""
    for i in range(warmup):
        router.predict(texts[i % len(texts)])
    out: list[float] = []
    for _ in range(repeats):
        for t in texts:
            t0 = time.perf_counter()
            router.predict(t)
            out.append((time.perf_counter() - t0) * 1000.0)
    return out


def stage_latency(
    serve_dir: Path,
    texts: list[str],
    backends: list[str],
    threads: list[int],
    warmup: int,
    repeats: int,
    gate: TestUse,
    fingerprint: str,
) -> dict[str, Any]:
    """Batch-1 CPU latency for every backend x thread count."""
    res: dict[str, Any] = {}
    for be in backends:
        gate.before(CALL_TYPE_LATENCY, fingerprint, be, len(texts), "serve_latency")
        res[be] = {}
        for n in threads:
            router = Router.load(serve_dir, be, n)
            ms = _time_batch1(router, texts, warmup, repeats)
            res[be][f"threads_{n}"] = {"threads": n, **latency_stats(ms)}
            print(
                f"[latency] {be} threads={n} p50={res[be][f'threads_{n}']['p50_ms']:.1f}ms "
                f"p95={res[be][f'threads_{n}']['p95_ms']:.1f}ms",
                flush=True,
            )
            del router
            gc.collect()
    return res


def run_backend(router: Router, texts: list[str], batch: int) -> dict[str, np.ndarray]:
    """logits + OOD scores of all texts in order, `batch` texts per forward."""
    lg: list[np.ndarray] = []
    ft: list[np.ndarray] = []
    for s in range(0, len(texts), batch):
        a, b = router.infer(texts[s : s + batch])
        lg.append(a)
        ft.append(b)
    logits = np.concatenate(lg)
    return {"logits": logits, "scores": router.ood_scores(np.concatenate(ft))}


def stage_quality(
    serve_dir: Path,
    ids: list[str],
    texts: list[str],
    labels_gold: list[str],
    backends: list[str],
    threads: int | None,
    gate: TestUse,
    fingerprint: str,
) -> dict[str, Any]:
    """Per-backend accuracy / macro-F1 / agreement with torch / OOD Spearman / abstention."""
    if "torch" not in backends:
        raise SystemExit("quality needs the torch backend as the reference")
    res: dict[str, Any] = {}
    ref: dict[str, np.ndarray] | None = None
    gold: np.ndarray | None = None
    for be in ["torch", *[b for b in backends if b != "torch"]]:
        router = Router.load(serve_dir, be, threads)
        if gold is None:
            gold = np.array([router.cfg.labels.index(x) for x in labels_gold])
        gate.before(CALL_TYPE_QUALITY, fingerprint, be, len(texts), "serve_quality")
        arrays = run_backend(router, texts, QUALITY_BATCH)
        if be == "torch":
            ref = arrays
        res[be] = backend_quality(
            gold,
            arrays["logits"],
            arrays["scores"],
            router.cfg.temperature,
            router.cfg.ood_threshold,
            ref,
            ids,
        )
        print(f"[quality] {be}: acc={res[be]['accuracy']:.4f} f1={res[be]['macro_f1']:.4f}")
        del router
        gc.collect()
    return res


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("latency", "quality", "idle"), required=True)
    ap.add_argument("--serve-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="val")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--out", type=Path, default=None, help="results JSON (numbers + ids only)")
    ap.add_argument("--require-idle", action="store_true", help="latency: 60 s idle check first")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    ap.add_argument("--threads", nargs="+", type=int, default=list(THREAD_COUNTS))
    ap.add_argument("--quality-threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--contended", action="store_true", help="CPU shared with other jobs: label")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--allow-repeat-test", action="store_true")
    ap.add_argument("--test-log", type=Path, default=None)
    a = ap.parse_args(argv)

    if a.stage == "idle":
        summ = idle_summary(sample_cpu(IDLE_WINDOW_S))
        print(json.dumps(summ))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        raise SystemExit(0 if summ["idle"] else 1)
    if a.out is None:
        raise SystemExit("--out is required for the latency / quality stages")
    idle_check: dict[str, Any] | None = None
    if a.stage == "latency" and a.require_idle:
        idle_check = idle_summary(sample_cpu(IDLE_WINDOW_S))
        print(f"[idle-check] {json.dumps(idle_check)}", flush=True)
        if not idle_check["idle"] and not a.contended:
            raise SystemExit("machine not idle: refusing to publish latency (use --contended)")
    cfg = json.loads((a.serve_dir / "router_config.json").read_text(encoding="utf-8"))
    fp = cfg["model_fingerprint"]
    gate = TestUse(a.split, a.allow_test, a.test_log, a.allow_repeat_test)
    ids, texts, gold_labels = load_split(a.data, a.splits, a.split)
    result: dict[str, Any] = {
        "stage": a.stage,
        "contended": a.contended,
        "split": a.split,
        "n_rows": len(ids),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "model_fingerprint": fp,
        "model_version": cfg["version"],
        "ood_threshold": cfg["ood_threshold"],
        "temperature": cfg["temperature"],
        "environment": environment(),
        "sizes_bytes": variant_sizes(a.serve_dir),
        "settings": {"warmup": a.warmup, "repeats": a.repeats, "threads": a.threads,
                     "quality_batch": QUALITY_BATCH, "backends": a.backends},
    }  # fmt: skip
    if a.contended:
        result["note"] = "measured while the CPU was shared with a training job: NOT final"
    if a.stage == "latency":
        result["idle_check_before"] = idle_check
        with LoadSampler() as sampler:
            result["latency"] = stage_latency(
                a.serve_dir, texts, a.backends, a.threads, a.warmup, a.repeats, gate, fp
            )
        result["load_during_benchmark"] = sampler.summary()
    else:
        result["quality"] = stage_quality(
            a.serve_dir, ids, texts, gold_labels, a.backends, a.quality_threads, gate, fp
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[serve_bench] wrote {a.out}")


if __name__ == "__main__":
    main()
