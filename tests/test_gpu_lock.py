from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from intent_router import gpu_lock
from intent_router.aggregate import summarize_config
from intent_router.cv import extrapolate


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    return tmp_path / ".gpu.lock"


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    p.wait()
    return p.pid


def test_acquire_release_roundtrip(lock: Path) -> None:
    content = gpu_lock.try_acquire(lock, 120, "unit test")
    assert content is not None
    on_disk = json.loads(lock.read_text())
    assert on_disk["project"] == "intent-router"
    assert on_disk["pid"] == os.getpid()
    assert on_disk["expected_duration_s"] == 120
    assert datetime.fromisoformat(on_disk["start_time"]).tzinfo is not None
    assert gpu_lock.release(lock) is True
    assert not lock.exists()


def test_second_acquirer_blocks(lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", lambda: [])
    assert gpu_lock.try_acquire(lock, 10, "first") is not None
    with (
        pytest.raises(TimeoutError),
        gpu_lock.gpu_exclusive(10, "second", poll_s=0.05, timeout_s=0.3, path=lock),
    ):
        pytest.fail("must not enter while lock is held")
    assert json.loads(lock.read_text())["command"] == "first"  # still the first holder's


def test_context_manager_releases(lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", lambda: [])
    with gpu_lock.gpu_exclusive(10, "ctx", poll_s=0.05, path=lock) as info:
        assert lock.exists()
        assert info["pid"] == os.getpid()
    assert not lock.exists()


def test_waits_for_foreign_then_runs(lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(
        [[{"pid": 1, "process_name": "x", "used_memory_mb": None}], [], []]
    )  # wait, pre-clear, post-clear
    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", lambda: next(answers))
    with gpu_lock.gpu_exclusive(10, "ctx", poll_s=0.01, timeout_s=5, path=lock) as info:
        assert info["waited_s"] >= 0.0
    assert not lock.exists()


def test_foreign_forever_times_out_and_releases(
    lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", lambda: [{"pid": 9}])
    with (
        pytest.raises(TimeoutError),
        gpu_lock.gpu_exclusive(10, "ctx", poll_s=0.02, timeout_s=0.1, path=lock),
    ):
        pytest.fail("must not run with foreign GPU processes")
    assert not lock.exists()


def test_stale_lock_is_reclaimed(lock: Path) -> None:
    stale = {
        "project": "other",
        "pid": _dead_pid(),
        "host": gpu_lock.socket.gethostname(),
        "start_time": "2026-01-01T00:00:00+00:00",
        "expected_duration_s": 1,
        "command": "dead",
    }
    lock.write_text(json.dumps(stale))
    content = gpu_lock.try_acquire(lock, 10, "me")
    assert content is not None
    assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_unparseable_lock_is_not_reclaimed(lock: Path) -> None:
    lock.write_text("{not json")
    assert gpu_lock.try_acquire(lock, 10, "me") is None
    assert lock.read_text() == "{not json"
    lock.write_text(json.dumps({"project": "x"}))  # parseable but no pid: also fail closed
    assert gpu_lock.try_acquire(lock, 10, "me") is None
    assert lock.exists()


def test_release_never_deletes_foreign_lock(lock: Path) -> None:
    other = {"project": "other", "pid": os.getpid() + 1, "host": "h"}
    lock.write_text(json.dumps(other))
    assert gpu_lock.release(lock) is False
    assert json.loads(lock.read_text()) == other
    assert gpu_lock.release(lock.with_name("missing.lock")) is False


def test_waits_for_foreign_before_acquiring(lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def foreign() -> list[dict[str, Any]]:
        seen.append(lock.exists())  # lock state at each foreign check
        return [{"pid": 1}] if len(seen) < 3 else []

    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", foreign)
    with gpu_lock.gpu_exclusive(10, "ctx", poll_s=0.01, timeout_s=5, path=lock):
        pass
    # first two polls happen with the lock NOT held; third (pre-acquire) clear, then post-check
    assert seen[:3] == [False, False, False]
    assert not lock.exists()


def test_releases_lock_when_foreign_appears_after_acquire(
    lock: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter([[], [{"pid": 7}], [], []])  # pre clear, post foreign, pre clear, post clear
    held_during_backoff: list[bool] = []
    monkeypatch.setattr(gpu_lock, "foreign_gpu_processes", lambda: next(answers))
    monkeypatch.setattr(gpu_lock.time, "sleep", lambda s: held_during_backoff.append(lock.exists()))
    with gpu_lock.gpu_exclusive(10, "ctx", poll_s=0.01, timeout_s=5, path=lock):
        assert lock.exists()
    assert held_during_backoff == [False]  # lock was released while backing off
    assert not lock.exists()


# ------------------------------------------------------------------ nvidia-smi
def test_parse_compute_apps_na_and_numbers() -> None:
    apps = gpu_lock.parse_compute_apps(
        "1234, C:\\Python\\python.exe, [N/A]\n5678, C:\\x\\ollama.exe, 2048\n\n"
    )
    assert apps == [
        {"pid": 1234, "process_name": "C:\\Python\\python.exe", "used_memory_mb": None},
        {"pid": 5678, "process_name": "C:\\x\\ollama.exe", "used_memory_mb": 2048.0},
    ]
    assert gpu_lock.parse_compute_apps("") == []
    with pytest.raises(gpu_lock.GpuQueryError):
        gpu_lock.parse_compute_apps("No devices were found")


def test_select_foreign_rules() -> None:
    apps: list[dict[str, Any]] = [
        {"pid": 1, "process_name": "me", "used_memory_mb": None},
        {"pid": 2, "process_name": "small", "used_memory_mb": 100.0},
        {"pid": 3, "process_name": "big", "used_memory_mb": 501.0},
        {"pid": 4, "process_name": "na", "used_memory_mb": None},
        {"pid": 5, "process_name": "edge", "used_memory_mb": 500.0},
    ]
    assert [a["pid"] for a in gpu_lock.select_foreign(apps, {1})] == [3, 4]


def test_foreign_excludes_own_and_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    out = f"{os.getpid()}, py, [N/A]\n{os.getppid()}, launcher, [N/A]\n"
    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", lambda args: out)
    assert gpu_lock.foreign_gpu_processes() == []


def test_foreign_clear_when_no_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", lambda args: "\n")
    assert gpu_lock.foreign_gpu_processes() == []


def test_foreign_detected_na(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", lambda args: "999999, ollama.exe, [N/A]\n")
    got = gpu_lock.foreign_gpu_processes()
    assert [g["pid"] for g in got] == [999999]


def test_nvidia_smi_failure_is_not_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(args: list[str]) -> str:
        raise gpu_lock.GpuQueryError("no nvidia-smi")

    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", boom)
    got = gpu_lock.foreign_gpu_processes()
    assert got and got[0]["process_name"] == "nvidia-smi-unavailable"
    snap = gpu_lock.gpu_snapshot()
    assert snap["foreign"] and snap["error"] and snap["memory"] is None


def test_unparseable_output_is_not_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", lambda args: "garbage\n")
    assert gpu_lock.foreign_gpu_processes()


def test_snapshot_memory_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(args: list[str]) -> str:
        return (
            ""
            if "--query-compute-apps=pid,process_name,used_memory" in args
            else ("1024 MiB, 8192 MiB, 3 %\n")
        )

    monkeypatch.setattr(gpu_lock, "_run_nvidia_smi", fake)
    snap = gpu_lock.gpu_snapshot()
    assert snap["memory"] == "1024 MiB, 8192 MiB, 3 %"
    assert snap["foreign"] == [] and snap["error"] is None


# ------------------------------------------------------- aggregate / extrapolate
def _run(excl: bool | None, wall: float, vram: float) -> dict[str, Any]:
    r: dict[str, Any] = {
        "fold_seed_idx": 0,
        "fold": 0,
        "epochs": [{"epoch": 1, "macro_f1": 0.5, "accuracy": 0.6}],
        "wall_clock_s": wall,
        "peak_vram_mb": vram,
    }
    if excl is not None:
        r["gpu_exclusive"] = excl
    return r


def test_summarize_excludes_non_exclusive_from_cost_only() -> None:
    s = summarize_config([_run(True, 10, 100), _run(False, 1000, 9000), _run(None, 500, 8000)])
    assert (s["n_exclusive"], s["n_total"], s["any_non_exclusive"]) == (1, 3, True)
    assert s["mean_wall_clock_s"] == 10.0 and s["max_peak_vram_mb"] == 100.0
    assert s["macro_f1_mean"] == 0.5  # metrics still use all runs


def test_summarize_no_exclusive_runs() -> None:
    s = summarize_config([_run(None, 10, 100)])
    assert s["mean_wall_clock_s"] is None and s["n_exclusive"] == 0


def test_extrapolate_arithmetic() -> None:
    cfg = {
        "epochs": 20,
        "n_folds": 5,
        "lr_grid": [1, 2, 3],
        "fold_seed_idx": [0, 1, 2],
        "model_seeds": [0, 1, 2],
    }
    t = {"m": {"model_load_s": 10.0, "train_epoch_s": 3.0, "eval_s": 1.0}}
    ex = extrapolate(t, cfg)
    assert ex["label"] == "estimate, extrapolated from 1-epoch probe"
    assert ex["per_model"]["m"]["per_run_s"] == 90.0
    assert (ex["per_model"]["m"]["sweep_runs"], ex["per_model"]["m"]["confirm_runs"]) == (15, 40)
    assert ex["per_stage_h"]["sweep"] == pytest.approx(90 * 15 / 3600)
    assert ex["total_h"] == pytest.approx(90 * 55 / 3600)


def test_default_lock_path_is_the_projects_dir_beside_the_repo() -> None:
    default = Path(gpu_lock.DEFAULT_LOCK_PATH)
    repo = Path(gpu_lock.__file__).resolve().parents[2]
    assert default.name == ".gpu.lock" and default.parent == repo.parent
