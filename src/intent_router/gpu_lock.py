from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import subprocess
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = "intent-router"
# The lock file shared by all projects sits beside the repositories: <projects dir>/.gpu.lock.
DEFAULT_LOCK_PATH = str(Path(__file__).resolve().parents[3] / ".gpu.lock")
LOCK_ENV = "INTENT_ROUTER_GPU_LOCK"
# Per-process VRAM above this marks another process as "foreign" (the GPU is not ours alone).
FOREIGN_MEM_MB = 500
NVIDIA_SMI_TIMEOUT_S = 30


class GpuQueryError(RuntimeError):
    """nvidia-smi failed or printed something we cannot parse (callers fail closed)."""


def lock_path() -> Path:
    """Lock file path: env INTENT_ROUTER_GPU_LOCK, else the default."""
    return Path(os.environ.get(LOCK_ENV) or DEFAULT_LOCK_PATH)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[gpu {_now()}] {msg}", flush=True)


# --------------------------------------------------------------------- liveness
def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists (psutil if installed, else tasklist)."""
    if pid <= 0:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        pass
    out = subprocess.run(  # noqa: S603 - fixed argv, integer pid
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=False,
    ).stdout
    return any(tok == str(pid) for line in out.splitlines() for tok in line.split())


def own_pids() -> set[int]:
    """This process, its ancestors (venv python.exe launcher) and all descendants."""
    pids = {os.getpid(), os.getppid()}
    try:
        import psutil

        me = psutil.Process()
        pids |= {p.pid for p in me.parents()}
        pids |= {p.pid for p in me.children(recursive=True)}
    except Exception:  # noqa: BLE001 - psutil missing or process vanished: keep the basics
        pass
    return pids


# ------------------------------------------------------------------- lock file
def _holder_state(path: Path) -> str:
    """'free' | 'held' | 'stale' for an existing lock file (fail closed on anything odd)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "free"
    try:
        info = json.loads(raw)
        pid = info["pid"]
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise TypeError("pid is not an int")
    except (ValueError, KeyError, TypeError) as exc:
        _log(f"LOUD: lock {path} is unparseable ({exc!r}); treating as HELD, not removing: {raw!r}")
        return "held"
    if info.get("host") != socket.gethostname():
        _log(f"lock held by another host {info.get('host')!r}; cannot verify PID, treating as held")
        return "held"
    if pid_alive(pid):
        return "held"
    _log(f"WARNING: stale lock (PID {pid} not alive); removing. Content: {raw!r}")
    return "stale"


def try_acquire(path: Path, expected_duration_s: int, command: str) -> dict[str, Any] | None:
    """One atomic acquire attempt. Returns the lock content if acquired, None if held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "project": PROJECT,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "start_time": _now(),
        "expected_duration_s": int(expected_duration_s),
        "command": command[:200],
    }
    for _ in range(2):  # second pass only after removing a stale lock
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _holder_state(path) == "stale":
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                continue
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(content))
        return content
    return None


def release(path: Path) -> bool:
    """Remove the lock only if it still names our PID. True if removed."""
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return False
    if not isinstance(info, dict) or info.get("pid") != os.getpid():
        _log(f"not releasing {path}: owned by {info!r}")
        return False
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return True


# ------------------------------------------------------------------ nvidia-smi
def _run_nvidia_smi(args: list[str]) -> str:
    """Run nvidia-smi; raise GpuQueryError on any failure."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuQueryError(f"nvidia-smi could not run: {exc!r}") from exc
    if proc.returncode != 0:
        raise GpuQueryError(f"nvidia-smi exit {proc.returncode}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def parse_compute_apps(text: str) -> list[dict[str, Any]]:
    """Parse `pid,process_name,used_memory` csv rows; used_memory_mb None if '[N/A]'."""
    apps: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            raise GpuQueryError(f"unparseable nvidia-smi row: {line!r}")
        try:
            pid = int(parts[0])
        except ValueError as exc:
            raise GpuQueryError(f"unparseable nvidia-smi row: {line!r}") from exc
        name = ",".join(parts[1:-1])
        try:
            mem: float | None = float(parts[-1])
        except ValueError:
            mem = None  # "[N/A]" / "[Not Supported]" on Windows WDDM
        apps.append({"pid": pid, "process_name": name, "used_memory_mb": mem})
    return apps


def query_compute_apps() -> list[dict[str, Any]]:
    """All compute apps nvidia-smi reports (raises GpuQueryError on failure)."""
    return parse_compute_apps(
        _run_nvidia_smi(
            [
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )


def select_foreign(apps: list[dict[str, Any]], mine: set[int]) -> list[dict[str, Any]]:
    """Foreign = not ours and (>500 MB or memory unavailable; conservative)."""
    return [
        a
        for a in apps
        if a["pid"] not in mine
        and (a["used_memory_mb"] is None or a["used_memory_mb"] > FOREIGN_MEM_MB)
    ]


def foreign_gpu_processes() -> list[dict[str, Any]]:
    """Foreign GPU processes. nvidia-smi failure => one sentinel entry (never 'clear')."""
    try:
        return select_foreign(query_compute_apps(), own_pids())
    except GpuQueryError as exc:
        return [{"pid": None, "process_name": "nvidia-smi-unavailable", "error": str(exc)}]


def gpu_memory_line() -> str | None:
    """`memory.used,memory.total,utilization.gpu` line, None if nvidia-smi fails."""
    try:
        out = _run_nvidia_smi(
            [
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
    except GpuQueryError:
        return None
    lines = out.strip().splitlines()
    return lines[0].strip() if lines else None


def gpu_snapshot() -> dict[str, Any]:
    """Timestamped record of GPU occupancy; `foreign` non-empty => not exclusive."""
    error: str | None = None
    try:
        apps = query_compute_apps()
    except GpuQueryError as exc:
        apps, error = [], str(exc)
    foreign = foreign_gpu_processes()
    return {
        "time": _now(),
        "processes": apps,
        "foreign": foreign,
        "memory": gpu_memory_line(),
        "error": error,
    }


# ------------------------------------------------------------- context manager
@contextlib.contextmanager
def gpu_exclusive(
    expected_duration_s: int,
    command: str,
    poll_s: float = 60,
    timeout_s: float | None = None,
    path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Hold exclusive GPU access for the duration of the block.

    Waits (polling every poll_s) while the lock is held by someone else or while foreign
    processes occupy the GPU; never runs concurrently. Foreign processes are awaited BEFORE
    acquiring and re-checked after; if any appear, the lock is released and we back off
    poll_s + jitter in [0, poll_s) before retrying (avoids a lock/idle-context deadlock).
    Raises TimeoutError after timeout_s.
    """
    path = path or lock_path()
    t0 = time.monotonic()

    def waited() -> float:
        return time.monotonic() - t0

    def check_timeout(why: str) -> None:
        if timeout_s is not None and waited() >= timeout_s:
            raise TimeoutError(f"GPU not available after {waited():.0f}s: {why}")

    # Seeded from pid (not the global RNG) so two processes desynchronise their back-off.
    rng = random.Random(os.getpid())  # noqa: S311 - jitter, not security
    while True:
        # (a) Wait for the GPU to be free of foreign processes BEFORE taking the lock: an idle
        # lock-honouring process with a live CUDA context must not wait on us while we hold it.
        while True:
            foreign = foreign_gpu_processes()
            if not foreign:
                break
            why = f"foreign GPU processes {foreign}"
            _log(f"waiting (lock not held): {why}; next check in {poll_s}s")
            check_timeout(why)
            time.sleep(poll_s)
        content = try_acquire(path, expected_duration_s, command)
        if content is None:
            why = f"lock {path} held"
            _log(f"waiting: {why}; next check in {poll_s}s")
            check_timeout(why)
            time.sleep(poll_s)
            continue
        # (b) Re-check after acquiring; if a foreign process appeared, back off and retry.
        foreign = foreign_gpu_processes()
        if not foreign:
            break
        release(path)
        why = f"foreign GPU processes {foreign}"
        delay = poll_s + rng.uniform(0, poll_s)
        _log(f"released lock after acquire: {why}; backing off {delay:.1f}s then retrying")
        check_timeout(why)
        time.sleep(delay)
    try:
        yield {**content, "waited_s": round(waited(), 1)}
    finally:
        release(path)
