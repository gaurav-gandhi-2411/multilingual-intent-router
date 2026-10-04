"""Execute notebooks/intent_router_colab.ipynb locally, cell by cell, to catch errors before Colab.

Only the Colab-specific parts of the code cells are replaced (see `localize`):
  * setup: `git clone` -> the current repo directory is used (no clone, no network);
  * install: no `pip install` (the venv is never modified; everything else in the cell runs);
  * data: `files.upload()` -> `data/dataset.csv` already in the working copy is used as the upload;
  * `/content/colab_cfg` -> `results_colab/_colab_cfg` (inside the gitignored output dir);
  * `/content/.gpu.lock` -> the local lock-file path used by the GPU guard;
  * Colab secrets are absent (`google.colab` is not importable, the secrets cell handles that).
All outputs go to `results_colab/` (gitignored); the committed `results/` is never written. Cell
stdout is mirrored to `<out dir>/run_notebook_local.log` (class names and metrics only).

Before the first GPU cell the runner waits (up to --gpu-wait-min, default 15) for the GPU to be free
of foreign processes (`intent_router.gpu_lock`), then stops with exit code 3.

    PYTHONPATH=src python scripts/run_notebook_local.py [--gpu-wait-min N] [--skip-gpu-check]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "intent_router_colab.ipynb"
DEFAULT_GPU_LOCK = str(ROOT.parent / ".gpu.lock")  # shared lock file beside the repositories
UPLOAD_BLOCK = (
    "    from google.colab import files\n\n"
    "    uploaded = files.upload()\n"
    "    if len(uploaded) != 1:\n"
    '        raise RuntimeError(f"upload exactly one file, got {len(uploaded)}")\n'
    "    dest.write_bytes(next(iter(uploaded.values())))\n"
)
UPLOAD_LOCAL = (
    "    pass  # local run: data/dataset.csv is already in place (stands in for the upload)\n"
)
PIP_INSTALL = (
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)'
)
PIP_SKIPPED = 'print("local run: pip install skipped; using the current environment")'
PIP_UNINSTALL = (
    'subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio", '
    '"peft"], check=False)'
)
CLONE_DIR = 'REPO_DIR = WORKDIR / Path(PUBLIC_REPO_URL.removesuffix(".git")).name'
PYTHONPATH_ENV = '"PYTHONPATH": str(REPO_DIR / "src"),'
PYTHONPATH_LOCAL = (
    '"PYTHONPATH": os.pathsep.join([str(REPO_DIR / "src"), '
    'os.environ.get("RUNNER_EXTRA_PATH", "")]),'
)
LATENCY_PIP_MARK = "SERVE_PINS"


def code_cells(nb: dict[str, Any]) -> list[tuple[int, str]]:
    """[(cell index in the notebook, source)] for code cells, in order."""
    return [
        (i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
    ]


def localize(src: str, root: Path = ROOT, out_name: str = "results_colab") -> str:
    """Swap ONLY the Colab-specific parts of a code cell for their local equivalents.

    `out_name` is the output root; the default is the real one. `--cpu-dry-run` uses a scratch root.
    """
    cfg_dir = (root / out_name / "_colab_cfg").as_posix()
    out = src.replace("/content/colab_cfg", cfg_dir)
    if out_name != "results_colab":
        out = out.replace('OUT_ROOT = "results_colab"', f'OUT_ROOT = "{out_name}"')
    out = out.replace('"/content/.gpu.lock"', repr(DEFAULT_GPU_LOCK))
    out = out.replace(CLONE_DIR, f"REPO_DIR = Path({root.as_posix()!r})")
    out = out.replace(UPLOAD_BLOCK, UPLOAD_LOCAL)
    out = out.replace(PIP_INSTALL, PIP_SKIPPED)
    out = out.replace(PIP_UNINSTALL, PIP_SKIPPED)  # never uninstall from the local venv
    # child processes keep the extra (onnx) site dir the runner was given, if any
    out = out.replace(PYTHONPATH_ENV, PYTHONPATH_LOCAL)
    if LATENCY_PIP_MARK in out:  # the serve pins would pip-install into the venv: skip, keep prints
        out = out.replace(
            'subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)',
            PIP_SKIPPED,
        )
    return out


def wait_for_gpu(max_wait_min: float, poll_s: float = 30.0) -> bool:
    """True when no foreign process holds the GPU within the budget (intent_router.gpu_lock)."""
    sys.path.insert(0, str(ROOT / "src"))
    from intent_router import gpu_lock

    deadline = time.monotonic() + max_wait_min * 60
    while True:
        foreign = gpu_lock.foreign_gpu_processes()
        if not foreign:
            return True
        print(f"[runner] GPU busy: {foreign}", flush=True)
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


class _Tee(io.TextIOBase):
    """Write to the real stdout and a log file at once."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def run_cells(
    nb: dict[str, Any], log: Any, out_name: str = "results_colab"
) -> list[dict[str, Any]]:
    """Execute the localized code cells sequentially in one namespace; one record per cell."""
    ns: dict[str, Any] = {"__name__": "__main__"}
    records: list[dict[str, Any]] = []
    os.chdir(ROOT)
    for idx, src in code_cells(nb):
        t0 = time.monotonic()
        rec: dict[str, Any] = {"cell": idx, "status": "ok", "error": None}
        print(f"\n===== cell {idx} =====", file=log, flush=True)
        try:
            with contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
                exec(compile(localize(src, ROOT, out_name), f"nb-cell-{idx}", "exec"), ns)  # noqa: S102
        except Exception:  # noqa: BLE001 - record and stop: later cells depend on this one
            rec["status"] = "ERROR"
            rec["error"] = traceback.format_exc()
            print(rec["error"], file=log, flush=True)
            print(rec["error"], file=sys.__stdout__, flush=True)
        rec["seconds"] = round(time.monotonic() - t0, 1)
        records.append(rec)
        if rec["status"] == "ERROR":
            break
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-wait-min", type=float, default=15.0)
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument(
        "--cpu-dry-run",
        action="store_true",
        help="no GPU: seed COMMITTED track_a/headline JSON into results_colab_dryrun/ so the "
        "training cells skip, then run every other cell (checks the cells, not the numbers)",
    )
    ap.add_argument(
        "--extra-site",
        type=Path,
        default=None,
        help="a site-packages dir APPENDED to sys.path (after the venv's own packages, so nothing "
        "is shadowed): for the onnx libs the Colab latency cell would pip-install; nothing is "
        "installed",
    )
    args = ap.parse_args(argv)
    if args.extra_site is not None:
        sys.path.append(str(args.extra_site))
        child = ROOT / "results_colab_dryrun" / "_extra_site"
        child.mkdir(parents=True, exist_ok=True)
        (child / "sitecustomize.py").write_text(
            f"import sys\nsys.path.append({str(args.extra_site)!r})\n", encoding="utf-8"
        )
        os.environ["RUNNER_EXTRA_PATH"] = str(child)
    out_name = "results_colab_dryrun" if args.cpu_dry_run else "results_colab"
    os.environ["MPLBACKEND"] = "Agg"  # plt.show() must never block on a GUI window
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    out_dir = ROOT / out_name
    (out_dir / "_colab_cfg").mkdir(parents=True, exist_ok=True)
    if args.cpu_dry_run:  # committed numbers stand in for a fresh run: NOT a reproduction
        for rel in ("final/track_a.json", "trackb/headline.json"):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "results" / rel, out_dir / rel)
    elif not args.skip_gpu_check and not wait_for_gpu(args.gpu_wait_min):
        print(f"GPU still busy after {args.gpu_wait_min} min: stopping (exit 3)")
        return 3
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with (out_dir / "run_notebook_local.log").open("w", encoding="utf-8") as log:
        records = run_cells(nb, log, out_name)
    (out_dir / "run_notebook_local_summary.json").write_text(json.dumps(records, indent=1))
    for r in records:
        print(f"cell {r['cell']:3d}  {r['status']:5s}  {r['seconds']:8.1f} s")
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
