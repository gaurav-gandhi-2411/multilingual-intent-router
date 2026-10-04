"""Re-train the FINAL v1 model with live W&B logging into the submission project.

Why: the shipped v1 run was logged to the old 662-run project, and the saved curves there have no
per-step loss / LR. This script repeats ONLY the training stage of `intent_router.final` with the
identical config (configs/final.yaml `train:` block, i.e. results/final/model_version.json
train_config) and logs per-step loss + LR, per-epoch train/val loss, val macro-F1 and accuracy
live to W&B project `multilingual-intent-router` (group `final`, run id `final-v1-train`).

Guards:
  * No test-split inference of any kind: it calls `train_model` directly (never `_run_one`, which
    owns the single guarded test evaluation) and never touches results/final/test_eval_log.jsonl.
  * Writes only to NEW locations (outputs/final_model_wandb_rerun, results_rerun/final_wandb);
    refuses to run if either is the shipped location.
  * The retrained weights must hash to the shipped fingerprint (models.state_dict_sha256 of the
    weights re-LOADED from the saved dir). Mismatch: exit non-zero, nothing is shipped.
  * Training runs with exclusive GPU access; > 15 min of waiting aborts (TimeoutError).

Usage:
    python scripts/wandb_final_rerun.py --dry-run   # offline, 1 epoch, fingerprint NOT asserted
    python scripts/wandb_final_rerun.py             # online, full run, fingerprint asserted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTITY = "gauravgandhi429-gaurav-gandhi"
PROJECT = "multilingual-intent-router"
GROUP = "final"
RUN_ID = "final-v1-train"
RUN_NAME = "final-v1"
SHIPPED_FINGERPRINT = "7ca22e16d2ace345d9dcf7737887e85fedf400a1dfc863678c2fc2663bf102bd"
CONFIG = Path("configs/final.yaml")
MODEL_VERSION = Path("results/final/model_version.json")
SPLITS = Path("splits/splits.csv")
MODEL_DIR = Path("outputs/final_model_wandb_rerun")
RESULTS_DIR = Path("results_rerun/final_wandb")
DRY_MODEL_DIR = Path("outputs/final_model_wandb_dryrun")
DRY_RESULTS_DIR = Path("results_rerun/final_wandb_dryrun")
PROTECTED = (Path("outputs/final_model"), Path("results/final"))
LOCK_TIMEOUT_S = 15 * 60  # GPU occupied by another project longer than this: stop and report


# ------------------------------------------------------------------ pure helpers
def sha256_file(path: Path) -> str:
    """sha256 hex digest of a file's bytes (the split hash logged in the run config)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_config_matches(yaml_train: dict[str, Any], shipped_train: dict[str, Any]) -> list[str]:
    """Keys whose value differs between configs/final.yaml `train:` and the shipped train_config.

    Empty list == the re-run uses exactly the shipped configuration.
    """
    keys = set(yaml_train) | set(shipped_train)
    return sorted(k for k in keys if yaml_train.get(k) != shipped_train.get(k))


def assert_safe_dirs(model_dir: Path, results_dir: Path) -> None:
    """Refuse to write into the shipped model or results directories."""
    for target in (model_dir, results_dir):
        for prot in PROTECTED:
            t, p = target.resolve(), prot.resolve()
            if t == p or p in t.parents or t in p.parents:
                raise ValueError(f"refusing to write to {target}: overlaps shipped path {prot}")


def fingerprint_check(got: str, expected: str = SHIPPED_FINGERPRINT) -> dict[str, Any]:
    """Result record logged to the run summary."""
    return {
        "fingerprint_expected": expected,
        "fingerprint_got": got,
        "fingerprint_match": got == expected,
    }


def library_versions() -> dict[str, str]:
    """Versions of the libraries that determine the numerics, plus python / OS."""
    import numpy
    import sklearn
    import torch
    import transformers
    import wandb

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "wandb": wandb.__version__,
        "os": platform.platform(),
    }


def build_run_meta(
    git_sha: str, split_sha: str, versions: dict[str, str], query_prefix: str
) -> dict[str, Any]:
    """Extra W&B config keys (merged with TrainConfig by train._wandb_init). No paths, no text."""
    return {
        "run_id": RUN_NAME,  # train._wandb_init uses this as the display name
        "role": "final-v1-train (live re-run of the shipped training stage)",
        "seed": 42,
        "query_prefix": query_prefix,
        "lr_schedule": "linear warmup (10% of 20-epoch horizon) then linear decay; stopped at 9",
        "stop_epoch": 9,
        "split_sha256": split_sha,
        "git_sha": git_sha,
        "library_versions": versions,
        "expected_fingerprint": SHIPPED_FINGERPRINT,
        "no_test_inference": True,
    }


# ------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    """Run the training stage with live logging; return the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="offline, 1 epoch, dry-run dirs")
    args = ap.parse_args(argv)

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # as seeding.seed_everything does
    os.environ["WANDB_ENTITY"] = ENTITY
    os.environ["WANDB_RUN_ID"] = RUN_ID
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_DIR"] = "outputs"
    if args.dry_run:
        os.environ["WANDB_MODE"] = "offline"
    model_dir = DRY_MODEL_DIR if args.dry_run else MODEL_DIR
    results_dir = DRY_RESULTS_DIR if args.dry_run else RESULTS_DIR
    assert_safe_dirs(model_dir, results_dir)

    import torch
    from transformers import AutoModelForSequenceClassification

    from intent_router import data as data_mod
    from intent_router import gpu_lock
    from intent_router.data import get_frame
    from intent_router.evaluate import git_sha
    from intent_router.final import save_final_model, write_json
    from intent_router.models import QUERY_PREFIX, state_dict_sha256
    from intent_router.train import TrainConfig, train_model

    cfg = yaml.safe_load(CONFIG.read_text())
    shipped = json.loads(MODEL_VERSION.read_text())["train_config"]
    tr = dict(cfg["train"])
    diff = train_config_matches(tr, shipped)
    if diff:
        raise SystemExit(f"configs/final.yaml train differs from shipped train_config: {diff}")
    prefix = tr.pop("query_prefix")
    if QUERY_PREFIX.get(tr["model_name"], "") != prefix:
        raise SystemExit("final.yaml query_prefix differs from models.QUERY_PREFIX")
    if args.dry_run:
        tr["stop_epoch"] = 1
    tcfg = TrainConfig.from_dict({**tr, "wandb_project": PROJECT, "wandb_group": GROUP})

    data_mod.load_data(cfg["data_path"])
    train_df = get_frame(["train"], cfg["data_path"], cfg["splits_path"])
    val_df = get_frame(["val"], cfg["data_path"], cfg["splits_path"])
    train_df = train_df.sort_values("id").reset_index(drop=True)
    val_df = val_df.sort_values("id").reset_index(drop=True)
    if (train_df["split"] != "train").any() or (val_df["split"] != "val").any():
        raise AssertionError("unexpected split in train/val frames")

    meta = build_run_meta(git_sha(), sha256_file(SPLITS), library_versions(), prefix)
    t0 = time.perf_counter()
    with gpu_lock.gpu_exclusive(
        int(cfg["gpu_lock_expected_s"]), "intent-router wandb final rerun",
        timeout_s=LOCK_TIMEOUT_S,
    ) as lock_info:  # fmt: skip
        res, model, tok = train_model(tcfg, train_df, val_df, meta)
        try:
            mem_fp = state_dict_sha256(model)
            save_final_model(model, tok, model_dir, tcfg)
        finally:
            model = None
            torch.cuda.empty_cache()
    train_s = time.perf_counter() - t0

    # Re-load the SAVED weights (fp32, as the shipped fingerprint was taken) and hash them.
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), dtype=torch.float32
    )
    saved_fp = state_dict_sha256(reloaded)
    del reloaded
    check = fingerprint_check(saved_fp)
    check["fingerprint_in_memory"] = mem_fp
    check["fingerprint_asserted"] = not args.dry_run
    last = res.epochs[-1]
    summary = {
        **check,
        "epochs_run": len(res.epochs),
        "wall_clock_s": res.wall_clock_s,
        "peak_vram_mb": res.peak_vram_mb,
        "gpu_exclusive": res.gpu_exclusive,
        "gpu_lock_waited_s": lock_info["waited_s"],
        "final_val_macro_f1": last["macro_f1"],
        "final_val_accuracy": last["accuracy"],
        "nan_detected": res.nan_detected,
        "wandb_url": res.wandb_url,
        "epoch_history": res.epochs,
        "train_stage_total_s": train_s,
    }
    write_json(results_dir / "rerun_summary.json", summary)

    # The run was finished inside train_model: resume the same id to attach the check result.
    import wandb

    run = wandb.init(project=PROJECT, entity=ENTITY, id=RUN_ID, resume="allow", group=GROUP)
    run.summary.update(
        {
            **check,
            "val/macro_f1_final_epoch": last["macro_f1"],
            "val/accuracy_final_epoch": last["accuracy"],
        }
    )
    run.finish()

    print(json.dumps({k: v for k, v in summary.items() if k != "epoch_history"}, indent=1))
    if not args.dry_run and not check["fingerprint_match"]:
        print("FINGERPRINT MISMATCH: nothing shipped; shipped model stays outputs/final_model")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
