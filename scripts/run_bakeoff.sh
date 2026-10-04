#!/usr/bin/env bash
# Runs the bake-off stages sequentially. Every fold-run takes exclusive GPU access
# itself (see src/intent_router/gpu_lock.py), so this script does no GPU waiting of its own.
# Each stage is resumable (finished runs are skipped), so a stage that exits nonzero is retried
# once to pick up runs that failed transiently.
# Usage: scripts/run_bakeoff.sh sweep confirm   (stages run in the order given)
set -u
cd "$(dirname "$0")/.." || exit 2
PY=.venv/Scripts/python.exe
export PYTHONPATH=src
for stage in "$@"; do
  for attempt in 1 2; do
    echo "$(date '+%F %T') stage=$stage attempt=$attempt start"
    if "$PY" -m intent_router.cv --config configs/bakeoff.yaml --stage "$stage"; then
      echo "$(date '+%F %T') stage=$stage OK"
      "$PY" -m intent_router.aggregate --config configs/bakeoff.yaml || exit 3
      break
    fi
    [ "$attempt" = 2 ] && { echo "$(date '+%F %T') stage=$stage FAILED twice"; exit 1; }
  done
done
echo "$(date '+%F %T') ALL DONE"
