#!/usr/bin/env bash
# Robustness and open-set training fixes stages in pre-registered order, sequential from one process tree
# so GPU jobs never overlap. Usage: scripts/run_phase4c.sh <stage> [<stage> ...]
# Stops at the first failing stage so later decisions never run on partial inputs.
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
for stage in "$@"; do
  echo "$(date '+%F %T') stage=$stage start"
  if ! .venv/Scripts/python.exe -u -m intent_router.phase4c --config configs/phase4c.yaml \
      --stage "$stage"; then
    echo "$(date '+%F %T') stage=$stage FAILED"; exit 1
  fi
  echo "$(date '+%F %T') stage=$stage OK"
done
echo "$(date '+%F %T') PHASE4C_STAGES_DONE"
