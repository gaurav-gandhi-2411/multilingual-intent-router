#!/usr/bin/env bash
# Robustness evaluation, then the local LLM label audit, run sequentially from one process tree so
# GPU jobs never overlap (see src/intent_router/gpu_lock.py). Optionally waits for a marker line in a log first.
# Usage: scripts/run_phase4.sh [log_to_wait_on] [marker]
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
if [ $# -ge 2 ]; then
  until grep -q "$2" "$1" 2>/dev/null; do sleep 60; done
fi
echo "$(date '+%F %T') robustness start"
.venv/Scripts/python.exe -u -m intent_router.robustness --config configs/robustness.yaml --stage all \
  || { echo "$(date '+%F %T') robustness FAILED"; exit 1; }
echo "$(date '+%F %T') robustness OK; llm_audit start"
.venv/Scripts/python.exe -u -m intent_router.llm_audit \
  || { echo "$(date '+%F %T') llm_audit FAILED"; exit 1; }
echo "$(date '+%F %T') PHASE4_DONE"
