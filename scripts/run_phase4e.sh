#!/usr/bin/env bash
# Multi-axis selection stages in pre-registered order, sequential from one process tree
# so GPU jobs never overlap. Curves for all candidates first, then epochs, folds, Track B, and
# finally the two read-only stages. Every stage is resumable (result-file existence), so after a
# crash simply re-run the same command. Stops at the first failing stage.
# Usage: scripts/run_phase4e.sh [<stage> ...]   (default: every stage in order)
#        scripts/run_phase4e.sh curves folds     (a subset, still in the order given)
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
mkdir -p outputs/logs
log="outputs/logs/phase4e_$(date '+%Y%m%d_%H%M%S').log"
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(reuse_check curves epoch folds trackb analyze select)
fi
for stage in "${stages[@]}"; do
  echo "$(date '+%F %T') stage=$stage start" | tee -a "$log"
  .venv/Scripts/python.exe -u -m intent_router.phase4e --config configs/phase4e.yaml \
      --stage "$stage" 2>&1 | tee -a "$log"
  # `tee` hides the python exit code; PIPESTATUS keeps the fail-fast contract.
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "$(date '+%F %T') stage=$stage FAILED" | tee -a "$log"; exit 1
  fi
  echo "$(date '+%F %T') stage=$stage OK" | tee -a "$log"
done
echo "$(date '+%F %T') PHASE4E_STAGES_DONE" | tee -a "$log"
