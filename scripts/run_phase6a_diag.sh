#!/usr/bin/env bash
# Open-set improvement round report-only diagnostics D1-D4, sequential from one
# process tree so GPU jobs never overlap (each training run takes exclusive GPU access).
# Default order d1 d3 d4 d2: the stages share one run store (outputs/phase6a_diag/runs), so v1's
# headline/DEV models trained for D1 are reused by D3 / D4 (fraction 1.0) / D2, and the longest
# stage (d2: 39 runs, of which 13 are already done by then) goes last. Every stage is resumable
# (a finished run / stage file is skipped), so after a crash simply re-run the same command.
# Stops at the first failing stage. D1 is an ORACLE (uses held-out labels): report-only, and
# nothing in the training/selection module may read results/phase6a/diag.
# Usage: scripts/run_phase6a_diag.sh [<stage> ...]    e.g.  scripts/run_phase6a_diag.sh d1 d3
#        SMOKE=1 scripts/run_phase6a_diag.sh          (tiny run, outputs/phase6a_diag_smoke only)
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
mkdir -p outputs/logs
log="outputs/logs/phase6a_diag_$(date '+%Y%m%d_%H%M%S').log"
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(d1 d3 d4 d2)
fi
extra=()
if [ "${SMOKE:-0}" = "1" ]; then
  extra=(--smoke)
fi
for stage in "${stages[@]}"; do
  echo "$(date '+%F %T') stage=$stage start" | tee -a "$log"
  .venv/Scripts/python.exe -u -m intent_router.phase6a_diag --config configs/phase6a_diag.yaml \
      --stage "$stage" ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "$log"
  # `tee` hides the python exit code; PIPESTATUS keeps the fail-fast contract.
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "$(date '+%F %T') stage=$stage FAILED" | tee -a "$log"; exit 1
  fi
  echo "$(date '+%F %T') stage=$stage OK" | tee -a "$log"
done
echo "$(date '+%F %T') PHASE6A_DIAG_STAGES_DONE" | tee -a "$log"
