#!/usr/bin/env bash
# Open-set improvement round stages in pre-registered order, sequential from one process tree so
# GPU jobs never overlap (each training run also takes exclusive GPU access). Every stage is
# resumable (result-file existence), so after a crash simply re-run the same command. Stops at the
# first failing stage. Order: ref -> i4 -> i3 -> combos -> select -> confirm_report.
#
# Usage: scripts/run_phase6a.sh                  default: ref i4 i3 combos select  (NOT confirm_report)
#        scripts/run_phase6a.sh ref i4           a subset, in the order given
#        scripts/run_phase6a.sh confirm_report   only AFTER `select`; reads/writes test-row inference
#        SMOKE=1 scripts/run_phase6a.sh ref      tiny smoke (2 folds, 1 epoch, pool of 5) under
#                                                outputs/phase6a_smoke
# final_retrain is deliberately not in any default list: run it by hand
#   (scripts/run_phase6a.sh final_retrain) only when selection.json chose a non-ref candidate and
#   confirm_report has run.
#
# Per-stage wall-clock ESTIMATES (one RTX 3070, sequential, nothing else on the GPU). They are
# ESTIMATES, not measurements of a full run. Per-run costs are the 4e measurements on the real data
# (results/phase4e/v1/s*/curves: 107 s mean per 20-epoch curve = 5.3 s/epoch; LOCO training 37 s at
# 9 epochs) scaled by epochs and rows (soup members train on 85% of the rows: 4.5 s/epoch), plus the
# fixed overheads seen in the smoke run (model load ~4 s, member save + hash ~5 s, one LOCO extraction
# with the raw + neutral passes ~10 s). I4 training is scaled by 2.1: in the smoke run an I4 LOCO run
# trained in 10-11 s against 5 s for the plain recipe (4e measured A1 alone at 1.3x); re-measure on
# the first real I4 curve. e* is assumed ~8 for every candidate (unknown until run).
#   ref             ~0.7 h  (4e v1 curves reused: 0 curve runs; 15 fold models ~55 s; 15 LOCO ~60 s;
#                            ~30 view tables per scorer/threshold variant, 1-8 s each)
#   i4              ~3.5 h  (2 candidates x [15 curves ~225 s + 15 fold models ~100 s + 15 LOCO ~95 s])
#   i3              ~2.3 h  (35 member curves ~96 s; 35 CV members + 35 LOCO members ~46 s each;
#                            CV soups ~9 min; LOCO soup extraction ~13 min; views)
#   combos          ~3.9 h  (the same pool as i3 with the twin loss: training x2.1)
#   select          ~5 min  (CPU: 10,000-resample paired bootstraps over ~12 candidates)
#   confirm_report  0.3-1.3 h (ref 8 runs ~8 min; plus the candidate: plain ~8-13 min, an I3 soup
#                            ~35 min (6 units x 7 members), an I3+I4 soup ~70 min)
#   total before confirm_report ~10.4 h (ESTIMATE; the pre-registered 10-14 GPU-hours also covers confirm and
#   the Colab-side work). Fold models are never saved; soup members (1.1 GB each, 7 per unit) are
#   deleted after each unit, peak extra disk ~8 GB; DEV npz arrays ~0.15 GB per weight set. Every
#   stage prints a [disk] line (outputs / results size, drive free space) when it finishes.
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
mkdir -p outputs/logs
log="outputs/logs/phase6a_$(date '+%Y%m%d_%H%M%S').log"
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then
  stages=(ref i4 i3 combos select)
fi
extra=()
if [ "${SMOKE:-0}" = "1" ]; then
  extra+=(--smoke)
fi
for stage in "${stages[@]}"; do
  echo "$(date '+%F %T') stage=$stage start" | tee -a "$log"
  .venv/Scripts/python.exe -u -m intent_router.phase6a --config configs/phase6a.yaml \
      --stage "$stage" "${extra[@]}" 2>&1 | tee -a "$log"
  # `tee` hides the python exit code; PIPESTATUS keeps the fail-fast contract.
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "$(date '+%F %T') stage=$stage FAILED" | tee -a "$log"; exit 1
  fi
  echo "$(date '+%F %T') stage=$stage OK" | tee -a "$log"
done
echo "$(date '+%F %T') PHASE6A_STAGES_DONE" | tee -a "$log"
