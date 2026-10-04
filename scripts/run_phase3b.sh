#!/usr/bin/env bash
# Open-set scorer comparison stages in pre-registered order. GPU stages run one at a time
# from this single process tree; each training takes exclusive GPU access itself.
# Stops at the first failing stage so later stages never run on partial inputs.
set -u
cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH=src
for stage in reproduce c2_guard c2_dev b1_ood c4_dev_confirm_headline select confirm business final_ood; do
  echo "$(date '+%F %T') stage=$stage start"
  if ! .venv/Scripts/python.exe -u -m intent_router.trackb_improve \
      --config configs/trackb_improve.yaml --stage "$stage"; then
    echo "$(date '+%F %T') stage=$stage FAILED"; exit 1
  fi
  echo "$(date '+%F %T') stage=$stage OK"
done
echo "$(date '+%F %T') PHASE3B_DONE"
