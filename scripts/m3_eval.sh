#!/usr/bin/env bash
# Milestone 3: end-to-end check on the first 200 images of each dataset for all 3 models,
# then the YOLO11s cross-check against Ultralytics' validator on all 5,000 COCO images.
# Usage: scripts/m3_eval.sh [limit|crosscheck|all]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
L=scripts/log_cmd.sh
PY=.venv/bin/python
STEP="${1:-all}"

if [[ "$STEP" == limit || "$STEP" == all ]]; then
  for model in faster_rcnn yolo11s gdino; do
    for ds in coco kitti; do
      $L $PY -m src.eval.run_inference --model $model --dataset $ds --limit 200
      $L $PY -m src.eval.coco_eval --run results/${model}_${ds}_limit200
    done
  done
fi

if [[ "$STEP" == crosscheck || "$STEP" == all ]]; then
  $L $PY -m src.eval.run_inference --model yolo11s --dataset coco      # full run, reused in M4
  $L $PY -m src.eval.coco_eval --run results/yolo11s_coco
  $L $PY scripts/yolo_crosscheck.py
fi
