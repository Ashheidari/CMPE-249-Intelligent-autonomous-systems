#!/usr/bin/env bash
# Milestone 4: full runs, 3 models x 2 datasets, one at a time, then the report tables.
# Each run logs to logs/m4/<model>_<dataset>.log. Finished runs are skipped; interrupted ones resume.
# Approx. time on an M5 (MPS): faster_rcnn 22 min COCO / 5 min KITTI, yolo11s 1-2 min each,
# gdino 3 h 50 min COCO / 45 min KITTI.
# Usage: scripts/m4_full_runs.sh [model dataset]   (no args: all 6, then tables)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
L=scripts/log_cmd.sh
PY=.venv/bin/python
mkdir -p logs/m4

run() {  # run <model> <dataset>
  caffeinate -i $L $PY -m src.eval.run_inference --model "$1" --dataset "$2" > "logs/m4/$1_$2.log" 2>&1
  $L $PY -m src.eval.coco_eval --run "results/$1_$2" >> "logs/m4/$1_$2.log" 2>&1
  grep -E "^AP " "logs/m4/$1_$2.log"
}

if [[ $# -eq 2 ]]; then
  run "$1" "$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [model dataset]" >&2; exit 2
else
  for m in faster_rcnn yolo11s gdino; do for d in coco kitti; do run $m $d; done; done
  $L $PY -m src.analysis.tables
fi
