#!/usr/bin/env bash
# Milestone 5: speed + memory benchmark (batch 1, 20 warm-up + 200 timed images, 2 repeats).
# Run with nothing else using the GPU. Logs to logs/m5/<model>_<dataset>.log.
# Approx. time on an M5: faster_rcnn ~2 min, yolo11s <1 min, gdino ~12 min KITTI / ~20 min COCO.
# Usage: scripts/m5_benchmark.sh [model dataset]   (no args: all 6, then report/tables/speed.md)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
L=scripts/log_cmd.sh
PY=.venv/bin/python
mkdir -p logs/m5

run() {  # run <model> <dataset>
  caffeinate -i $L $PY -m src.eval.benchmark --model "$1" --dataset "$2" > "logs/m5/$1_$2.log" 2>&1
  grep -E "^run |stability|text encoder|peak" "logs/m5/$1_$2.log"
}

if [[ $# -eq 2 ]]; then
  run "$1" "$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [model dataset]" >&2; exit 2
else
  for m in faster_rcnn yolo11s gdino; do for d in coco kitti; do run $m $d; done; done
  $L $PY -m src.analysis.tables
fi
