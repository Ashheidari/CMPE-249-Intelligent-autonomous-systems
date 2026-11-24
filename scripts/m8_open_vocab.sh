#!/usr/bin/env bash
# Milestone 8 (bonus): closed-set vs. open-vocabulary, plus two M7 follow-up experiments.
# Logs to logs/m8/. Finished runs are skipped; interrupted ones resume.
# Approx. time on an M5 (MPS): ~13 min per Grounding DINO KITTI run, ~10 min for the experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
L=scripts/log_cmd.sh
PY=.venv/bin/python
mkdir -p logs/m8

# Native KITTI GT: car, van, truck, pedestrian, cyclist as real classes (ids 101-105)
$L $PY -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_native.yaml \
    --out data/kitti/annotations/kitti_subset_native.json --stats results/analysis/kitti_stats_native.md

infer() {  # infer <log name> <run_inference args...>; then evaluates the run it wrote
  local log="logs/m8/$1.log"; shift
  caffeinate -i $L $PY -m src.eval.run_inference "$@" > "$log" 2>&1
  local run; run=$(grep -o 'results/[^/]*/predictions.json' "$log" | head -1 | cut -d/ -f2)
  [[ -z "$run" ]] && run=$(grep -o 'results/[^/]*/predictions.json already' "$log" | cut -d/ -f2)
  $L $PY -m src.eval.coco_eval --run "results/$run" >> "$log" 2>&1
  echo "$run: $(grep -E '^AP ' "$log" | tail -1)"
}

# 1. Grounding DINO with the KITTI-native prompt, plus two wordings for the pedestrian class
infer gdino_kitti_native          --model gdino --dataset kitti_native
infer gdino_kitti_native_person   --model gdino --dataset kitti_native --rename pedestrian=person
infer gdino_kitti_native_walking  --model gdino --dataset kitti_native --rename "pedestrian=walking person"

# 2. Experiment (M7 H1): YOLO11s on KITTI at imgsz 1280 instead of 640
infer yolo11s_kitti_imgsz1280     --model yolo11s --dataset kitti --set imgsz=1280
caffeinate -i $L $PY -m src.eval.benchmark --model yolo11s --dataset kitti --set imgsz=1280 > logs/m8/bench_yolo11s_kitti_imgsz1280.log 2>&1

# 3. Experiment (M7 H2): Grounding DINO with one 80-class prompt instead of 4 chunks of 20 (COCO, first 200)
infer gdino_coco_chunk80_limit200 --model gdino --dataset coco --limit 200 --set chunk_size=80
caffeinate -i $L $PY -m src.eval.benchmark --model gdino --dataset coco --set chunk_size=80 > logs/m8/bench_gdino_coco_chunk80.log 2>&1

$L $PY -m src.analysis.open_vocab
