#!/usr/bin/env bash
# Milestone 2: KITTI subset -> COCO GT JSON, stats, GT renders + evaluator sanity check.
set -euo pipefail
cd "$(dirname "$0")/.."
L=scripts/log_cmd.sh
$L .venv/bin/python -m src.datasets.kitti_to_coco "$@"
$L .venv/bin/python scripts/kitti_gt_check.py
