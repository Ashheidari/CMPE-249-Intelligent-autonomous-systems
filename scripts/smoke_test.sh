#!/usr/bin/env bash
# Milestone 1: run all 3 detectors on the same 5 COCO images -> results/smoke/. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
scripts/log_cmd.sh .venv/bin/python scripts/smoke_test.py "$@"
