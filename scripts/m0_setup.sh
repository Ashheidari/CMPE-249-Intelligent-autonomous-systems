#!/usr/bin/env bash
# Milestone 0: environment + data. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
L=scripts/log_cmd.sh
[[ -d .venv ]] || $L python3.11 -m venv .venv
$L .venv/bin/pip install -r requirements.txt
$L .venv/bin/python scripts/record_env.py
$L scripts/download_coco.sh
$L .venv/bin/python scripts/check_coco.py
$L .venv/bin/python scripts/check_kitti.py || echo ">> Download KITTI manually (see above), then re-run scripts/check_kitti.py"
