#!/usr/bin/env bash
# Download COCO val2017 images (~780 MB) and 2017 train/val annotations (~250 MB)
# into data/coco/. Safe to re-run: skips files that already exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/coco"
mkdir -p "$DEST"
cd "$DEST"

fetch() {  # fetch <url>: download (resumable) and unzip once
  local url="$1" zip
  zip="$(basename "$url")"
  if [[ ! -f "$zip.done" ]]; then
    curl -L --fail -C - -o "$zip" "$url"
    unzip -q -o "$zip"
    touch "$zip.done"
    rm -f "$zip"
  else
    echo "already have $zip"
  fi
}

fetch http://images.cocodataset.org/zips/val2017.zip
fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "val2017 images: $(ls val2017 | wc -l | tr -d ' ')"
ls -lh annotations/instances_val2017.json
