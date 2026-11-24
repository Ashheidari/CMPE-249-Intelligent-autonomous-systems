"""Check the KITTI training split: 7,481 paired frames with 15-field labels."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.kitti_files import DEFAULT_KITTI_ROOT, EXPECTED_FRAMES, kitti_index  # noqa: E402

INSTRUCTIONS = f"""
KITTI is not (fully) present. Get the labeled training split with the ORIGINAL 15-field
labels (Ultralytics' YOLO-format re-pack is not suitable: no DontCare/occlusion fields).
Either:
  - official: register at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d,
    download data_object_image_2.zip + data_object_label_2.zip, unzip both into a folder
    (giving training/image_2 and training/label_2), or
  - the Kaggle re-pack with train/ and val/ {{images,labels}} folders,
and pass that folder with --root (default {DEFAULT_KITTI_ROOT}).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_KITTI_ROOT)
    args = ap.parse_args()

    try:
        index = kitti_index(args.root)
    except FileNotFoundError as e:
        print(e, INSTRUCTIONS, sep="\n")
        return 1

    no_img = [k for k, (img, _) in index.items() if img is None]
    no_lbl = [k for k, (_, lbl) in index.items() if lbl is None]
    paired = [(img, lbl) for img, lbl in index.values() if img and lbl]
    print(f"root: {args.root}")
    print(f"frames: {len(index)} (expected {EXPECTED_FRAMES}), paired: {len(paired)}, "
          f"missing image: {len(no_img)}, missing label: {len(no_lbl)}")

    fields, classes = Counter(), Counter()
    for _, lbl in paired:
        for line in lbl.read_text().splitlines():
            if line.strip():
                parts = line.split()
                fields[len(parts)] += 1
                classes[parts[0]] += 1
    print(f"fields per label line: {dict(fields)}")
    print("boxes per type: " + ", ".join(f"{k}={v}" for k, v in sorted(classes.items())))

    ok = (len(paired) == len(index) == EXPECTED_FRAMES
          and set(fields) == {15} and classes["DontCare"] > 0)
    if ok:
        print("KITTI check passed.")
        return 0
    print("KITTI check FAILED.", INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
