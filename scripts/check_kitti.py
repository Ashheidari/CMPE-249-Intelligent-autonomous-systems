"""Verify the KITTI 2D object training split is in place.

Expects data/kitti/training/image_2/*.png and data/kitti/training/label_2/*.txt,
7,481 of each with matching stems. Exits 0 if OK, 1 with download instructions otherwise.

Usage: .venv/bin/python scripts/check_kitti.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "data" / "kitti" / "training"
EXPECTED = 7481

INSTRUCTIONS = f"""
KITTI is not (fully) present. It requires registration, so download it manually:

  1. Register / log in at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
  2. Download:
       - "Download left color images of object data set (12 GB)"  -> data_object_image_2.zip
       - "Download training labels of object data set (5 MB)"     -> data_object_label_2.zip
  3. Unzip both into {ROOT / 'data' / 'kitti'}:
       mkdir -p data/kitti
       unzip data_object_image_2.zip -d data/kitti
       unzip data_object_label_2.zip -d data/kitti
     giving data/kitti/training/image_2/ and data/kitti/training/label_2/
  4. Re-run: .venv/bin/python scripts/check_kitti.py
"""


def main() -> int:
    img_dir, lbl_dir = KITTI / "image_2", KITTI / "label_2"
    images = {p.stem for p in img_dir.glob("*.png")} if img_dir.is_dir() else set()
    labels = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()

    print(f"{img_dir}: {len(images)} images (expected {EXPECTED})")
    print(f"{lbl_dir}: {len(labels)} labels (expected {EXPECTED})")

    ok = len(images) == EXPECTED and len(labels) == EXPECTED and images == labels
    if ok:
        print("KITTI check passed.")
        return 0
    if images and labels and images != labels:
        print(f"Mismatch: {len(images - labels)} images without labels, "
              f"{len(labels - images)} labels without images.")
    print(INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
