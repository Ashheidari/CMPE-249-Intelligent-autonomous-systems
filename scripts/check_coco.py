"""Verify COCO val2017: 5,000 images on disk and instances_val2017.json loads with pycocotools.

Usage: .venv/bin/python scripts/check_coco.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = ROOT / "data" / "coco"
EXPECTED = 5000


def main() -> int:
    images = list((COCO_DIR / "val2017").glob("*.jpg"))
    ann_file = COCO_DIR / "annotations" / "instances_val2017.json"
    print(f"val2017 images on disk: {len(images)} (expected {EXPECTED})")
    if not ann_file.is_file():
        print(f"missing {ann_file}; download it as in README step 1")
        return 1
    coco = COCO(str(ann_file))
    n_img, n_ann, n_cat = len(coco.getImgIds()), len(coco.getAnnIds()), len(coco.getCatIds())
    print(f"annotation JSON: {n_img} images, {n_ann} annotations, {n_cat} categories")
    on_disk = {p.name for p in images}
    missing = [i["file_name"] for i in coco.loadImgs(coco.getImgIds()) if i["file_name"] not in on_disk]
    ok = len(images) == EXPECTED and n_img == EXPECTED and n_cat == 80 and not missing
    print("COCO check passed." if ok else f"COCO check FAILED ({len(missing)} annotated images missing)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
