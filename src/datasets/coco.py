"""COCO-format dataset registry shared by inference and evaluation.

Both datasets are described by a COCO-format GT JSON plus an image root:
  - coco:  COCO val2017, 5,000 images, 80 categories (ids 1..90 with gaps).
  - kitti: the fixed 1,500-frame KITTI subset from M2, categories car=3, person=1, truck=8
           (COCO's own ids), ignore regions as iscrowd=1.
  - kitti_native: the same 1,500 frames with KITTI-native classes car, van, truck, pedestrian,
           cyclist (ids 101-105; M8, Grounding DINO only). Its class names are the text prompt.

Image selection is deterministic: images sorted by id, `--limit N` keeps the first N. Inference
and evaluation both call `select_image_ids`, so they always agree on the image set.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]
COCO_GT = ROOT / "data" / "coco" / "annotations" / "instances_val2017.json"

DATASETS: dict[str, dict[str, Path]] = {
    "coco": {"gt": COCO_GT, "images": ROOT / "data" / "coco" / "val2017"},
    "kitti": {"gt": ROOT / "data" / "kitti" / "annotations" / "kitti_subset_coco.json",
              "images": ROOT / "data" / "kitti"},  # file_name is relative to data/kitti
    "kitti_native": {"gt": ROOT / "data" / "kitti" / "annotations" / "kitti_subset_native.json",
                     "images": ROOT / "data" / "kitti"},
}
# Text prompt / label space per dataset: COCO's 80 names, or the dataset's own category names.
PROMPT = {"coco": "coco80", "kitti": "coco80", "kitti_native": "dataset"}


def load_gt(dataset: str) -> COCO:
    """Load the GT of `dataset` with pycocotools."""
    return COCO(str(DATASETS[dataset]["gt"]))


def select_image_ids(gt: COCO, limit: int | None = None) -> list[int]:
    """Sorted image ids, truncated to the first `limit` if given."""
    ids = sorted(gt.getImgIds())
    return ids[:limit] if limit else ids


@lru_cache(maxsize=1)
def coco80_names() -> list[str]:
    """The 80 COCO class names, ordered by category id. Used as the label space / prompt for
    every model on every dataset, so KITTI runs see the same 80-way problem as COCO runs."""
    cats = json.loads(COCO_GT.read_text())["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def prompt_names(dataset: str, gt: COCO) -> list[str]:
    """Class names given to the detector for this dataset (Grounding DINO: its text prompt)."""
    if PROMPT[dataset] == "coco80":
        return coco80_names()
    return [c["name"] for c in sorted(gt.loadCats(gt.getCatIds()), key=lambda c: c["id"])]


def name_to_cat_id(gt: COCO) -> dict[str, int]:
    """Class name -> category id for the categories this dataset evaluates."""
    return {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
