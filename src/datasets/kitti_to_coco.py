"""Convert KITTI 2D-object labels (seeded subset of the training split) to a COCO GT JSON.

Label policy comes from configs/kitti_mapping.yaml. Each KITTI type maps either to a COCO
class name in `eval_classes` or to "ignore".

Why ignore regions are written as iscrowd=1: pycocotools treats a GT with iscrowd=1 as an ignore
region. A detection matched to it is neither a true positive nor a false positive, and missing it
costs nothing. This is what we want for KITTI types COCO has no clean class for (Van, Cyclist,
Tram, Misc) and for DontCare areas, so a model is not punished for correctly boxing a van.
COCOeval matches per category, so an ignore box only absorbs detections of its own category;
we therefore write every ignore box once per eval class (car, person, truck).
Crowd matching uses IoU = intersection / detection area, so a large DontCare region can absorb
several small detections inside it.

Box format: KITTI bbox is (left, top, right, bottom) in pixels; COCO bbox is [x, y, w, h].
Usage: .venv/bin/python -m src.datasets.kitti_to_coco [--mapping configs/kitti_mapping.yaml] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .kitti_files import DEFAULT_KITTI_ROOT, kitti_index

ROOT = Path(__file__).resolve().parents[2]
SUBSET_FILE = DEFAULT_KITTI_ROOT / "subset_1500.txt"
DEFAULT_OUT = DEFAULT_KITTI_ROOT / "annotations" / "kitti_subset_coco.json"
COCO_IDS = {"person": 1, "car": 3, "truck": 8}  # COCO's own category ids
SUBSET_SIZE, SEED = 1500, 42
SMALL, MEDIUM = 32 ** 2, 96 ** 2  # COCO area cutoffs (px^2)


def parse_label_file(path: Path) -> list[dict[str, Any]]:
    """Parse one KITTI label file. Keeps type, truncated, occluded and the bbox (xyxy px).

    Line format (15 fields): type truncated occluded alpha left top right bottom h w l x y z rot_y.
    truncated is a float in [0, 1]; occluded is 0 (visible), 1 (partly), 2 (largely), 3 (unknown);
    DontCare lines use -1 for both.
    """
    objs = []
    for line in path.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        objs.append({"type": f[0], "truncated": float(f[1]), "occluded": int(float(f[2])),
                     "bbox_xyxy": [float(v) for v in f[4:8]]})
    return objs


def select_subset(all_ids: list[str], path: Path = SUBSET_FILE) -> list[str]:
    """Return the fixed subset: read it if saved, else draw SUBSET_SIZE ids with SEED and save."""
    if path.exists():
        ids = path.read_text().split()
        missing = set(ids) - set(all_ids)
        if missing:
            raise ValueError(f"{len(missing)} subset frames not found on disk, e.g. {sorted(missing)[:3]}")
        return ids
    ids = sorted(random.Random(SEED).sample(sorted(all_ids), SUBSET_SIZE))
    path.write_text("\n".join(ids) + "\n")
    print(f"wrote new subset list {path}")
    return ids


def size_bucket(area: float) -> str:
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def build_coco(frames: list[str], index: dict, mapping_cfg: dict, kitti_root: Path) -> dict:
    """Build the COCO GT dict for `frames`. image_id = int(frame id); file_name relative to kitti_root."""
    eval_classes, mapping = mapping_cfg["eval_classes"], mapping_cfg["mapping"]
    # Category ids: COCO's own ids by default; a mapping file may give its own (M8 native classes).
    cat_ids = mapping_cfg.get("category_ids", COCO_IDS)
    unknown = set(mapping.values()) - set(eval_classes) - {"ignore"}
    if unknown:
        raise ValueError(f"mapping targets not in eval_classes: {unknown}")
    images, anns = [], []
    for fid in frames:
        img_path, lbl_path = index[fid]
        W, H = Image.open(img_path).size
        images.append({"id": int(fid), "file_name": str(img_path.relative_to(kitti_root)),
                       "width": W, "height": H, "frame_id": fid})
        for o in parse_label_file(lbl_path):
            target = mapping[o["type"]]  # KeyError on an unexpected type is intentional
            x0, y0, x1, y1 = o["bbox_xyxy"]
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            cats = eval_classes if target == "ignore" else [target]
            for c in cats:
                anns.append({"id": len(anns) + 1, "image_id": int(fid), "category_id": cat_ids[c],
                             "bbox": [x0, y0, w, h], "area": w * h,
                             "iscrowd": int(target == "ignore"),
                             "kitti_type": o["type"], "truncated": o["truncated"], "occluded": o["occluded"]})
    categories = [{"id": cat_ids[c], "name": c, "supercategory": c} for c in eval_classes]
    return {"info": {"description": "KITTI 2D object, seeded training subset, COCO format",
                     "mapping": mapping}, "images": images, "annotations": anns, "categories": categories}


def stats_markdown(coco: dict) -> str:
    """Images, boxes per eval class, ignore boxes per KITTI type, COCO size distribution."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    real = [a for a in coco["annotations"] if not a["iscrowd"]]
    # each ignore box is replicated per eval class; count it once (first category only)
    first = coco["categories"][0]["id"]
    ign = [a for a in coco["annotations"] if a["iscrowd"] and a["category_id"] == first]
    lines = ["# KITTI subset statistics\n",
             f"- images: {len(coco['images'])}",
             f"- evaluated boxes: {len(real)}",
             f"- ignore regions (iscrowd=1, counted once each): {len(ign)}",
             f"- mapping: `{coco['info']['mapping']}`\n",
             "## Evaluated boxes by class and COCO size (area < 32², < 96², else large)\n",
             "| class | boxes | small | medium | large | from KITTI types |", "|---|---|---|---|---|---|"]
    for cid, name in names.items():
        a = [x for x in real if x["category_id"] == cid]
        sz = Counter(size_bucket(x["area"]) for x in a)
        src = Counter(x["kitti_type"] for x in a)
        lines.append(f"| {name} | {len(a)} | {sz['small']} | {sz['medium']} | {sz['large']} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(src.items())) + " |")
    sz = Counter(size_bucket(x["area"]) for x in real)
    lines.append(f"| **all** | {len(real)} | {sz['small']} | {sz['medium']} | {sz['large']} | |")
    lines += ["\n## Ignore regions by KITTI type\n", "| KITTI type | regions |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(Counter(a["kitti_type"] for a in ign).items())]
    lines += ["\n## Evaluated boxes by occlusion level (0 visible, 1 partly, 2 largely, 3 unknown)\n",
              "| class | occ 0 | occ 1 | occ 2 | occ 3 |", "|---|---|---|---|---|"]
    for cid, name in names.items():
        oc = Counter(x["occluded"] for x in real if x["category_id"] == cid)
        lines.append(f"| {name} | {oc[0]} | {oc[1]} | {oc[2]} | {oc[3]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", type=Path, default=ROOT / "configs" / "kitti_mapping.yaml")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    ap.add_argument("--stats", type=Path, default=ROOT / "results" / "kitti_stats.md")
    args = ap.parse_args()

    index = kitti_index(args.kitti_root)
    frames = select_subset([k for k, (i, lbl) in index.items() if i and lbl])
    coco = build_coco(frames, index, yaml.safe_load(args.mapping.read_text()), args.kitti_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco))
    print(f"wrote {args.out}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    md = stats_markdown(coco)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
