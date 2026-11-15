"""M2 acceptance check for the KITTI COCO-format GT.

1. Loads the JSON with pycocotools.COCO.
2. Renders 20 random subset images (seed 42) with GT to results/kitti_gt_check/
   (evaluated boxes solid, ignore regions dashed gray).
3. Evaluator sanity test: GT boxes submitted as predictions must give AP = 1.0, and adding
   detections of every eval class on top of each ignore region must leave AP at 1.0
   (they must be neither TP nor FP).

Usage: .venv/bin/python scripts/kitti_gt_check.py [--ann data/kitti/annotations/kitti_subset_coco.json]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from src.datasets.kitti_files import DEFAULT_KITTI_ROOT  # noqa: E402
from src.datasets.kitti_to_coco import DEFAULT_OUT  # noqa: E402
from src.viz import draw_gt  # noqa: E402

OUT = ROOT / "results" / "kitti_gt_check"


def ap_of(gt: COCO, dets: list[dict]) -> float:
    """COCO AP@[.5:.95] of `dets` against `gt` (stdout suppressed)."""
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, gt.loadRes(dets), iouType="bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    gt = COCO(str(args.ann))
    cats = {c["id"]: c["name"] for c in gt.loadCats(gt.getCatIds())}
    print(f"loaded: {len(gt.getImgIds())} images, {len(gt.getAnnIds())} annotations, categories {cats}")

    OUT.mkdir(parents=True, exist_ok=True)
    for img_id in random.Random(42).sample(sorted(gt.getImgIds()), 20):
        info = gt.loadImgs(img_id)[0]
        anns = gt.loadAnns(gt.getAnnIds(imgIds=img_id))
        n_eval = sum(not a["iscrowd"] for a in anns)
        title = f"{info['frame_id']} | {n_eval} eval boxes | dashed = ignore"
        draw_gt(Image.open(DEFAULT_KITTI_ROOT / info["file_name"]), anns, cats, title) \
            .save(OUT / f"{info['frame_id']}.png")
    print(f"rendered 20 images to {OUT}")

    real = [a for a in gt.dataset["annotations"] if not a["iscrowd"]]
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
               for a in real]
    on_ignore = [{"image_id": a["image_id"], "category_id": c, "bbox": a["bbox"], "score": 0.99}
                 for a in gt.dataset["annotations"] if a["iscrowd"] for c in cats]
    ap_perfect, ap_with_ignored = ap_of(gt, perfect), ap_of(gt, perfect + on_ignore)
    print(f"AP, GT as predictions: {ap_perfect:.4f} (expect 1.0)")
    print(f"AP, + {len(on_ignore)} detections on ignore regions: {ap_with_ignored:.4f} (expect 1.0)")
    ok = ap_perfect > 0.999 and ap_with_ignored > 0.999
    print("KITTI GT check passed." if ok else "KITTI GT check FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
