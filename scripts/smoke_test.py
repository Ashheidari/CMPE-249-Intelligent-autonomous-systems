"""M1 smoke test: run all 3 detectors on the same 5 COCO val2017 images.

Saves visualizations to results/smoke/<model>/, prints detection counts and per-image time,
and writes results/smoke/summary.json. As a numeric alignment check, reports the fraction of
detections with score >= 0.5 that overlap a same-class GT box at IoU >= 0.5 (a resize offset
would make this collapse).

Usage: .venv/bin/python scripts/smoke_test.py [--device mps] [--models faster_rcnn yolo11s gdino]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402
import torch  # noqa: E402

from src.devices import synchronize  # noqa: E402
from src.models import MODEL_NAMES, build_detector  # noqa: E402
from src.viz import draw_detections  # noqa: E402

N_IMAGES = 5
OUT = ROOT / "results" / "smoke"


def gt_match_rate(det: dict, coco: COCO, img_id: int, min_score: float = 0.5) -> tuple[int, int]:
    """(# confident detections matching a same-class GT at IoU>=0.5, # confident detections)."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    gt = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in anns], dtype=torch.float32).reshape(-1, 4)
    gt_lbl = [names[a["category_id"]] for a in anns]
    keep = det["scores"] >= min_score
    boxes = torch.from_numpy(det["boxes"][keep])
    labels = [lab for lab, k in zip(det["labels"], keep) if k]
    if len(boxes) == 0 or len(gt) == 0:
        return 0, len(boxes)
    iou = box_iou(boxes, gt)  # [N, G]
    same = torch.tensor([[lab == g for g in gt_lbl] for lab in labels])
    return int(((iou >= 0.5) & same).any(1).sum()), len(boxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    args = ap.parse_args()

    coco = COCO(str(ROOT / "data/coco/annotations/instances_val2017.json"))
    class_names = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    img_ids = sorted(coco.getImgIds())[:N_IMAGES]  # first 5 by id: deterministic
    images = {i: Image.open(ROOT / "data/coco/val2017" / coco.loadImgs(i)[0]["file_name"]).convert("RGB")
              for i in img_ids}
    display = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["display_thresh"]

    summary = {"image_ids": img_ids, "display_thresh": display, "models": {}}
    for name in args.models:
        det_model = build_detector(name, args.device)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        rows, times, matched, confident, unknown = [], [], 0, 0, set()
        for img_id, img in images.items():
            synchronize(det_model.device)
            t0 = time.perf_counter()
            det = det_model.predict(img, class_names)
            synchronize(det_model.device)
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            W, H = img.size
            b = det["boxes"]
            in_bounds = bool(len(b) == 0 or (b[:, [0, 2]].min() >= -1 and b[:, [0, 2]].max() <= W + 1
                                             and b[:, [1, 3]].min() >= -1 and b[:, [1, 3]].max() <= H + 1))
            unknown |= set(det["labels"]) - set(class_names)
            m, c = gt_match_rate(det, coco, img_id)
            matched, confident = matched + m, confident + c
            n_disp = int((det["scores"] >= display).sum())
            rows.append({"image_id": img_id, "size": [W, H], "n_dets": len(b), "n_dets_display": n_disp,
                         "ms": round(dt, 1), "boxes_in_bounds": in_bounds, "gt_matched_ge0.5": f"{m}/{c}"})
            draw_detections(img, det, display, f"{name} | {img_id} | {n_disp} dets >= {display}") \
                .save(OUT / name / f"{img_id:012d}.png")
        summary["models"][name] = {
            "device": det_model.device,
            "device_fallback": det_model.stats.get("device_fallback"),
            # first image includes one-off costs (e.g. prompt tokenization, kernel compilation)
            "ms_per_image_mean_excl_first": round(float(np.mean(times[1:])), 1),
            "confident_dets_matching_gt": f"{matched}/{confident}",
            "labels_not_in_coco": sorted(unknown),
            "stats": det_model.stats,
            "per_image": rows,
        }
        print(f"\n== {name} on {det_model.device} ==")
        for r in rows:
            print(f"  img {r['image_id']:>6} {r['size']}: {r['n_dets']:3d} dets, {r['n_dets_display']:2d} >= {display}, "
                  f"{r['ms']:7.1f} ms, in-bounds={r['boxes_in_bounds']}, conf. dets matching GT {r['gt_matched_ge0.5']}")
        print(f"  mean ms/img (excl. first): {summary['models'][name]['ms_per_image_mean_excl_first']}"
              f" | confident dets matching GT: {matched}/{confident} | unknown labels: {sorted(unknown) or 'none'}"
              f" | stats: {det_model.stats}")
        del det_model
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'summary.json'} and PNGs under {OUT}/<model>/")


if __name__ == "__main__":
    main()
