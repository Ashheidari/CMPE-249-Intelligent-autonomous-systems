"""Latency of images with a new vs. an already-seen input shape on MPS -> results/benchmark/shape_effect.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.datasets.coco import DATASETS, coco80_names, load_gt, select_image_ids  # noqa: E402
from src.eval.benchmark import time_one  # noqa: E402
from src.models import build_detector  # noqa: E402

INPUT_SHAPE = {  # model -> shape of the network input produced by preprocess()
    "yolo11s": lambda inp: tuple(inp["x"].shape),
    "faster_rcnn": lambda inp: tuple(inp["images"].tensors.shape),
}


def main() -> None:
    names, rows = coco80_names(), []
    for ds in ("coco", "kitti"):
        gt = load_gt(ds)
        ids = select_image_ids(gt, 220)
        imgs = [Image.open(DATASETS[ds]["images"] / gt.loadImgs(i)[0]["file_name"]).convert("RGB") for i in ids]
        for model, shape_of in INPUT_SHAPE.items():
            det = build_detector(model)
            seen: set = set()
            for im in imgs[:20]:
                seen.add(shape_of(det.preprocess(im, names)))
                time_one(det, im, names)
            new, old = [], []
            for im in imgs[20:]:
                s = shape_of(det.preprocess(im, names))
                (old if s in seen else new).append(sum(time_one(det, im, names).values()))
                seen.add(s)
            rows.append({"model": model, "dataset": ds, "distinct_shapes": len(seen), "n_new": len(new),
                         "ms_new": float(np.mean(new)) if new else None, "ms_seen": float(np.mean(old))})
            print(rows[-1])
    out = ROOT / "results" / "benchmark" / "shape_effect.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
