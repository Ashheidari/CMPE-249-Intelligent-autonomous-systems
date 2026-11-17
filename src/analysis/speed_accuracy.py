"""M7 analysis 2: speed-accuracy tradeoff. Joins M4 accuracy with M5 latency/memory.

Usage: .venv/bin/python -m src.analysis.speed_accuracy

Also measures, on the same 200 timed benchmark images, the network input size each model actually
uses (after its own resize/letterbox), and counts model parameters, so latency can be related to
input pixels. Writes results/analysis/speed_accuracy.json and report/tables/speed_accuracy.md.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from ..datasets.coco import DATASETS, ROOT, coco80_names, load_gt
from ..models import build_detector
from .tables import DATASETS as DS_NAMES
from .tables import MODELS, md_table

RESULTS = ROOT / "results"
TABLES = ROOT / "report" / "tables"


def input_shape(model: str, inputs: dict) -> tuple[int, int]:
    """(H, W) of the tensor the network sees, padding included."""
    if model == "faster_rcnn":
        return tuple(inputs["images"].tensors.shape[-2:])
    if model == "yolo11s":
        return tuple(inputs["x"].shape[-2:])
    return tuple(inputs["pix"]["pixel_values"].shape[-2:])  # gdino


def n_params(det) -> int:
    return sum(p.numel() for p in det.model.parameters()) if hasattr(det.model, "parameters") \
        else sum(p.numel() for p in det.model.model.parameters())


def measure_inputs() -> dict:
    names, out = coco80_names(), {}
    images = {}
    for ds in DS_NAMES:
        gt = load_gt(ds)
        ids = [int(x) for x in (RESULTS / "benchmark" / f"{ds}_images.txt").read_text().split()][20:]  # timed set
        images[ds] = [Image.open(DATASETS[ds]["images"] / gt.loadImgs(i)[0]["file_name"]).convert("RGB") for i in ids]
    for m in MODELS:
        det = build_detector(m)
        out[m] = {"params_M": n_params(det) / 1e6}
        for ds, imgs in images.items():
            shapes = [input_shape(m, det.preprocess(im, names)) for im in imgs]
            px = [h * w for h, w in shapes]
            orig = [im.size[0] * im.size[1] for im in imgs]
            common = max(set(shapes), key=shapes.count)
            out[m][ds] = {"input_megapixels_mean": float(np.mean(px)) / 1e6,
                          "original_megapixels_mean": float(np.mean(orig)) / 1e6,
                          "most_common_input_hw": list(common), "n_distinct_shapes": len(set(shapes))}
        del det
    return out


def main() -> None:
    inputs = measure_inputs()
    rows, res = [], {}
    for m in MODELS:
        for ds in DS_NAMES:
            met = json.loads((RESULTS / f"{m}_{ds}" / "metrics.json").read_text())["stats"]
            b = json.loads((RESULTS / "benchmark" / f"{m}_{ds}" / "benchmark.json").read_text())
            r = b["runs"][-1]
            rec = {"AP": met["AP"], "AP50": met["AP50"],
                   "latency_ms": {s: r[s]["mean"] for s in ("preprocess", "forward", "postprocess")},
                   "latency_total_ms_mean": r["total"]["mean"], "latency_total_ms_median": r["total"]["median"],
                   "fps": r["fps"], "peak_driver_mb": b["memory_mb"]["peak_during_timed"]["driver_allocated_mb"],
                   "peak_alloc_mb": b["memory_mb"]["peak_during_timed"]["current_allocated_mb"],
                   "params_M": inputs[m]["params_M"], **inputs[m][ds]}
            rec["forward_ms_per_input_mpx"] = rec["latency_ms"]["forward"] / rec["input_megapixels_mean"]
            res.setdefault(m, {})[ds] = rec
            h, w = rec["most_common_input_hw"]
            rows.append([MODELS[m], DS_NAMES[ds], f"{100 * rec['AP']:.1f}", f"{rec['latency_total_ms_median']:.1f}",
                         f"{rec['fps']:.1f}", f"{rec['peak_driver_mb']:.0f}", f"{rec['params_M']:.1f}",
                         f"{w}×{h}", f"{rec['input_megapixels_mean']:.2f}", f"{rec['original_megapixels_mean']:.2f}",
                         f"{rec['forward_ms_per_input_mpx']:.0f}"])
    out = RESULTS / "analysis" / "speed_accuracy.json"
    out.write_text(json.dumps(res, indent=2))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "speed_accuracy.md").write_text(
        "# Speed-accuracy summary\n\n"
        + md_table(["model", "dataset", "AP", "latency median (ms)", "FPS", "peak driver mem (MB)", "params (M)",
                    "typical input W×H", "input Mpx (mean)", "original Mpx (mean)", "forward ms / input Mpx"], rows)
        + "\n\nAP: COCO = 80 classes, KITTI = car/person/truck (M4). Latency, FPS and memory: M5 benchmark "
        "(batch 1, MPS, last repeat). Input size is measured after each model's own resize/letterbox (padding "
        "included) on the 200 timed benchmark images. Grounding DINO's forward time covers 4 passes (one per "
        "prompt chunk), so its ms per input Mpx counts each pixel 4 times.\n")
    print((TABLES / "speed_accuracy.md").read_text())


if __name__ == "__main__":
    main()
