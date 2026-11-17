"""Turn results/<model>_<dataset>/{metrics,config}.json into report tables.

Usage: .venv/bin/python -m src.analysis.tables

Writes to report/tables/:
  main_results.md / .csv   model x dataset, all 12 COCOeval stats, plus device and mean predict time
  per_class_kitti.md       AP and AP50 for car / person / truck per model, COCO vs. KITTI, with the drop
  published_check.md       our COCO AP vs. the published value for each checkpoint
All AP/AR values are in percent (COCOeval numbers x 100), rounded to 1 decimal.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..eval.coco_eval import STAT_NAMES

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "tables"
MODELS = {"faster_rcnn": "Faster R-CNN R50-FPN v2", "yolo11s": "YOLO11s", "gdino": "Grounding DINO tiny"}
DATASETS = {"coco": "COCO val2017", "kitti": "KITTI subset"}
SHARED_CLASSES = ["car", "person", "truck"]

# Published COCO val2017 box AP for the exact checkpoints we use.
PUBLISHED = {
    "faster_rcnn": (46.7, "torchvision weights metadata, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"),
    "yolo11s": (47.0, "Ultralytics YOLO11 docs, mAP val 50-95 at 640"),
    "gdino": (48.4, "IDEA-Research/GroundingDINO model zoo, Swin-T (O365, GoldG, Cap4M), zero-shot"),
}
FLAG_AP = 3.0  # plan: flag results more than ~3 AP from the published value


def load_run(model: str, dataset: str) -> tuple[dict, dict] | None:
    d = RESULTS / f"{model}_{dataset}"
    if not (d / "metrics.json").exists():
        return None
    return json.loads((d / "metrics.json").read_text()), json.loads((d / "config.json").read_text())


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def main_results() -> None:
    header = ["model", "dataset", "images", *STAT_NAMES, "device", "predict ms/img"]
    rows = []
    for m in MODELS:
        for d in DATASETS:
            run = load_run(m, d)
            if run is None:
                continue
            met, cfg = run
            rows.append([MODELS[m], DATASETS[d], str(met["n_images"]),
                         *[pct(met["stats"][k]) for k in STAT_NAMES],
                         cfg["detector"]["device"], f"{cfg['timing']['predict_ms_mean']:.1f}"])
    with (OUT / "main_results.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    note = ("AP/AR in %, from pycocotools COCOeval (bbox, maxDets=100). COCO: 80 classes; KITTI: car, person, "
            "truck with Van/Cyclist/Tram/Misc/DontCare as ignore regions. predict ms/img is the mean time of "
            "one predict() call during the evaluation run (batch 1), not the M5 benchmark.")
    (OUT / "main_results.md").write_text(f"# Main results\n\n{md_table(header, rows)}\n\n{note}\n")


def per_class_kitti() -> None:
    header = ["model", "class", "AP COCO", "AP KITTI", "drop", "rel. drop",
              "AP50 COCO", "AP50 KITTI", "GT COCO", "GT KITTI"]
    rows = []
    for m in MODELS:
        coco, kitti = load_run(m, "coco"), load_run(m, "kitti")
        if coco is None or kitti is None:
            continue
        for c in SHARED_CLASSES:
            a, b = coco[0]["per_class"][c], kitti[0]["per_class"][c]
            drop = a["AP"] - b["AP"]
            rows.append([MODELS[m], c, pct(a["AP"]), pct(b["AP"]), pct(drop), f"{100 * drop / a['AP']:.0f}%",
                         pct(a["AP50"]), pct(b["AP50"]), str(a["n_gt"]), str(b["n_gt"])])
    note = ("Per-class AP (%) = precision averaged over IoU 0.50:0.95 and recall, area=all, maxDets=100. "
            "drop = AP COCO - AP KITTI (AP points); rel. drop = drop / AP COCO. "
            "COCO AP here is from the 80-class evaluation; M6 re-evaluates COCO on these 3 classes only.")
    (OUT / "per_class_kitti.md").write_text(f"# Per-class AP: COCO vs. KITTI\n\n{md_table(header, rows)}\n\n{note}\n")


def published_check() -> None:
    header = ["model", "our COCO AP", "published AP", "difference", f"flag (>{FLAG_AP:g} AP)", "source"]
    rows = []
    for m, (ref, src) in PUBLISHED.items():
        run = load_run(m, "coco")
        if run is None:
            continue
        ours = 100 * run[0]["stats"]["AP"]
        rows.append([MODELS[m], f"{ours:.1f}", f"{ref:.1f}", f"{ours - ref:+.1f}",
                     "yes" if abs(ours - ref) > FLAG_AP else "no", src])
    (OUT / "published_check.md").write_text(f"# COCO val2017: ours vs. published\n\n{md_table(header, rows)}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_results()
    per_class_kitti()
    published_check()
    for f in ("main_results.md", "per_class_kitti.md", "published_check.md"):
        print((OUT / f).read_text())


if __name__ == "__main__":
    main()
