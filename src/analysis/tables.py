"""Turn results/<model>_<dataset>/{metrics,config}.json into report tables.

Usage: .venv/bin/python -m src.analysis.tables

Writes to report/tables/:
  main_results.md / .csv   model x dataset, all 12 COCOeval stats, plus device and mean predict time
  per_class_kitti.md       AP and AP50 for car / person / truck per model, COCO vs. KITTI, with the drop
  published_check.md       our COCO AP vs. the published value for each checkpoint
  speed.md                 latency per stage, FPS, stability and memory from results/benchmark/*/benchmark.json
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


def rel_diff(v: list[float]) -> float:
    """(max - min) / min across repeats."""
    return (max(v) - min(v)) / min(v)


def speed() -> None:
    header = ["model", "dataset", "preprocess", "forward", "postprocess", "total mean", "total median",
              "total p90", "FPS", "run diff (mean)", "run diff (median)", "peak MPS alloc (MB)", "peak driver (MB)"]
    rows, notes, device, p = [], [], "-", {}
    for m in MODELS:
        for d in DATASETS:
            f = RESULTS / "benchmark" / f"{m}_{d}" / "benchmark.json"
            if not f.exists():
                continue
            b = json.loads(f.read_text())
            device, p = b["device"], b["protocol"]
            r = b["runs"][-1]  # last repeat; stability column compares all repeats
            mem = b["memory_mb"]["peak_during_timed"]
            rows.append([MODELS[m], DATASETS[d], *[f"{r[s]['mean']:.1f}" for s in ("preprocess", "forward", "postprocess")],
                         f"{r['total']['mean']:.1f}", f"{r['total']['median']:.1f}", f"{r['total']['p90']:.1f}",
                         f"{r['fps']:.1f}", f"{100 * b['stability_rel_diff_mean_total']:.1f}%",
                         f"{100 * rel_diff([x['total']['median'] for x in b['runs']]):.1f}%",
                         f"{mem.get('current_allocated_mb', 0):.0f}", f"{mem.get('driver_allocated_mb', 0):.0f}"])
            if "gdino_text_encoder" in b:
                te = b["gdino_text_encoder"]
                notes.append(f"- Grounding DINO on {DATASETS[d]}: text encoder (BERT, 4 prompt chunks) alone takes "
                             f"{te['ms_per_image_all_chunks']['mean']:.1f} ms/img, about "
                             f"{100 * te['share_of_forward']:.1f}% of the forward time.")
    text = (f"# Speed and memory (batch 1, device {device})\n\n"
            "Latency in ms per image (stage means; total = sum of stages per image).\n\n"
            f"{md_table(header, rows)}\n\n"
            f"Protocol: {p.get('warmup')} warm-up + {p.get('n_timed')} timed images (same list for every model), "
            f"{p.get('repeats')} repeats; table shows the last repeat. run diff = (max-min)/min across repeats of the "
            "mean or median total; the median is the stability criterion (<10%), because the mean of the first "
            "repeat includes one-off MPS kernel compilation for input shapes not seen during warm-up (see below). Image decoding is not timed. Device synchronized before every timestamp. Evaluation "
            "thresholds (score 0.001, 100 dets). NMS: Faster R-CNN final NMS in postprocess (proposal NMS inside "
            "forward); YOLO11s NMS in postprocess; Grounding DINO forward = 4 prompt-chunk passes, postprocess "
            "includes the cross-chunk NMS.\n\nMemory: peak of torch.mps.current_allocated_memory() (tensors) and "
            "driver_allocated_memory() (all Metal allocations) after each image. Apple Silicon unified memory is "
            "shared with the CPU, so this is not equivalent to CUDA VRAM.\n\n" + "\n".join(notes) + "\n" + shape_note())
    (OUT / "speed.md").write_text(text)


def shape_note() -> str:
    """Append the first-seen-input-shape measurement (scripts/mps_shape_effect.py), if available."""
    f = RESULTS / "benchmark" / "shape_effect.json"
    if not f.exists():
        return ""
    lines = ["\n## First-seen input shapes on MPS\n",
             "COCO images vary in size, so the network input shape varies; MPS compiles kernels the first time it "
             "sees a shape. KITTI frames all have one size. Measured after the same 20 warm-up images:\n",
             md_table(["model", "dataset", "distinct shapes", "imgs with new shape", "ms (new shape)", "ms (seen shape)"],
                      [[MODELS[r["model"]], DATASETS[r["dataset"]], str(r["distinct_shapes"]), str(r["n_new"]),
                        f"{r['ms_new']:.1f}" if r["n_new"] else "-", f"{r['ms_seen']:.1f}"]
                       for r in json.loads(f.read_text())])]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_results()
    per_class_kitti()
    published_check()
    speed()
    for f in ("main_results.md", "per_class_kitti.md", "published_check.md", "speed.md"):
        print((OUT / f).read_text())


if __name__ == "__main__":
    main()
