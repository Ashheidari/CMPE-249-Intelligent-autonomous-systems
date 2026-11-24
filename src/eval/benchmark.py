"""Latency and memory benchmark for one detector on one dataset (batch size 1).

Usage:
  .venv/bin/python -m src.eval.benchmark --model {faster_rcnn,yolo11s,gdino} --dataset {coco,kitti}
      [--warmup 20] [--n 200] [--repeats 2] [--device mps] [--check 20] [--set KEY=VALUE ...]

Protocol:
  - Image list: the first warmup + n images of the dataset sorted by id, the same for every model
    (saved to results/benchmark/<dataset>_images.txt). Images are decoded into memory first, so
    JPEG/PNG decoding is NOT timed; timing starts from a PIL RGB image.
  - `warmup` untimed images, then `n` timed images, repeated `repeats` times (to check stability).
  - Each image is split into three stages, with a device synchronize before every timestamp:
      preprocess   PIL -> model input on device (resize/letterbox, normalize, tokenized prompts)
      forward      the network
      postprocess  decode, score threshold, NMS, rescale to original pixels, top-k, copy to CPU
    Settings are the evaluation settings from configs/models.yaml (score threshold 0.001, 100 dets),
    so postprocess times are for the mAP setting, not a deployment threshold.
    Where NMS happens:
      faster_rcnn  final class-wise NMS in postprocess; RPN proposal NMS is inside forward
      yolo11s      NMS in postprocess
      gdino        no NMS within a prompt; the cross-chunk class-wise NMS is in postprocess. forward
                   is 4 passes (one per prompt chunk of 20 COCO classes), text encoder included.
  - Grounding DINO text-encoder share: the BERT text backbone is also timed on its own on the same
    prompt chunks (n times) and reported next to the forward time. This is an estimate: inside the
    real forward pass the text encoder uses a slightly different attention mask, same sequence length.
  - Memory (MPS): torch.mps.current_allocated_memory() (tensors) and driver_allocated_memory()
    (everything the Metal driver holds), peak sampled after each image. Apple Silicon has unified
    memory shared by CPU and GPU, so this is NOT equivalent to CUDA VRAM.
  - --check N first runs predict() and the staged path on N images and asserts they agree.

Output: results/benchmark/<model>_<dataset>/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..datasets.coco import DATASETS, ROOT, coco80_names, load_gt, select_image_ids
from ..devices import synchronize
from ..models import MODEL_NAMES, build_detector, parse_overrides
from .run_inference import versions

OUT = ROOT / "results" / "benchmark"
STAGES = ("preprocess", "forward", "postprocess")


def mem_now(device: str) -> dict[str, float]:
    """Allocated memory in MB. On MPS: tensor allocations and total driver allocations."""
    if device.startswith("mps"):
        return {"current_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
                "driver_allocated_mb": torch.mps.driver_allocated_memory() / 2**20}
    if device.startswith("cuda"):
        return {"current_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                "driver_allocated_mb": torch.cuda.memory_reserved() / 2**20}
    return {}


def summarize(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "std": float(a.std())}


def time_one(det, img: Image.Image, names: list[str]) -> dict[str, float]:
    """Run the three stages on one image; return per-stage milliseconds."""
    dev = det.device
    synchronize(dev)
    t0 = time.perf_counter()
    inputs = det.preprocess(img, names)
    synchronize(dev)
    t1 = time.perf_counter()
    outputs = det.forward(inputs)
    synchronize(dev)
    t2 = time.perf_counter()
    det.postprocess(outputs, inputs)  # returns CPU numpy arrays, so it ends synchronized
    synchronize(dev)
    t3 = time.perf_counter()
    return {"preprocess": (t1 - t0) * 1e3, "forward": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}


def check_staged(det, images: list[Image.Image], names: list[str]) -> dict:
    """Compare predict() with the staged path. Returns the worst differences seen."""
    worst = {"count_diff": 0, "max_box_diff_px": 0.0, "max_score_diff": 0.0, "labels_equal": True}
    for img in images:
        a = det.predict(img, names)
        with torch.no_grad():
            inputs = det.preprocess(img, names)
            b = det.postprocess(det.forward(inputs), inputs)
        worst["count_diff"] = max(worst["count_diff"], abs(len(a["scores"]) - len(b["scores"])))
        n = min(len(a["scores"]), len(b["scores"]))
        if n:
            worst["max_box_diff_px"] = max(worst["max_box_diff_px"], float(np.abs(a["boxes"][:n] - b["boxes"][:n]).max()))
            worst["max_score_diff"] = max(worst["max_score_diff"], float(np.abs(a["scores"][:n] - b["scores"][:n]).max()))
            worst["labels_equal"] &= a["labels"][:n] == b["labels"][:n]
    return worst


@torch.no_grad()
def text_encoder_ms(det, img: Image.Image, names: list[str], n: int) -> dict[str, float]:
    """Grounding DINO only: time the BERT text backbone alone on all prompt chunks, per image."""
    inputs = det.preprocess(img, names)
    bert = det.model.model.text_backbone
    ms = []
    for _ in range(n):
        synchronize(det.device)
        t0 = time.perf_counter()
        for enc in inputs["encs"]:
            bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 token_type_ids=enc.get("token_type_ids"))
        synchronize(det.device)
        ms.append((time.perf_counter() - t0) * 1e3)
    return summarize(ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check", type=int, default=20, help="compare predict() vs. staged path on N images (0: skip)")
    ap.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                    help="override model settings (output dir gets a suffix, e.g. yolo11s_kitti_imgsz1280)")
    args = ap.parse_args()
    overrides = parse_overrides(args.set)

    gt = load_gt(args.dataset)
    ids = select_image_ids(gt, args.warmup + args.n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.dataset}_images.txt").write_text("".join(f"{i}\n" for i in ids))
    root = DATASETS[args.dataset]["images"]
    images = [Image.open(root / gt.loadImgs(i)[0]["file_name"]).convert("RGB") for i in ids]
    warm, timed = images[:args.warmup], images[args.warmup:]
    names = coco80_names()

    det = build_detector(args.model, args.device, overrides)
    dev = det.device
    if dev.startswith("mps"):
        torch.mps.empty_cache()
    mem_after_load = mem_now(dev)

    check = check_staged(det, images[:args.check], names) if args.check else None
    if check:
        print(f"staged vs predict(): {check}")
        if check["count_diff"] or check["max_box_diff_px"] > 0.5 or not check["labels_equal"]:
            raise SystemExit("staged path disagrees with predict(); fix before benchmarking")

    for img in tqdm(warm, desc="warmup"):
        time_one(det, img, names)

    runs, peak = [], {}
    for r in range(args.repeats):
        per_img = []
        for img in tqdm(timed, desc=f"timed run {r + 1}/{args.repeats}"):
            per_img.append(time_one(det, img, names))
            for k, v in mem_now(dev).items():
                peak[k] = max(peak.get(k, 0.0), v)
        total = [sum(t.values()) for t in per_img]
        run = {s: summarize([t[s] for t in per_img]) for s in STAGES}
        run["total"] = summarize(total)
        run["fps"] = 1000 / run["total"]["mean"]
        runs.append(run)
        print(f"run {r + 1}: total mean {run['total']['mean']:.1f} ms, median {run['total']['median']:.1f}, "
              f"p90 {run['total']['p90']:.1f}, FPS {run['fps']:.1f} | "
              + ", ".join(f"{s} {run[s]['mean']:.1f}" for s in STAGES))

    means = [r["total"]["mean"] for r in runs]
    stability = (max(means) - min(means)) / min(means) if len(means) > 1 else None
    result = {
        "model": args.model, "dataset": args.dataset, "device": dev,
        "device_fallback": det.stats.get("device_fallback"),
        "protocol": {"batch_size": 1, "warmup": args.warmup, "n_timed": args.n, "repeats": args.repeats,
                     "image_ids_file": str((OUT / f"{args.dataset}_images.txt").relative_to(ROOT)),
                     "decode_timed": False, "settings": "eval thresholds from configs/models.yaml"},
        "detector": det.describe(), "overrides": overrides, "versions": versions(),
        "staged_check": check, "runs": runs,
        "stability_rel_diff_mean_total": stability,
        "memory_mb": {"after_load": mem_after_load, "peak_during_timed": peak,
                      "note": "MPS unified memory: shared with the CPU, not equivalent to CUDA VRAM"},
    }
    if args.model == "gdino":
        te = text_encoder_ms(det, timed[0], names, args.n)
        fwd = float(np.mean([r["forward"]["mean"] for r in runs]))
        result["gdino_text_encoder"] = {"ms_per_image_all_chunks": te, "share_of_forward": te["mean"] / fwd,
                                        "note": "BERT text backbone timed alone on the same prompt chunks"}
        print(f"text encoder: {te['mean']:.1f} ms/img = {100 * te['mean'] / fwd:.1f}% of forward")

    out = OUT / (f"{args.model}_{args.dataset}" + "".join(f"_{k}{v}" for k, v in overrides.items()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    if stability is not None:
        print(f"stability (max-min)/min of mean total over {args.repeats} runs: {100 * stability:.1f}%")
    print(f"peak memory: {peak} | wrote {out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
