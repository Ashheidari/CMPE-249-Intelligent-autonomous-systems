"""Run one detector on one dataset and write COCO-format predictions.

Usage:
  .venv/bin/python -m src.eval.run_inference --model {faster_rcnn,yolo11s,gdino} --dataset {coco,kitti}
      [--limit N] [--device mps] [--run-id ID] [--overwrite]

Outputs, in results/<run_id>/ (run_id defaults to <model>_<dataset>[_limit<N>]):
  predictions.json      COCO results format: [{image_id, category_id, bbox: [x, y, w, h], score}]
  config.json           model settings, dataset, image count, device, versions, label stats, timing
  predictions.partial.jsonl
                        one line per finished image; used to resume an interrupted run and
                        deleted once predictions.json is written

Label handling: every model is asked for the same 80 COCO classes on both datasets (for Grounding
DINO this is the text prompt). Predicted class names are then mapped to the dataset's category ids:
  - on COCO all 80 names map;
  - on KITTI only the eval classes (car, person, truck) map; all other predicted classes are dropped
    and counted in config.json label_stats.dropped_non_eval_class.
A predicted name outside COCO's 80 (should never happen) is counted in dropped_unknown_label.
Boxes come from the detector as xyxy original-image pixels and are converted to xywh here.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from ..datasets.coco import DATASETS, ROOT, coco80_names, load_gt, name_to_cat_id, select_image_ids
from ..devices import synchronize  # also sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads
from ..models import MODEL_NAMES, build_detector

RESULTS = ROOT / "results"


def versions() -> dict[str, str]:
    pkgs = ["torch", "torchvision", "ultralytics", "transformers", "pycocotools", "numpy", "pillow"]
    return {"python": platform.python_version(), **{p: version(p) for p in pkgs}}


def to_coco_results(det: dict, image_id: int, name2id: dict[str, int],
                    all_names: set[str]) -> tuple[list[dict], dict[str, int]]:
    """Convert one image's detections to COCO result dicts.

    det: {'boxes': [N, 4] xyxy original px, 'scores': [N], 'labels': list[str]}.
    Returns (results with bbox [x, y, w, h], counts of dropped detections by reason).
    """
    out, dropped = [], {"non_eval_class": 0, "unknown_label": 0}
    for (x0, y0, x1, y1), s, name in zip(det["boxes"].tolist(), det["scores"].tolist(), det["labels"]):
        if name not in name2id:
            dropped["non_eval_class" if name in all_names else "unknown_label"] += 1
            continue
        out.append({"image_id": image_id, "category_id": name2id[name],
                    "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                    "score": round(s, 5)})
    return out, dropped


def read_partial(path: Path) -> list[dict]:
    """Finished-image records from a previous session; a truncated last line is discarded."""
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (sorted by id)")
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: best available)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--overwrite", action="store_true", help="discard any previous output for this run id")
    args = ap.parse_args()

    run_id = args.run_id or f"{args.model}_{args.dataset}" + (f"_limit{args.limit}" if args.limit else "")
    run_dir = RESULTS / run_id
    pred_path, partial_path, cfg_path = (run_dir / "predictions.json", run_dir / "predictions.partial.jsonl",
                                         run_dir / "config.json")
    if args.overwrite:
        for p in (pred_path, partial_path, cfg_path):
            p.unlink(missing_ok=True)
    if pred_path.exists():
        print(f"{pred_path} already exists; use --overwrite to redo. Nothing to do.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.dataset)
    img_ids = select_image_ids(gt, args.limit)
    name2id = name_to_cat_id(gt)
    prompt_names = coco80_names()

    # Resume: keep finished images only if the previous session used the same model/dataset/limit.
    prev_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    done = read_partial(partial_path)
    if done and (prev_cfg.get("model"), prev_cfg.get("dataset"), prev_cfg.get("limit")) != \
            (args.model, args.dataset, args.limit):
        raise SystemExit(f"{partial_path} was written by a different run config; use --overwrite")
    done_ids = {r["image_id"] for r in done}
    todo = [i for i in img_ids if i not in done_ids]
    if done:
        print(f"resuming: {len(done_ids)} images done, {len(todo)} to go")

    det_model = build_detector(args.model, args.device)
    sessions = prev_cfg.get("sessions", [])
    started = datetime.now().isoformat(timespec="seconds")
    config: dict[str, Any] = {
        "run_id": run_id, "model": args.model, "dataset": args.dataset, "limit": args.limit,
        "n_images": len(img_ids), "gt": str(DATASETS[args.dataset]["gt"].relative_to(ROOT)),
        "eval_categories": name2id, "prompt_classes": "COCO-80 (ordered by category id)",
        "detector": det_model.describe(), "versions": versions(), "status": "running",
        "sessions": sessions + [{"started": started, "n_images": len(todo)}],
    }
    cfg_path.write_text(json.dumps(config, indent=2))

    img_root = DATASETS[args.dataset]["images"]
    t_session = time.perf_counter()
    with partial_path.open("a") as f:
        for img_id in tqdm(todo, desc=run_id, unit="img"):
            info = gt.loadImgs(img_id)[0]
            img = Image.open(img_root / info["file_name"]).convert("RGB")
            synchronize(det_model.device)
            t0 = time.perf_counter()
            det = det_model.predict(img, prompt_names)
            synchronize(det_model.device)
            ms = (time.perf_counter() - t0) * 1000
            res, dropped = to_coco_results(det, img_id, name2id, set(prompt_names))
            f.write(json.dumps({"image_id": img_id, "ms": round(ms, 2), "dropped": dropped, "dets": res}) + "\n")
            f.flush()
    wall = time.perf_counter() - t_session

    recs = {r["image_id"]: r for r in read_partial(partial_path)}
    missing = [i for i in img_ids if i not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from {partial_path}; rerun to resume")
    preds = [d for i in img_ids for d in recs[i]["dets"]]
    pred_path.write_text(json.dumps(preds))

    ms = [recs[i]["ms"] for i in img_ids]
    config["sessions"][-1]["wall_clock_s"] = round(wall, 1)
    config.update({
        "status": "done",
        "detector": det_model.describe(),  # stats now include counters from this session
        "n_predictions": len(preds),
        "label_stats": {f"dropped_{k}": sum(recs[i]["dropped"][k] for i in img_ids)
                        for k in ("non_eval_class", "unknown_label")},
        # predict() time per image incl. model preprocessing/postprocessing, excl. image decode;
        # not a benchmark (see src/eval/benchmark.py), just a record of this run
        "timing": {"predict_ms_mean": round(sum(ms) / len(ms), 1),
                   "predict_s_total": round(sum(ms) / 1000, 1),
                   "wall_clock_s_total": round(sum(s.get("wall_clock_s", 0) for s in config["sessions"]), 1)},
    })
    cfg_path.write_text(json.dumps(config, indent=2))
    partial_path.unlink()
    print(f"wrote {len(preds)} predictions for {len(img_ids)} images -> {pred_path}")
    print(f"label stats: {config['label_stats']} | mean predict ms: {config['timing']['predict_ms_mean']}"
          f" | device: {det_model.device}")


if __name__ == "__main__":
    main()
