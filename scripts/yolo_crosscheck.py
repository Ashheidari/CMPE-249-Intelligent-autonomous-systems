"""M3 sanity check: YOLO11s on COCO val2017 through our pipeline vs. Ultralytics' own validator.

Three numbers, so a gap can be attributed:
  The validator is run twice: with its default multi-label NMS and forced to single-label NMS
  (what model.predict does), since that is the main inference difference.
  A. Ultralytics `val` metric (its own AP code, its own dataloader/letterbox/NMS)
  B. our pycocotools evaluator on Ultralytics' predictions.json   (A vs B isolates the metric code)
  C. our full pipeline: results/yolo11s_coco/metrics.json          (B vs C isolates inference)

Ultralytics needs YOLO-format labels, so this builds data/coco_yolo/ once:
  images/val2017 -> symlink to data/coco/val2017
  labels/val2017/*.txt   converted from instances_val2017.json (crowd boxes skipped, 91->80 ids)
  val2017.txt            list of all 5,000 images; the name makes Ultralytics treat it as COCO
                         (category ids mapped back to COCO's 91-id space in predictions.json)
Validator settings match our pipeline: imgsz 640, conf 0.001, NMS IoU 0.7, max_det 100, batch 1
(batch 1 so each image gets its own minimal letterbox, like model.predict).

Usage: .venv/bin/python scripts/yolo_crosscheck.py [--device mps]   (run after the full yolo11s_coco run)
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Ultralytics would otherwise pip-install faster-coco-eval for its own COCO eval; we use pycocotools.
os.environ["YOLO_AUTOINSTALL"] = "False"

import yaml  # noqa: E402

from src.datasets.coco import COCO_GT, DATASETS, load_gt  # noqa: E402
from src.devices import pick_device  # noqa: E402
from src.eval.coco_eval import evaluate  # noqa: E402
from src.models import load_model_cfg  # noqa: E402

YOLO_DATA = ROOT / "data" / "coco_yolo"
OUT = ROOT / "results" / "yolo_crosscheck"


def build_yolo_dataset() -> Path:
    """Create data/coco_yolo/ (see module docstring) and return the dataset YAML path."""
    from ultralytics.data.converter import convert_coco

    data_yaml = YOLO_DATA / "coco_val2017.yaml"
    if (YOLO_DATA / "labels").exists():
        return data_yaml
    (YOLO_DATA / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DATA / "images" / "val2017").symlink_to(DATASETS["coco"]["images"].resolve())
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "annotations"  # convert_coco converts every *.json in the folder
        ann_dir.mkdir()
        (ann_dir / COCO_GT.name).symlink_to(COCO_GT.resolve())
        convert_coco(str(ann_dir), save_dir=str(Path(tmp) / "out"), use_segments=False, cls91to80=True)
        shutil.move(str(Path(tmp) / "out" / "labels"), str(YOLO_DATA / "labels"))
    files = sorted(p.name for p in DATASETS["coco"]["images"].glob("*.jpg"))
    (YOLO_DATA / "val2017.txt").write_text("".join(f"./images/val2017/{f}\n" for f in files))
    from ultralytics import YOLO
    names = YOLO(load_model_cfg("yolo11s")["checkpoint"]).names
    data_yaml.write_text(yaml.safe_dump({"path": str(YOLO_DATA), "train": "val2017.txt",  # train: required key, unused
                                          "val": "val2017.txt", "names": names}))
    return data_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    ours_path = ROOT / "results" / "yolo11s_coco" / "metrics.json"
    if not ours_path.exists():
        raise SystemExit(f"run the full pipeline first: {ours_path} missing")

    from ultralytics import YOLO
    cfg = load_model_cfg("yolo11s")
    data_yaml = build_yolo_dataset()
    device = pick_device(args.device)
    settings = {"imgsz": cfg["imgsz"], "conf": cfg["conf"], "iou": cfg["iou"], "max_det": cfg["max_det"],
                "batch": 1, "device": device}
    gt = load_gt("coco")
    ours = json.loads(ours_path.read_text())
    keys = ("AP", "AP50", "AP75", "APs", "APm", "APl")
    rows: dict[str, dict] = {}
    n_preds: dict[str, int] = {}
    # The validator's NMS uses multi_label=True (a box may be kept for several classes);
    # model.predict, which our pipeline calls, uses multi_label=False. Run both to measure it.
    for multi_label in (True, False):
        tag = "multi_label" if multi_label else "single_label"
        with nms_multi_label(multi_label):
            res = YOLO(cfg["checkpoint"]).val(data=str(data_yaml), save_json=True, plots=False, verbose=False,
                                              project=str(OUT), name=f"ultralytics_val_{tag}", exist_ok=True,
                                              **settings)
        ul_preds = json.loads((Path(res.save_dir) / "predictions.json").read_text())
        b = evaluate(gt, ul_preds, sorted(gt.getImgIds()))
        rows[f"A_ultralytics_metric ({tag} NMS)"] = {"AP": float(res.box.map), "AP50": float(res.box.map50)}
        rows[f"B_pycocotools_on_ultralytics_preds ({tag} NMS)"] = {k: b["stats"][k] for k in keys}
        n_preds[tag] = len(ul_preds)
    rows["C_our_pipeline (model.predict, single_label NMS)"] = {k: ours["stats"][k] for k in keys}
    n_preds["ours"] = ours["n_predictions"]

    c = rows["C_our_pipeline (model.predict, single_label NMS)"]["AP"]
    diffs = {f"C - {k}": c - r["AP"] for k, r in rows.items() if not k.startswith("C_")}
    summary = {"validator_settings": settings, "n_predictions": n_preds, "results": rows, "diff_AP": diffs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# YOLO11s COCO val2017 cross-check (M3)", "",
             f"Validator settings: {settings}", "",
             "| source | AP | AP50 | AP75 | APs | APm | APl |", "|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        lines.append(f"| {k} | " + " | ".join(f"{100 * r[m]:.1f}" if m in r else "-" for m in keys) + " |")
    lines += [""] + [f"- {k}: {100 * v:+.2f} AP" for k, v in diffs.items()]
    (OUT / "crosscheck.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


@contextlib.contextmanager
def nms_multi_label(value: bool):
    """Temporarily force the `multi_label` argument of Ultralytics' NMS (used by the validator)."""
    from ultralytics.utils import nms
    orig = nms.non_max_suppression
    nms.non_max_suppression = functools.wraps(orig)(lambda *a, **k: orig(*a, **{**k, "multi_label": value}))
    try:
        yield
    finally:
        nms.non_max_suppression = orig

if __name__ == "__main__":
    main()
