"""pycocotools COCOeval (bbox) -> metrics.json: the 12 standard stats plus per-class AP / AP50 / AP75.

Usage: python -m src.eval.coco_eval --run results/<run_id> [--cat-ids 1 3 8] [--out FILE]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..datasets.coco import ROOT, select_image_ids

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def _mean_valid(p: np.ndarray) -> float | None:
    p = p[p > -1]
    return float(p.mean()) if p.size else None


def evaluate(gt: COCO, preds: list[dict], img_ids: list[int], cat_ids: list[int] | None = None) -> dict:
    """COCOeval on img_ids x cat_ids. preds: COCO results with bbox [x, y, w, h]."""
    cat_ids = sorted(cat_ids or gt.getCatIds())
    img_set = set(img_ids)
    preds = [p for p in preds if p["image_id"] in img_set and p["category_id"] in cat_ids]
    # loadRes rejects an empty list; an empty prediction set simply scores zero.
    dt = gt.loadRes(preds) if preds else COCO()
    if not preds:
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()
    ev = COCOeval(gt, dt, iouType="bbox")
    ev.params.imgIds = sorted(img_ids)
    ev.params.catIds = cat_ids
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: [T=10 IoUs, R=101 recall pts, K cats, A=4 areas (all,s,m,l), M=3 maxDets (1,10,100)]
    prec = ev.eval["precision"]
    per_class = {}
    for k, cid in enumerate(cat_ids):
        name = gt.loadCats(cid)[0]["name"]
        n_gt = len(gt.getAnnIds(imgIds=img_ids, catIds=[cid], iscrowd=False))
        per_class[name] = {"category_id": cid, "n_gt": n_gt,
                           "AP": _mean_valid(prec[:, :, k, 0, -1]),
                           "AP50": _mean_valid(prec[0, :, k, 0, -1]),
                           "AP75": _mean_valid(prec[5, :, k, 0, -1]),
                           # AP at each IoU threshold 0.50, 0.55, ..., 0.95 (shows where AP is lost)
                           "AP_by_iou": [_mean_valid(prec[t, :, k, 0, -1]) for t in range(prec.shape[0])]}
    return {"stats": dict(zip(STAT_NAMES, map(float, ev.stats))), "per_class": per_class,
            "n_images": len(img_ids), "cat_ids": cat_ids, "n_predictions": len(preds),
            "summary": buf.getvalue()}


def print_metrics(m: dict) -> None:
    print(m["summary"].rstrip())
    print(f"\n{'class':<16}{'n_gt':>7}{'AP':>8}{'AP50':>8}")
    for name, c in m["per_class"].items():
        fmt = lambda v: f"{100 * v:8.1f}" if v is not None else f"{'-':>8}"  # noqa: E731
        print(f"{name:<16}{c['n_gt']:>7}{fmt(c['AP'])}{fmt(c['AP50'])}")
    s = m["stats"]
    print(f"\nAP {100 * s['AP']:.1f} | AP50 {100 * s['AP50']:.1f} | AP75 {100 * s['AP75']:.1f} | "
          f"{m['n_images']} images, {len(m['cat_ids'])} categories")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="results/<run_id> directory")
    ap.add_argument("--gt", type=Path, help="GT JSON (instead of --run)")
    ap.add_argument("--pred", type=Path, help="predictions JSON (instead of --run)")
    ap.add_argument("--cat-ids", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.run:
        cfg = json.loads((args.run / "config.json").read_text())
        if cfg.get("status") != "done":
            raise SystemExit(f"{args.run} is not finished (status={cfg.get('status')})")
        gt_path, pred_path = ROOT / cfg["gt"], args.run / "predictions.json"
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(gt_path))
        img_ids = select_image_ids(gt, cfg["limit"])
        out = args.out or args.run / "metrics.json"
        meta = {"run_id": cfg["run_id"], "model": cfg["model"], "dataset": cfg["dataset"]}
    elif args.gt and args.pred and args.out:
        gt_path, pred_path = args.gt, args.pred
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(gt_path))
        img_ids = sorted(gt.getImgIds())
        out, meta = args.out, {}
    else:
        raise SystemExit("give --run, or --gt --pred --out")

    preds = json.loads(pred_path.read_text())
    m = evaluate(gt, preds, img_ids, args.cat_ids)
    rel = lambda p: str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p)  # noqa: E731
    m = {**meta, "gt": rel(Path(gt_path)), "predictions": rel(Path(pred_path)), **m}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print_metrics(m)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
