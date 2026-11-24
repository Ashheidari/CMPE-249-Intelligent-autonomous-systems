"""Analysis 1 (COCO -> KITTI): shared-class COCO eval, recall by occlusion/truncation, label-policy re-scoring.

Uses saved predictions only. Writes results/analysis/generalization.json and report/tables/generalization_*.md.
"""
from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from ..datasets.coco import ROOT, load_gt, select_image_ids
from ..eval.coco_eval import evaluate
from .tables import MODELS, md_table

RESULTS = ROOT / "results"
TABLES = ROOT / "report" / "tables"
SHARED_IDS = {"person": 1, "car": 3, "truck": 8}
SCORE_THRESHOLDS = [0.001, 0.3]
IOU_MATCH = 0.5
# KITTI's own difficulty levels cap truncation at 0.15 (easy), 0.30 (moderate), 0.50 (hard).
TRUNC_BUCKETS = [("0-0.15", 0.0, 0.15), ("0.15-0.3", 0.15, 0.30), ("0.3-0.5", 0.30, 0.50), (">0.5", 0.50, 1.01)]
OCC_LEVELS = {0: "0 fully visible", 1: "1 partly occluded", 2: "2 largely occluded", 3: "3 unknown"}
VARIANTS = {"van_car": "Van: car", "cyclist_person": "Cyclist: person"}


def quiet_coco(path: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def load_preds(model: str, dataset: str) -> list[dict]:
    return json.loads((RESULTS / f"{model}_{dataset}" / "predictions.json").read_text())


# ---------------------------------------------------------------- 1. shared-class COCO evaluation
def shared_class_eval(model: str, coco_gt: COCO) -> dict:
    m = evaluate(coco_gt, load_preds(model, "coco"), select_image_ids(coco_gt), list(SHARED_IDS.values()))
    out = {"run_id": f"{model}_coco", "model": model, "dataset": "coco", "note": "catIds restricted to 1, 3, 8", **m}
    (RESULTS / f"{model}_coco" / "metrics_shared3.json").write_text(json.dumps(out, indent=2))
    return m


# ---------------------------------------------------------------- 2. occlusion / truncation recall
def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a [N, 4] and b [M, 4], both COCO [x, y, w, h]."""
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1) - np.maximum(ax0[:, None], bx0), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1) - np.maximum(ay0[:, None], by0), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(gt_boxes: np.ndarray, det_boxes: np.ndarray, det_scores: np.ndarray,
                 iou_thr: float = IOU_MATCH) -> np.ndarray:
    """Greedy score-sorted one-to-one matching (IoU >= iou_thr) for one image and class; returns found flags [G]."""
    found = np.zeros(len(gt_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return found
    iou = iou_xywh(det_boxes[np.argsort(-det_scores, kind="stable")], gt_boxes)  # [D, G]
    for row in iou:
        row = np.where(found, -1.0, row)
        g = int(row.argmax())
        if row[g] >= iou_thr:
            found[g] = True
    return found


def recall_by_attribute(kitti_gt: COCO, preds: list[dict], score_thr: float) -> list[dict]:
    """Per GT box (car / person, non-crowd): was it found at IoU >= 0.5 by a detection >= score_thr?"""
    by_img_cat = defaultdict(list)
    for p in preds:
        if p["score"] >= score_thr:
            by_img_cat[(p["image_id"], p["category_id"])].append(p)
    rows = []
    for name in ("car", "person"):
        cid = SHARED_IDS[name]
        for img_id in kitti_gt.getImgIds():
            anns = kitti_gt.loadAnns(kitti_gt.getAnnIds(imgIds=img_id, catIds=[cid], iscrowd=False))
            if not anns:
                continue
            dets = by_img_cat[(img_id, cid)]
            found = greedy_match(np.array([a["bbox"] for a in anns], dtype=float),
                                 np.array([d["bbox"] for d in dets], dtype=float).reshape(-1, 4),
                                 np.array([d["score"] for d in dets], dtype=float))
            rows += [{"class": name, "occluded": a["occluded"], "truncated": a["truncated"],
                      "height": a["bbox"][3], "found": bool(f)} for a, f in zip(anns, found)]
    return rows


def summarize_recall(rows: list[dict]) -> dict:
    """Recall, count and median box height (px) per class x occlusion level and x truncation bucket."""
    out: dict = {}
    for name in ("car", "person"):
        r = [x for x in rows if x["class"] == name]
        groups = {f"occ {OCC_LEVELS[k]}": [x for x in r if x["occluded"] == k] for k in OCC_LEVELS}
        groups |= {f"trunc {lab}": [x for x in r if lo <= x["truncated"] < hi] for lab, lo, hi in TRUNC_BUCKETS}
        groups["all"] = r
        out[name] = {g: {"n": len(v), "recall": float(np.mean([x["found"] for x in v])) if v else None,
                         "median_height_px": float(np.median([x["height"] for x in v])) if v else None}
                     for g, v in groups.items()}
    return out


# ---------------------------------------------------------------- 3. label-policy sensitivity
def label_policy(model: str, kitti_ids: list[int], variants_gt: dict[str, COCO], base: dict) -> dict:
    preds = load_preds(model, "kitti")
    res = {"ignore (default)": base}
    for tag, gt in variants_gt.items():
        res[VARIANTS[tag]] = evaluate(gt, preds, kitti_ids)
    return {k: {"AP": v["stats"]["AP"], **{c: v["per_class"][c]["AP"] for c in SHARED_IDS},
                "n_gt": {c: v["per_class"][c]["n_gt"] for c in SHARED_IDS}} for k, v in res.items()}


# ---------------------------------------------------------------- tables
def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def write_tables(res: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    keys = ["AP", "AP50", "APs", "APm", "APl"]
    rows = []
    for m in MODELS:
        c, k = res[m]["coco_shared3"]["stats"], res[m]["kitti"]["stats"]
        rows.append([MODELS[m], *[f"{pct(c[x])} / {pct(k[x])}" for x in keys],
                     f"{100 * (c['AP'] - k['AP']):.1f}", f"{100 * (c['AP'] - k['AP']) / c['AP']:.0f}%"])
    size_md = md_table(["model", *[f"{x} (COCO / KITTI)" for x in keys], "AP drop", "rel. drop"], rows)

    rec_rows = []
    for m in MODELS:
        for thr in SCORE_THRESHOLDS:
            s = res[m]["recall"][str(thr)]
            for name in ("car", "person"):
                rec_rows.append([MODELS[m], f"{thr:g}", name] + [pct(s[name][g]["recall"]) for g in s[name]])
    groups = list(res[next(iter(MODELS))]["recall"][str(SCORE_THRESHOLDS[0])]["car"])
    counts = res[next(iter(MODELS))]["recall"][str(SCORE_THRESHOLDS[0])]
    count_rows = [[name, "GT boxes"] + [str(counts[name][g]["n"]) for g in groups] for name in ("car", "person")]
    count_rows += [[name, "median height (px)"] + [f"{counts[name][g]['median_height_px']:.0f}"
                                                  if counts[name][g]["n"] else "-" for g in groups]
                   for name in ("car", "person")]

    pol_rows = []
    for m in MODELS:
        base = res[m]["label_policy"]["ignore (default)"]
        for pol, v in res[m]["label_policy"].items():
            pol_rows.append([MODELS[m], pol, pct(v["AP"]), pct(v["car"]), pct(v["person"]), pct(v["truck"]),
                             f"{100 * (v['car'] - base['car']):+.1f}", f"{100 * (v['person'] - base['person']):+.1f}",
                             f"{100 * (v['truck'] - base['truck']):+.1f}"])
    n_gt = {p: v["n_gt"] for p, v in res[next(iter(MODELS))]["label_policy"].items()}

    (TABLES / "generalization_size.md").write_text(
        "# COCO vs. KITTI on the shared classes (car, person, truck)\n\n" + size_md +
        "\n\nCOCO is re-evaluated with catIds = [1, 3, 8] so both columns cover the same 3 classes. "
        "Size buckets are COCO's (area < 32², < 96², else large). AP in %.\n")
    (TABLES / "generalization_occlusion.md").write_text(
        "# KITTI recall at IoU >= 0.5 by occlusion and truncation\n\n"
        "Greedy score-sorted matching per image and class; a GT box counts as found if a same-class detection "
        "with score >= threshold overlaps it at IoU >= 0.5 and was not already used. Recall in %.\n\n"
        + md_table(["model", "score >=", "class", *groups], rec_rows) + "\n\n## GT counts and box size per group\n\n"
        + md_table(["class", "", *groups], count_rows) + "\n")
    (TABLES / "generalization_label_policy.md").write_text(
        "# KITTI label-policy sensitivity (same predictions, different GT)\n\n"
        + md_table(["model", "policy", "AP (3 cls)", "car AP", "person AP", "truck AP", "Δ car", "Δ person", "Δ truck"], pol_rows)
        + "\n\nGT boxes (car / person / truck): "
        + "; ".join(f"{p}: {v['car']} / {v['person']} / {v['truck']}" for p, v in n_gt.items())
        + ". When a type stops being an ignore region, detections on it are no longer neutral for the other "
        "classes either (e.g. a 'truck' box on a van becomes a truck false positive under Van: car).\n")


def main() -> None:
    coco_gt = load_gt("coco")
    kitti_gt = load_gt("kitti")
    kitti_ids = select_image_ids(kitti_gt)
    variants_gt = {t: quiet_coco(ROOT / "data" / "kitti" / "annotations" / f"kitti_subset_coco_{t}.json")
                   for t in VARIANTS}
    res = {}
    for m in MODELS:
        print(f"== {m}")
        kitti_metrics = json.loads((RESULTS / f"{m}_kitti" / "metrics.json").read_text())
        preds = load_preds(m, "kitti")
        res[m] = {
            "coco_shared3": shared_class_eval(m, coco_gt),
            "kitti": {k: kitti_metrics[k] for k in ("stats", "per_class")},
            "recall": {str(t): summarize_recall(recall_by_attribute(kitti_gt, preds, t)) for t in SCORE_THRESHOLDS},
            "label_policy": label_policy(m, kitti_ids, variants_gt, kitti_metrics),
        }
        res[m]["coco_shared3"] = {x: res[m]["coco_shared3"][x] for x in ("stats", "per_class")}
    out = RESULTS / "analysis" / "generalization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    write_tables(res)
    for f in ("generalization_size.md", "generalization_occlusion.md", "generalization_label_policy.md"):
        print((TABLES / f).read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
