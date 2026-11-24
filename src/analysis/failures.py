"""M9: failure-case mining and gallery on the KITTI subset (default label policy).

Usage: .venv/bin/python -m src.analysis.failures

Every detection with score >= DISPLAY (0.3) is classified per image and class (car / person / truck)
with the greedy score-sorted IoU >= 0.5 matcher from M6:
  TP          matched to a GT box of its class
  duplicate   overlaps (IoU > 0.5) a GT box of its class that a higher-scoring detection already took
  ignored     not matched, but inside an ignore region of its class (intersection / detection area
              >= 0.5, as COCOeval does for crowd regions): Van, Cyclist, Tram, Misc or DontCare
  FP          everything else, split into
                wrong class   IoU >= 0.5 with a GT box of another eval class (e.g. "truck" on a car)
                localization  0.1 <= best IoU with a same-class GT < 0.5
                background    no meaningful overlap with any GT box
A high-confidence FP has score > 0.5. A missed GT box has no TP detection at score >= DISPLAY.
Mapping-induced cases: detections on Van / Cyclist ignore regions, and (from a live 80-class
re-run on frames containing cyclists) "bicycle" and "person" boxes on KITTI Cyclist boxes.

Gallery (report/figures/failures/): GT green (ignore regions dashed gray), TP blue, FP and duplicates
red, detections on ignore regions dashed light blue, the box the caption is about drawn thick.
Picks prefer disagreement: a miss that the other models found, an FP the other models did not make.
Writes report/failure_cases.md, report/tables/failure_counts.md, results/analysis/failures.json.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..datasets.coco import DATASETS, ROOT, coco80_names, load_gt
from ..models import build_detector
from ..viz import dashed_rect
from .generalization import iou_xywh
from .tables import MODELS, md_table

RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures" / "failures"
DISPLAY, HIGH_CONF, IOU = 0.3, 0.5, 0.5
CLASSES = {1: "person", 3: "car", 8: "truck"}
GREEN, BLUE, RED, LIGHT_BLUE, GRAY = (40, 200, 80), (60, 130, 255), (235, 50, 50), (150, 200, 255), (200, 200, 200)


def ioa(det: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Intersection over detection area, [D, R], boxes COCO xywh."""
    iou = iou_xywh(det, regions)
    da = (det[:, 2] * det[:, 3])[:, None]
    ra = regions[:, 2] * regions[:, 3]
    inter = iou * (da + ra) / (1 + iou)  # invert iou = I / (A + B - I)
    return inter / np.maximum(da, 1e-9)


def classify_image(anns: list[dict], dets: list[dict]) -> tuple[list[dict], list[dict]]:
    """Classify dets (score >= DISPLAY) of one image; return (detections with 'kind', GT with 'found')."""
    gts = [dict(a, found=False) for a in anns if not a["iscrowd"]]
    out = []
    for cid in CLASSES:
        g = [a for a in gts if a["category_id"] == cid]
        ign = [a for a in anns if a["iscrowd"] and a["category_id"] == cid]
        d = sorted([x for x in dets if x["category_id"] == cid], key=lambda x: -x["score"])
        gb = np.array([a["bbox"] for a in g], dtype=float).reshape(-1, 4)
        for x in d:
            box = np.array([x["bbox"]], dtype=float)
            iou = iou_xywh(box, gb)[0] if len(g) else np.zeros(0)
            free = np.array([not a["found"] for a in g], dtype=bool)
            rec = dict(x, kind="FP", best_iou=float(iou.max()) if len(iou) else 0.0)
            if len(iou) and (iou * free).max() >= IOU:
                k = int(np.argmax(np.where(free, iou, -1)))
                g[k]["found"] = True
                rec.update(kind="TP", gt=g[k])
            elif len(iou) and iou.max() > IOU:
                rec.update(kind="duplicate", gt=g[int(iou.argmax())])
            elif ign and (r := ioa(box, np.array([a["bbox"] for a in ign], dtype=float))[0]).max() >= 0.5:
                rec.update(kind="ignored", region=ign[int(r.argmax())]["kitti_type"])
            else:
                other = [a for a in gts if a["category_id"] != cid]
                oi = iou_xywh(box, np.array([a["bbox"] for a in other], dtype=float))[0] if other else np.zeros(0)
                if len(oi) and oi.max() >= IOU:
                    rec.update(fp_type="wrong class", gt=other[int(oi.argmax())])
                elif rec["best_iou"] >= 0.1:
                    rec.update(fp_type="localization", gt=g[int(iou.argmax())])
                else:
                    rec["fp_type"] = "background"
            out.append(rec)
    return out, gts


def size_occ(a: dict) -> str:
    occ = {0: "fully visible", 1: "partly occluded", 2: "largely occluded", 3: "occlusion unknown"}[a["occluded"]]
    return f"{a['bbox'][3]:.0f} px tall, {occ}, truncation {a['truncated']:.2f}"


# ------------------------------------------------------------------------------------------ mining
def mine(gt) -> tuple[dict, dict]:
    """Per model and image: classified detections and GT found flags."""
    per = {}
    for m in MODELS:
        preds = json.loads((RESULTS / f"{m}_kitti" / "predictions.json").read_text())
        by_img = defaultdict(list)
        for p in preds:
            if p["score"] >= DISPLAY:
                by_img[p["image_id"]].append(p)
        per[m] = {i: classify_image(gt.loadAnns(gt.getAnnIds(imgIds=i)), by_img[i]) for i in gt.getImgIds()}
    counts = {}
    for m, imgs in per.items():
        c = Counter()
        for dets, gts in imgs.values():
            for d in dets:
                c[d["kind"]] += 1
                if d["kind"] == "FP":
                    c[f"FP: {d['fp_type']}"] += 1
                    c["FP score > 0.5"] += d["score"] > HIGH_CONF
                if d["kind"] == "ignored":
                    c[f"on {d['region']} region"] += 1
                    if d["region"] == "Van":
                        c[f"on Van region as {CLASSES[d['category_id']]}"] += 1
            for a in gts:
                if not a["found"]:
                    c["missed GT"] += 1
                    c["missed: small (<32² px)"] += a["area"] < 32 ** 2
                    c["missed: occluded (level 1-2)"] += a["occluded"] in (1, 2)
            c["GT total"] += len(gts)
        counts[m] = dict(c)
    return per, counts


def cyclist_check(gt) -> dict:
    """Live 80-class re-run on frames with KITTI Cyclist boxes: how often each model puts a 'bicycle'
    (inside the cyclist box) and/or a 'person' (IoU >= 0.3) on a cyclist, at score >= DISPLAY."""
    first = gt.getCatIds()[0]
    cyc = defaultdict(list)
    for a in gt.loadAnns(gt.getAnnIds(iscrowd=True)):
        if a["kitti_type"] == "Cyclist" and a["category_id"] == first:
            cyc[a["image_id"]].append(a["bbox"])
    names = coco80_names()
    root = DATASETS["kitti"]["images"]
    out, live = {}, {}
    for m in MODELS:
        det = build_detector(m)
        c = Counter()
        live[m] = {}
        for img_id, boxes in cyc.items():
            img = Image.open(root / gt.loadImgs(img_id)[0]["file_name"]).convert("RGB")
            r = det.predict(img, names)
            keep = r["scores"] >= DISPLAY
            xy = r["boxes"][keep]
            xywh = np.c_[xy[:, :2], xy[:, 2:] - xy[:, :2]]
            labels = [lab for lab, k in zip(r["labels"], keep) if k]
            live[m][img_id] = {"boxes": xywh.tolist(), "scores": r["scores"][keep].tolist(), "labels": labels}
            cb = np.array(boxes, dtype=float)
            bike = np.array([lab == "bicycle" for lab in labels], dtype=bool)
            pers = np.array([lab == "person" for lab in labels], dtype=bool)
            # bicycle: >= 70% of the bicycle box inside the cyclist box; person: IoU >= 0.3 with it
            b_hit = (ioa(xywh[bike], cb) >= 0.7).any(0) if bike.any() else np.zeros(len(cb), bool)
            p_hit = (iou_xywh(xywh[pers], cb) >= 0.3).any(0) if pers.any() else np.zeros(len(cb), bool)
            c["cyclists with a bicycle box"] += int(b_hit.sum())
            c["cyclists with a person box"] += int(p_hit.sum())
            c["cyclists with person + bicycle"] += int((b_hit & p_hit).sum())
        c["cyclist boxes"] = sum(len(b) for b in cyc.values())
        out[m] = dict(c)
        del det
    return {"counts": out, "live": live, "images": sorted(cyc)}


# ------------------------------------------------------------------------------------------ drawing
def render(img: Image.Image, anns: list[dict], dets: list[dict], focus: dict | None, title: str,
           extra: list[tuple[list[float], str]] = ()) -> Image.Image:
    """Draw GT + classified detections on a copy of img; `focus` (a det or GT dict) is drawn thick."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=13)
    xyxy = lambda b: [b[0], b[1], b[0] + b[2], b[1] + b[3]]  # noqa: E731
    seen = set()
    for a in anns:
        if a["iscrowd"]:
            if tuple(a["bbox"]) not in seen:
                seen.add(tuple(a["bbox"]))
                dashed_rect(d, xyxy(a["bbox"]), GRAY, width=1)
                d.text((a["bbox"][0] + 2, a["bbox"][1] + 1), a["kitti_type"], fill=GRAY, font=font)
        else:
            is_focus = focus is not None and "kind" not in focus and focus["bbox"] == a["bbox"]  # a focused GT box
            d.rectangle(xyxy(a["bbox"]), outline=GREEN, width=4 if is_focus else 2)
    for x in dets:
        b, color = xyxy(x["bbox"]), {"TP": BLUE, "ignored": LIGHT_BLUE}.get(x["kind"], RED)
        w = 5 if focus is x else 2
        if x["kind"] == "ignored":
            dashed_rect(d, b, color, width=w)
        else:
            d.rectangle(b, outline=color, width=w)
        if x["kind"] != "TP" or focus is x:
            tag = {"duplicate": " dup"}.get(x["kind"], "")
            d.text((b[0] + 2, max(0, b[1] - 15)), f"{CLASSES[x['category_id']]} {x['score']:.2f}{tag}", fill=color, font=font)
    for box, label in extra:
        dashed_rect(d, xyxy(box), (255, 170, 0), width=3)
        d.text((box[0] + 2, box[1] + box[3] + 1), label, fill=(255, 170, 0), font=font)
    if focus is not None:  # zoomed crop of the case, to the right of the frame
        x, y, w, h = focus["bbox"]
        m = max(25.0, 0.6 * max(w, h))
        box = [max(0, x - m), max(0, y - m), min(im.width, x + w + m), min(im.height, y + h + m)]
        crop = im.crop([int(v) for v in box])
        s = min(4.0, im.height / crop.height)
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
        ImageDraw.Draw(im).rectangle(box, outline=(255, 255, 255), width=1)
        wide = Image.new("RGB", (im.width + crop.width + 6, im.height), (20, 20, 20))
        wide.paste(im, (0, 0))
        wide.paste(crop, (im.width + 6, 0))
        ImageDraw.Draw(wide).text((im.width + 10, 4), f"zoom x{s:.1f}", fill=(255, 255, 255), font=font)
        im = wide
    strip = Image.new("RGB", (im.width, 24), (20, 20, 20))
    ImageDraw.Draw(strip).text((6, 5), title, fill=(240, 240, 240), font=font)
    out = Image.new("RGB", (im.width, im.height + 24))
    out.paste(strip, (0, 0))
    out.paste(im, (0, 24))
    return out


# ------------------------------------------------------------------------------------------ picking
HYP = {
    "wrong class": "COCO's '{det}' and KITTI's '{gt}' draw the class boundary differently (e.g. pickups, SUVs, vans).",
    "localization": "The box covers only the visible part, or several objects. KITTI boxes of occluded objects "
                    "appear to span the full object extent (e.g. frame 000112), which a visible-part box cannot match.",
    "background": "Unlabelled object or clutter; possibly a real object without a KITTI label.",
    "missed": "Hard instance ({attr}); the other models found it, so the cue is visible. If occluded, the GT box "
              "may span the hidden part too, so a visible-part box falls below IoU 0.5.",
    "duplicate": "Two boxes on one object survive NMS ({why}).",
    "van": "COCO has no van class, so the model calls a van '{det}'. Neutral under Van: ignore, an FP under Van: car.",
    "cyclist": "COCO splits a cyclist into a person and a bicycle; KITTI labels one Cyclist box, so neither matches.",
}
# Notes from looking at the rendered gallery images (frame id -> observation). They describe what is
# visible in that frame only; they are not general claims.
NOTES = {
    1755: "The 0.99 'person' sits right beside a KITTI Cyclist ignore region and appears to be part of that "
          "cyclist, but less than half of the box lies inside the region, so it counts as an FP.",
    1045: "KITTI labels this box truck as Van; COCO's 'truck' is arguably the better name for it.",
    112: "Parked cars with occlusion level 2: their KITTI boxes span the full car behind the cars in front, "
         "while the detectors box the visible part, giving IoU < 0.5 (red) plus a missed GT (green). The white "
         "van on the right is labelled Car in this frame.",
    6139: "Also visible: the two nearest pedestrians on the right have GT boxes much wider than the people, "
          "while the detections fit the people tightly (the M6 person-localization effect).",
    6689: "The duplicate is one YOLO11s box spanning two adjacent distant cars.",
}
DUP_WHY = {"faster_rcnn": "class-wise NMS at IoU 0.5 keeps boxes that overlap less than that",
           "yolo11s": "NMS IoU 0.7 is permissive",
           "gdino": "several queries claim the same object; no NMS within one prompt"}


def pick(per: dict, cyc: dict, gt) -> list[dict]:
    used, picks = set(), []

    def take(item):
        used.add(item["image_id"])
        picks.append(item)

    # 1. unique high-confidence FP, one per model
    for m in MODELS:
        cands = []
        for i, (dets, _) in per[m].items():
            for x in dets:
                if x["kind"] == "FP" and x["score"] > HIGH_CONF and i not in used:
                    others = [y for o in MODELS if o != m for y in per[o][i][0] if y["category_id"] == x["category_id"]]
                    if not others or iou_xywh(np.array([x["bbox"]], float), np.array([y["bbox"] for y in others], float)).max() < 0.5:
                        cands.append((x["score"], i, x))
        if cands:
            s, i, x = max(cands, key=lambda t: t[0])
            hyp = HYP[x["fp_type"]].format(det=CLASSES[x["category_id"]], gt=CLASSES.get(x.get("gt", {}).get("category_id"), ""))
            take({"type": f"high-confidence FP ({x['fp_type']})", "model": m, "image_id": i, "focus": x,
                  "what": f"{MODELS[m]} predicts {CLASSES[x['category_id']]} at {x['score']:.2f}; "
                          f"the other models have no such box", "hyp": hyp})
    # 2. missed GT that both other models found, small or occluded preferred
    for m in MODELS:
        cands = []
        for i, (_, gts) in per[m].items():
            if i in used:
                continue
            for k, a in enumerate(gts):
                if not a["found"] and all(per[o][i][1][k]["found"] for o in MODELS if o != m):
                    hard = (a["area"] < 32 ** 2) + (a["occluded"] in (1, 2))
                    cands.append((hard, a["area"], i, a))
        if cands:
            _, _, i, a = max(cands, key=lambda t: (t[0], t[1]))
            take({"type": "missed object", "model": m, "image_id": i, "focus": a,
                  "what": f"{MODELS[m]} misses a {CLASSES[a['category_id']]} ({size_occ(a)}) that the other two find",
                  "hyp": HYP["missed"].format(attr=size_occ(a))})
    # 3. duplicates: the two images with the most duplicate boxes for one model
    dups = sorted(((sum(x["kind"] == "duplicate" for x in dets), m, i) for m in MODELS
                   for i, (dets, _) in per[m].items() if i not in used), reverse=True)
    for n, m, i in dups[:2]:
        if i in used:
            continue
        x = max((y for y in per[m][i][0] if y["kind"] == "duplicate"), key=lambda y: y["score"])
        take({"type": "duplicate detections", "model": m, "image_id": i, "focus": x,
              "what": f"{MODELS[m]}: {n} duplicate boxes (score >= {DISPLAY}) on already-matched objects",
              "hyp": HYP["duplicate"].format(why=DUP_WHY[m])})
    # 4a. mapping: a confident detection on a Van ignore region, preferring 'truck'
    vans = [(x["category_id"] == 8, x["score"], m, i, x) for m in MODELS for i, (dets, _) in per[m].items()
            for x in dets if x["kind"] == "ignored" and x["region"] == "Van" and i not in used]
    if vans:
        _, s, m, i, x = max(vans, key=lambda t: (t[0], t[1]))
        take({"type": "mapping: detection on a Van", "model": m, "image_id": i, "focus": x,
              "what": f"{MODELS[m]} calls a KITTI Van '{CLASSES[x['category_id']]}' ({x['score']:.2f}); "
                      "neutral because Van is an ignore region",
              "hyp": HYP["van"].format(det=CLASSES[x["category_id"]])})
    # 4b. mapping: bicycle + person on a KITTI Cyclist (live 80-class predictions)
    best = None
    for m in MODELS:
        for i in cyc["images"]:
            if i in used:
                continue
            lv = cyc["live"][m][i]
            n_bike = sum(lab == "bicycle" for lab in lv["labels"])
            n_pers = sum(lab == "person" for lab in lv["labels"])
            if n_bike and n_pers and (best is None or n_bike + n_pers > best[0]):
                best = (n_bike + n_pers, m, i)
    if best:
        _, m, i = best
        # focus on the cyclist box with the most person/bicycle boxes on it
        first = gt.getCatIds()[0]
        cbs = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=True))
               if a["kitti_type"] == "Cyclist" and a["category_id"] == first]
        lv = cyc["live"][m][i]
        pb = np.array([b for b, lab in zip(lv["boxes"], lv["labels"]) if lab in ("bicycle", "person")], float)
        focus = max(cbs, key=lambda a: int((ioa(pb, np.array([a["bbox"]], float))[:, 0] >= 0.7).sum()))
        take({"type": "mapping: cyclist", "model": m, "image_id": i, "focus": focus, "live": True,
              "what": f"{MODELS[m]} reports a 'person' and a 'bicycle' where KITTI has one Cyclist box "
                      "(orange dashed: 80-class output on that cyclist)", "hyp": HYP["cyclist"]})
    return picks


def disagreement(per: dict, i: int) -> int:
    """GT boxes found by some but not all models + high-confidence FPs, for side-by-side picks."""
    n_gt = len(per[next(iter(MODELS))][i][1])
    found = [[per[m][i][1][k]["found"] for m in MODELS] for k in range(n_gt)]
    split = sum(0 < sum(f) < len(MODELS) for f in found)
    fps = sum(x["kind"] == "FP" and x["score"] > HIGH_CONF for m in MODELS for x in per[m][i][0])
    return split + fps


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    gt = load_gt("kitti")
    root = DATASETS["kitti"]["images"]
    per, counts = mine(gt)
    cyc = cyclist_check(gt)
    picks = pick(per, cyc, gt)
    FIG.mkdir(parents=True, exist_ok=True)
    for old in FIG.glob("*.png"):
        old.unlink()

    entries = []
    for n, p in enumerate(picks, 1):
        i, m = p["image_id"], p["model"]
        img = Image.open(root / gt.loadImgs(i)[0]["file_name"]).convert("RGB")
        anns = gt.loadAnns(gt.getAnnIds(imgIds=i))
        extra = []
        if p.get("live"):
            lv = cyc["live"][m][i]
            fb = np.array([p["focus"]["bbox"]], float)
            extra = [(b, f"{lab} {s:.2f}") for b, lab, s in zip(lv["boxes"], lv["labels"], lv["scores"])
                     if lab in ("bicycle", "person") and ioa(np.array([b], float), fb)[0, 0] >= 0.7]
        name = f"{n:02d}_{p['type'].split(' (')[0].replace(':', '').replace(' ', '_')}_{m}_{i:06d}.png"
        render(img, anns, per[m][i][0], p["focus"], f"{MODELS[m]} | frame {i:06d} | {p['type']}", extra).save(FIG / name)
        entries.append({**{k: v for k, v in p.items() if k not in ("focus", "live")}, "file": name})

    side = sorted((i for i in gt.getImgIds() if i not in {p["image_id"] for p in picks}),
                  key=lambda i: -disagreement(per, i))[:3]
    for n, i in enumerate(side, len(picks) + 1):
        img = Image.open(root / gt.loadImgs(i)[0]["file_name"]).convert("RGB")
        anns = gt.loadAnns(gt.getAnnIds(imgIds=i))
        panels = []
        for m in MODELS:
            dets, gts = per[m][i]
            tp = sum(x["kind"] == "TP" for x in dets)
            fp = sum(x["kind"] in ("FP", "duplicate") for x in dets)
            miss = sum(not a["found"] for a in gts)
            panels.append(render(img, anns, dets, None, f"{MODELS[m]} | frame {i:06d} | TP {tp}  FP {fp}  missed {miss}"))
        canvas = Image.new("RGB", (img.width, sum(p.height for p in panels)))
        y = 0
        for p in panels:
            canvas.paste(p, (0, y))
            y += p.height
        name = f"{n:02d}_side_by_side_{i:06d}.png"
        canvas.save(FIG / name)
        stats = {m: {"TP": sum(x["kind"] == "TP" for x in per[m][i][0]),
                     "FP": sum(x["kind"] in ("FP", "duplicate") for x in per[m][i][0]),
                     "missed": sum(not a["found"] for a in per[m][i][1])} for m in MODELS}
        entries.append({"type": "side by side", "model": "all", "image_id": i, "file": name, "stats": stats,
                        "what": "; ".join(f"{MODELS[m]} TP {s['TP']} / FP {s['FP']} / missed {s['missed']}"
                                          for m, s in stats.items()),
                        "hyp": "TODO(user): what the models disagree on in this frame."})

    out = {"counts": counts, "cyclist_check": cyc["counts"], "gallery": entries}
    (RESULTS / "analysis" / "failures.json").write_text(json.dumps(out, indent=2))
    write_md(counts, cyc["counts"], entries)
    print((ROOT / "report" / "tables" / "failure_counts.md").read_text())
    for e in entries:
        print(f"{e['file']}: {e['what']}")


def summary(counts: dict, cyc: dict) -> list[str]:
    """A few count-based bullets (numbers only; interpretation left to the user)."""
    out = []
    for m, c in counts.items():
        fp = c.get("FP", 0)
        out.append(f"- **{MODELS[m]}:** {c.get('missed GT', 0)} of {c['GT total']} GT boxes missed "
                   f"({c.get('missed: occluded (level 1-2)', 0)} of them occluded, {c.get('missed: small (<32² px)', 0)} "
                   f"small); {fp} FPs ({c.get('FP score > 0.5', 0)} above 0.5): "
                   f"{100 * c.get('FP: background', 0) / max(fp, 1):.0f}% background, "
                   f"{100 * c.get('FP: localization', 0) / max(fp, 1):.0f}% localization, "
                   f"{100 * c.get('FP: wrong class', 0) / max(fp, 1):.0f}% wrong class; "
                   f"{c.get('ignored', 0)} detections absorbed by ignore regions "
                   f"({c.get('on DontCare region', 0)} DontCare, {c.get('on Van region', 0)} Van, "
                   f"{c.get('on Cyclist region', 0)} Cyclist).")
    out.append("- **Cyclists (80-class re-run):** " + "; ".join(
        f"{MODELS[m]} puts both a person and a bicycle on {v.get('cyclists with person + bicycle', 0)} of "
        f"{v['cyclist boxes']}" for m, v in cyc.items()) + ".")
    out.append("- TODO(user): Faster R-CNN's FP count at score >= 0.3 is about 5x YOLO11s's, while its KITTI AP is only "
               "6.4 points higher. Is that a calibration difference (score scale), and does it matter for your use case?")
    return out


def write_md(counts: dict, cyc: dict, entries: list[dict]) -> None:
    keys = ["GT total", "TP", "missed GT", "missed: small (<32² px)", "missed: occluded (level 1-2)", "FP",
            "FP score > 0.5", "FP: wrong class", "FP: localization", "FP: background", "duplicate", "ignored",
            "on Van region", "on Van region as truck", "on Van region as car", "on Cyclist region", "on DontCare region"]
    rows = [[k, *[str(counts[m].get(k, 0)) for m in MODELS]] for k in keys]
    ck = ["cyclist boxes", "cyclists with a person box", "cyclists with a bicycle box", "cyclists with person + bicycle"]
    rows += [[k + " (80-class re-run)", *[str(cyc[m].get(k, 0)) for m in MODELS]] for k in ck]
    (ROOT / "report" / "tables" / "failure_counts.md").write_text(
        f"# Failure counts on the KITTI subset (detections with score >= {DISPLAY}, IoU 0.5)\n\n"
        + md_table(["count", *MODELS.values()], rows)
        + "\n\nSee src/analysis/failures.py for the definitions. 'ignored' = inside an ignore region of its class "
        "(intersection / detection area >= 0.5); such detections do not count as FP in COCOeval either. "
        "Cyclist rows come from a re-run with all 80 COCO classes kept (the saved KITTI predictions keep only "
        "car/person/truck): person = IoU >= 0.3 with the cyclist box, bicycle = >= 70% of the bicycle box inside it.\n")
    lines = ["# Failure cases (KITTI subset)", "",
             "Counts: [tables/failure_counts.md](tables/failure_counts.md). Images: `report/figures/failures/`. "
             f"Legend: green = GT, gray dashed = ignore region (KITTI type written on it), blue = correct detection, "
             f"red = false positive or duplicate, light blue dashed = detection inside an ignore region (neutral), "
             f"thick box = the case described. Detections shown at score >= {DISPLAY}. Generated by "
             "`python -m src.analysis.failures`.", "",
             "Each hypothesis is a first guess for you to check and rewrite. TODO(user) marks where your "
             "explanation is needed.", "", "## Summary of the counts", ""] + summary(counts, cyc) + [""]
    for e in entries:
        lines += [f"## {e['file'][:2]}. {e['type'][0].upper() + e['type'][1:]}: {MODELS.get(e['model'], 'all models')}, "
                  f"frame {e['image_id']:06d}", "", f"![{e['type']}](figures/failures/{e['file']})", "",
                  f"- **What went wrong:** {e['what']}."]
        if e["image_id"] in NOTES:
            lines.append(f"- **Observed in the image:** {NOTES[e['image_id']]}")
        lines += [f"- **Hypothesis:** {e['hyp']} TODO(user)", ""]
    (ROOT / "report" / "failure_cases.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
