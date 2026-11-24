"""Draw detection boxes with label and score."""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def class_color(name: str) -> tuple[int, int, int]:
    """Stable, fairly saturated color per class name."""
    h = hashlib.md5(name.encode()).digest()
    return tuple(60 + b % 180 for b in h[:3])


def draw_detections(image: Image.Image, det: dict, thresh: float = 0.3, title: str | None = None) -> Image.Image:
    """Return a copy of `image` with boxes (xyxy original px) whose score >= thresh drawn on it."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, img.width // 60))
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        if score < thresh:
            continue
        x0, y0, x1, y1 = np.asarray(box, dtype=float).tolist()
        c = class_color(label)
        d.rectangle([x0, y0, x1, y1], outline=c, width=max(2, img.width // 300))
        text = f"{label} {score:.2f}"
        tb = d.textbbox((x0, y0), text, font=font)
        ty = y0 - (tb[3] - tb[1]) - 4 if y0 > tb[3] - tb[1] + 4 else y0
        tb = d.textbbox((x0, ty), text, font=font)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=c)
        d.text((x0, ty), text, fill="black", font=font)
    if title:
        tb = d.textbbox((4, 4), title, font=font)
        d.rectangle([0, 0, tb[2] + 6, tb[3] + 6], fill="white")
        d.text((4, 4), title, fill="black", font=font)
    return img


def dashed_rect(d: ImageDraw.ImageDraw, box: list[float], color, width: int = 2, dash: int = 6) -> None:
    """Draw a dashed rectangle (PIL has no dashed lines)."""
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        length = max(abs(bx - ax), abs(by - ay))
        n = max(1, int(length // dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min(1.0, (i + 1) / n)
            d.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1],
                   fill=color, width=width)


def draw_gt(image: Image.Image, anns: list[dict], cat_names: dict[int, str], title: str | None = None) -> Image.Image:
    """Draw COCO-format GT (bbox xywh). Evaluated boxes solid with class name; ignore
    regions (iscrowd=1) dashed gray with their KITTI type (drawn once even if replicated per class)."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    seen = set()
    for a in anns:
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        if a["iscrowd"]:
            if tuple(box) in seen:
                continue
            seen.add(tuple(box))
            dashed_rect(d, box, (200, 200, 200))
            d.text((x + 2, y + 1), a.get("kitti_type", "ignore"), fill=(230, 230, 230), font=font)
        else:
            c = class_color(cat_names[a["category_id"]])
            d.rectangle(box, outline=c, width=2)
            d.text((x + 2, y - 13 if y > 13 else y + 1), cat_names[a["category_id"]], fill=c, font=font)
    if title:
        tb = d.textbbox((4, 4), title, font=font)
        d.rectangle([0, 0, tb[2] + 6, tb[3] + 6], fill="white")
        d.text((4, 4), title, fill="black", font=font)
    return img
