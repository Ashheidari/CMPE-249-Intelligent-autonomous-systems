"""YOLO11s (one-stage, anchor-free) from Ultralytics, COCO-pretrained."""
from __future__ import annotations

from typing import Any

from PIL import Image
from ultralytics import YOLO

from .base import Detector, make_result


class YOLO11(Detector):
    name = "yolo11s"

    def __init__(self, device: str, cfg: dict[str, Any]) -> None:
        super().__init__(device, cfg)
        self.model = YOLO(cfg["checkpoint"])
        self.names: dict[int, str] = self.model.names  # class index -> COCO class name

    def predict(self, image: Image.Image, class_names: list[str]) -> dict:
        # Ultralytics letterboxes to imgsz (long side 640, minimal padding), runs NMS, and
        # scales boxes back, so result.boxes.xyxy is in original-image pixels.
        # A PIL image is treated as RGB and converted to BGR internally.
        r = self.model.predict(image.convert("RGB"), imgsz=self.cfg["imgsz"], conf=self.cfg["conf"],
                               iou=self.cfg["iou"], max_det=self.cfg["max_det"],
                               device=self.device, verbose=False)[0]
        labels = [self.names[int(c)] for c in r.boxes.cls.tolist()]
        return make_result(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), labels,
                           class_names, self.cfg["max_det"])
