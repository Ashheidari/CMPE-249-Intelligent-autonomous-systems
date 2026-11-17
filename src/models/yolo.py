"""YOLO11s (one-stage, anchor-free) from Ultralytics, COCO-pretrained."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import nms, ops

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

    # ---- staged path for the benchmark: the three steps of Ultralytics' predictor, called directly.
    def _predictor(self, image: Image.Image):
        if self.model.predictor is None:  # first call sets up the predictor with our settings
            self.predict(image, [])
        return self.model.predictor

    def preprocess(self, image: Image.Image, class_names: list[str]) -> dict:
        p = self._predictor(image)
        im0 = np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1])  # BGR, as Ultralytics does for PIL
        x = p.preprocess([im0])  # letterbox to imgsz (minimal padding), to device, [1, 3, h, w] in [0, 1]
        return {"x": x, "orig_shape": im0.shape, "class_names": class_names}

    def forward(self, inputs: dict):
        return self.model.predictor.model(inputs["x"])  # raw head output, before NMS

    def postprocess(self, outputs, inputs: dict) -> dict:
        p = self.model.predictor
        det = nms.non_max_suppression(outputs, self.cfg["conf"], self.cfg["iou"], None, False,
                                      max_det=self.cfg["max_det"], nc=0,
                                      end2end=getattr(p.model, "end2end", False))[0]  # [N, 6] xyxy, conf, cls
        det[:, :4] = ops.scale_boxes(inputs["x"].shape[2:], det[:, :4], inputs["orig_shape"])  # undo letterbox
        labels = [self.names[int(c)] for c in det[:, 5].tolist()]
        return make_result(det[:, :4].cpu().numpy(), det[:, 4].cpu().numpy(), labels,
                           inputs["class_names"], self.cfg["max_det"])
