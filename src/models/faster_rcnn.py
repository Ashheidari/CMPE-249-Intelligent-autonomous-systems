"""Faster R-CNN R50-FPN v2 (two-stage, anchor-based) from torchvision, COCO-pretrained."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from .base import Detector, make_result


class FasterRCNN(Detector):
    name = "faster_rcnn"

    def __init__(self, device: str, cfg: dict[str, Any]) -> None:
        super().__init__(device, cfg)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights[cfg["weights"]]
        # Note: torchvision resizes INSIDE the model (GeneralizedRCNNTransform: min side 800,
        # max side 1333) and maps boxes back, so outputs are already in original-image pixels.
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=cfg["box_score_thresh"],
            box_detections_per_img=cfg["box_detections_per_img"],
        ).eval().to(device)
        # weights.transforms() only converts PIL -> float tensor in [0, 1], shape [3, H, W];
        # mean/std normalization also happens inside the model.
        self.preprocess = weights.transforms()
        # Index into the 91-entry torchvision label space ('__background__' and 'N/A' gaps
        # included), so label id k -> COCO class name.
        self.categories: list[str] = weights.meta["categories"]

    @torch.no_grad()
    def predict(self, image: Image.Image, class_names: list[str]) -> dict:
        x = self.preprocess(image.convert("RGB")).to(self.device)  # [3, H, W]
        out = self.model([x])[0]  # boxes [N, 4] xyxy original px, scores [N], labels [N]
        labels = [self.categories[i] for i in out["labels"].tolist()]
        return make_result(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), labels,
                           class_names, self.cfg["box_detections_per_img"])
