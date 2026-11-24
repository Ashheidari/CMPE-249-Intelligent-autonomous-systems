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
        # Resize (min 800 / max 1333) happens inside the model; boxes come back in original px.
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=cfg["box_score_thresh"],
            box_detections_per_img=cfg["box_detections_per_img"],
        ).eval().to(device)
        # PIL -> float tensor [3, H, W] in [0, 1]; normalization is inside the model.
        self.to_tensor = weights.transforms()
        # 91-entry torchvision label list (with N/A gaps): label id -> COCO class name.
        self.categories: list[str] = weights.meta["categories"]

    @torch.no_grad()
    def predict(self, image: Image.Image, class_names: list[str]) -> dict:
        x = self.to_tensor(image.convert("RGB")).to(self.device)  # [3, H, W]
        out = self.model([x])[0]  # boxes [N, 4] xyxy original px, scores [N], labels [N]
        labels = [self.categories[i] for i in out["labels"].tolist()]
        return make_result(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), labels,
                           class_names, self.cfg["box_detections_per_img"])

    # Staged path: GeneralizedRCNN.forward split so the final NMS is in postprocess (RPN NMS stays in forward).
    @torch.no_grad()
    def preprocess(self, image: Image.Image, class_names: list[str]) -> dict:
        x = self.to_tensor(image.convert("RGB")).to(self.device)  # [3, H, W] in [0, 1]
        images, _ = self.model.transform([x])  # resize (min 800 / max 1333) + normalize, batched
        return {"images": images, "orig_size": [tuple(x.shape[-2:])], "class_names": class_names}

    @torch.no_grad()
    def forward(self, inputs: dict):
        m, images = self.model, inputs["images"]
        features = m.backbone(images.tensors)
        proposals, _ = m.rpn(images, features)  # includes proposal NMS
        rh = m.roi_heads
        box_feats = rh.box_head(rh.box_roi_pool(features, proposals, images.image_sizes))
        class_logits, box_regression = rh.box_predictor(box_feats)
        return class_logits, box_regression, proposals

    @torch.no_grad()
    def postprocess(self, outputs, inputs: dict) -> dict:
        class_logits, box_regression, proposals = outputs
        images = inputs["images"]
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(
            class_logits, box_regression, proposals, images.image_sizes)  # decode + score thresh + NMS
        dets = [{"boxes": boxes[0], "scores": scores[0], "labels": labels[0]}]
        out = self.model.transform.postprocess(dets, images.image_sizes, inputs["orig_size"])[0]  # -> original px
        names = [self.categories[i] for i in out["labels"].tolist()]
        return make_result(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), names,
                           inputs["class_names"], self.cfg["box_detections_per_img"])
