# COCO val2017: ours vs. published

| model | our COCO AP | published AP | difference | flag (>3 AP) | source |
|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | 46.9 | 46.7 | +0.2 | no | torchvision weights metadata, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 |
| YOLO11s | 45.5 | 47.0 | -1.5 | no | Ultralytics YOLO11 docs, mAP val 50-95 at 640 |
| Grounding DINO tiny | 46.7 | 48.4 | -1.7 | no | IDEA-Research/GroundingDINO model zoo, Swin-T (O365, GoldG, Cap4M), zero-shot |
