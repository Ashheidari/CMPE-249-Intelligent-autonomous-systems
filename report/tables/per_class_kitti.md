# Per-class AP: COCO vs. KITTI

| model | class | AP COCO | AP KITTI | drop | rel. drop | AP50 COCO | AP50 KITTI | GT COCO | GT KITTI |
|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | car | 49.9 | 50.2 | -0.3 | -1% | 76.0 | 85.6 | 1918 | 5687 |
| Faster R-CNN R50-FPN v2 | person | 59.6 | 29.9 | 29.7 | 50% | 86.5 | 65.3 | 10777 | 936 |
| Faster R-CNN R50-FPN v2 | truck | 42.7 | 33.3 | 9.4 | 22% | 63.5 | 55.2 | 414 | 199 |
| YOLO11s | car | 44.8 | 47.3 | -2.5 | -6% | 66.2 | 83.0 | 1918 | 5687 |
| YOLO11s | person | 58.6 | 27.0 | 31.7 | 54% | 81.4 | 60.0 | 10777 | 936 |
| YOLO11s | truck | 35.4 | 20.0 | 15.4 | 44% | 49.4 | 34.0 | 414 | 199 |
| Grounding DINO tiny | car | 50.4 | 48.4 | 2.1 | 4% | 73.6 | 82.1 | 1918 | 5687 |
| Grounding DINO tiny | person | 62.3 | 29.7 | 32.6 | 52% | 85.3 | 66.5 | 10777 | 936 |
| Grounding DINO tiny | truck | 37.0 | 31.3 | 5.8 | 16% | 51.1 | 51.8 | 414 | 199 |

Per-class AP (%) = precision averaged over IoU 0.50:0.95 and recall, area=all, maxDets=100. drop = AP COCO - AP KITTI (AP points); rel. drop = drop / AP COCO. COCO AP here is from the 80-class evaluation; M6 re-evaluates COCO on these 3 classes only.
