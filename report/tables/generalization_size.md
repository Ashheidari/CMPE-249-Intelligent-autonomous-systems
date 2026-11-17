# COCO vs. KITTI on the shared classes (car, person, truck)

| model | AP (COCO / KITTI) | AP50 (COCO / KITTI) | APs (COCO / KITTI) | APm (COCO / KITTI) | APl (COCO / KITTI) | AP drop | rel. drop |
|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | 50.8 / 37.8 | 75.3 / 68.7 | 36.3 / 25.2 | 57.9 / 35.9 | 67.4 / 47.8 | 12.9 | 26% |
| YOLO11s | 46.3 / 31.4 | 65.7 / 59.0 | 27.7 / 21.3 | 55.6 / 29.1 | 69.6 / 43.5 | 14.9 | 32% |
| Grounding DINO tiny | 49.9 / 36.5 | 70.0 / 66.8 | 33.5 / 29.8 | 57.1 / 34.0 | 69.3 / 46.6 | 13.5 | 27% |

COCO is re-evaluated with catIds = [1, 3, 8] so both columns cover the same 3 classes. Size buckets are COCO's (area < 32², < 96², else large). AP in %.
