# KITTI label-policy sensitivity (same predictions, different GT)

| model | policy | AP (3 cls) | car AP | person AP | truck AP | Δ car | Δ person | Δ truck |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | ignore (default) | 37.8 | 50.2 | 29.9 | 33.3 | +0.0 | +0.0 | +0.0 |
| Faster R-CNN R50-FPN v2 | Van: car | 33.4 | 48.4 | 29.9 | 22.1 | -1.8 | -0.0 | -11.3 |
| Faster R-CNN R50-FPN v2 | Cyclist: person | 35.5 | 50.2 | 23.1 | 33.3 | -0.0 | -6.8 | -0.0 |
| YOLO11s | ignore (default) | 31.4 | 47.3 | 27.0 | 20.0 | +0.0 | +0.0 | +0.0 |
| YOLO11s | Van: car | 27.7 | 45.6 | 26.9 | 10.6 | -1.7 | -0.0 | -9.3 |
| YOLO11s | Cyclist: person | 29.5 | 47.3 | 21.2 | 20.0 | -0.0 | -5.8 | +0.0 |
| Grounding DINO tiny | ignore (default) | 36.5 | 48.4 | 29.7 | 31.3 | +0.0 | +0.0 | +0.0 |
| Grounding DINO tiny | Van: car | 32.0 | 45.6 | 29.7 | 20.6 | -2.8 | -0.0 | -10.7 |
| Grounding DINO tiny | Cyclist: person | 33.8 | 48.3 | 21.9 | 31.2 | -0.0 | -7.8 | -0.0 |

GT boxes (car / person / truck): ignore (default): 5687 / 936 / 199; Van: car: 6251 / 936 / 199; Cyclist: person: 5687 / 1242 / 199. When a type stops being an ignore region, detections on it are no longer neutral for the other classes either (e.g. a 'truck' box on a van becomes a truck false positive under Van: car).
