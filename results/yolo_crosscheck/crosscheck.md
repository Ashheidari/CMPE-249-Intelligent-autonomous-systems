# YOLO11s COCO val2017 cross-check (M3)

Validator settings: {'imgsz': 640, 'conf': 0.001, 'iou': 0.7, 'max_det': 100, 'batch': 1, 'device': 'mps'}

| source | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| A_ultralytics_metric (multi_label NMS) | 46.2 | 62.9 | - | - | - | - |
| B_pycocotools_on_ultralytics_preds (multi_label NMS) | 46.8 | 63.5 | 50.5 | 28.8 | 51.5 | 64.4 |
| A_ultralytics_metric (single_label NMS) | 45.6 | 62.1 | - | - | - | - |
| B_pycocotools_on_ultralytics_preds (single_label NMS) | 46.2 | 62.7 | 49.8 | 28.2 | 50.9 | 63.6 |
| C_our_pipeline (model.predict, single_label NMS) | 45.5 | 62.0 | 49.2 | 27.6 | 50.1 | 62.5 |

- C - A_ultralytics_metric (multi_label NMS): -0.69 AP
- C - B_pycocotools_on_ultralytics_preds (multi_label NMS): -1.25 AP
- C - A_ultralytics_metric (single_label NMS): -0.11 AP
- C - B_pycocotools_on_ultralytics_preds (single_label NMS): -0.66 AP
