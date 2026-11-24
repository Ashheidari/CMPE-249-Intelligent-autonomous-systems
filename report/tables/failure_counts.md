# Failure counts on the KITTI subset (detections with score >= 0.3, IoU 0.5)

| count | Faster R-CNN R50-FPN v2 | YOLO11s | Grounding DINO tiny |
|---|---|---|---|
| GT total | 6822 | 6822 | 6822 |
| TP | 5861 | 5107 | 5695 |
| missed GT | 961 | 1715 | 1127 |
| missed: small (<32² px) | 135 | 484 | 162 |
| missed: occluded (level 1-2) | 768 | 1266 | 872 |
| FP | 5538 | 1008 | 3264 |
| FP score > 0.5 | 3487 | 366 | 564 |
| FP: wrong class | 140 | 43 | 108 |
| FP: localization | 1897 | 630 | 1140 |
| FP: background | 3501 | 335 | 2016 |
| duplicate | 95 | 149 | 18 |
| ignored | 6189 | 1544 | 4026 |
| on Van region | 960 | 462 | 570 |
| on Van region as truck | 363 | 158 | 360 |
| on Van region as car | 537 | 304 | 172 |
| on Cyclist region | 432 | 172 | 320 |
| on DontCare region | 4598 | 844 | 2989 |
| cyclist boxes (80-class re-run) | 306 | 306 | 306 |
| cyclists with a person box (80-class re-run) | 265 | 163 | 248 |
| cyclists with a bicycle box (80-class re-run) | 159 | 82 | 231 |
| cyclists with person + bicycle (80-class re-run) | 146 | 67 | 208 |

See src/analysis/failures.py for the definitions. 'ignored' = inside an ignore region of its class (intersection / detection area >= 0.5); such detections do not count as FP in COCOeval either. Cyclist rows come from a re-run with all 80 COCO classes kept (the saved KITTI predictions keep only car/person/truck): person = IoU >= 0.3 with the cyclist box, bicycle = >= 70% of the bicycle box inside it.
