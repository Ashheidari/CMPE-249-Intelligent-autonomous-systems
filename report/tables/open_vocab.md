# Closed-set vs. open-vocabulary on KITTI (Grounding DINO tiny)

## (a) Mapped to car / person / truck (default KITTI GT, Van and Cyclist ignored)

| model / prompt | AP (3 cls) | car | person | truck |
|---|---|---|---|---|
| Grounding DINO, COCO-80 prompt (M4) | 36.5 | 48.4 | 29.7 | 31.3 |
| Grounding DINO, KITTI prompt, car+van -> car | 39.0 | 50.8 | 22.1 | 44.3 |
| Grounding DINO, KITTI prompt, car only (van dropped) | 38.3 | 48.6 | 22.1 | 44.3 |
| Faster R-CNN (closed-set, M4) | 37.8 | 50.2 | 29.9 | 33.3 |
| YOLO11s (closed-set, M4) | 31.4 | 47.3 | 27.0 | 20.0 |

KITTI prompt: `car . van . truck . pedestrian . cyclist .`. Cyclist detections are dropped in (a).

## (b) Native KITTI classes (Van and Cyclist are real classes)

|  | AP (5 cls) | car | van | truck | pedestrian | cyclist |
|---|---|---|---|---|---|---|
| AP | 27.6 | 48.4 | 24.7 | 36.2 | 18.1 | 10.9 |
| AP50 | 47.5 | 81.0 | 36.9 | 54.5 | 37.0 | 27.9 |
| GT boxes |  | 5687 | 564 | 199 | 936 | 306 |

Faster R-CNN and YOLO11s cannot be scored here: COCO has no van or cyclist class.

## Prompt wording for the pedestrian class

| pedestrian sent as | pedestrian AP (native) | pedestrian AP50 | person AP (mapped) | AP (5 cls) | car | van | truck | cyclist | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| "pedestrian" | 18.1 | 37.0 | 22.1 | 27.6 | 48.4 | 24.7 | 36.2 | 10.9 | 387 |
| "person" | 24.0 | 56.3 | 28.0 | 27.6 | 46.2 | 22.7 | 36.9 | 8.0 | 385 |
| "walking person" | 12.0 | 25.4 | 14.4 | 26.2 | 48.4 | 23.3 | 37.2 | 9.8 | 384 |

Only the pedestrian wording changes; the other four names stay the same. All prompt words share one text encoding, so the other classes can move too.
