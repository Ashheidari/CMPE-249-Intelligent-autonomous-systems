# KITTI recall at IoU >= 0.5 by occlusion and truncation

Greedy score-sorted matching per image and class; a GT box counts as found if a same-class detection with score >= threshold overlaps it at IoU >= 0.5 and was not already used. Recall in %.

| model | score >= | class | occ 0 fully visible | occ 1 partly occluded | occ 2 largely occluded | occ 3 unknown | trunc 0-0.15 | trunc 0.15-0.3 | trunc 0.3-0.5 | trunc >0.5 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | 0.001 | car | 98.9 | 96.6 | 80.5 | 86.6 | 94.3 | 96.7 | 92.5 | 93.3 | 94.3 |
| Faster R-CNN R50-FPN v2 | 0.001 | person | 93.6 | 77.8 | 31.9 | 63.2 | 81.2 | 72.7 | 66.7 | 36.8 | 79.9 |
| Faster R-CNN R50-FPN v2 | 0.3 | car | 98.1 | 93.4 | 61.3 | 73.8 | 89.0 | 92.4 | 87.7 | 85.4 | 88.7 |
| Faster R-CNN R50-FPN v2 | 0.3 | person | 91.4 | 65.1 | 15.3 | 52.6 | 74.3 | 63.6 | 50.0 | 36.8 | 73.0 |
| YOLO11s | 0.001 | car | 98.0 | 93.5 | 73.2 | 84.8 | 91.5 | 98.9 | 93.4 | 88.3 | 91.4 |
| YOLO11s | 0.001 | person | 88.2 | 70.3 | 27.8 | 68.4 | 75.4 | 77.3 | 58.3 | 36.8 | 74.5 |
| YOLO11s | 0.3 | car | 93.7 | 79.7 | 45.2 | 65.9 | 78.6 | 91.3 | 86.8 | 81.2 | 79.2 |
| YOLO11s | 0.3 | person | 76.3 | 42.5 | 9.7 | 42.1 | 58.2 | 68.2 | 33.3 | 36.8 | 57.7 |
| Grounding DINO tiny | 0.001 | car | 98.0 | 93.8 | 66.5 | 77.4 | 90.4 | 95.7 | 90.6 | 84.4 | 89.9 |
| Grounding DINO tiny | 0.001 | person | 93.0 | 79.2 | 24.3 | 63.2 | 79.8 | 68.2 | 66.7 | 47.4 | 78.7 |
| Grounding DINO tiny | 0.3 | car | 96.1 | 89.2 | 56.7 | 68.9 | 85.9 | 90.2 | 85.8 | 80.2 | 85.5 |
| Grounding DINO tiny | 0.3 | person | 90.9 | 73.6 | 16.7 | 57.9 | 75.9 | 68.2 | 66.7 | 42.1 | 74.9 |

## GT counts and box size per group

| class |  | occ 0 fully visible | occ 1 partly occluded | occ 2 largely occluded | occ 3 unknown | trunc 0-0.15 | trunc 0.15-0.3 | trunc 0.3-0.5 | trunc >0.5 | all |
|---|---|---|---|---|---|---|---|---|---|---|
| car | GT boxes | 2685 | 1725 | 1113 | 164 | 5009 | 92 | 106 | 480 | 5687 |
| person | GT boxes | 561 | 212 | 144 | 19 | 883 | 22 | 12 | 19 | 936 |
| car | median height (px) | 49 | 43 | 45 | 119 | 43 | 136 | 161 | 168 | 47 |
| person | median height (px) | 107 | 92 | 78 | 156 | 93 | 215 | 182 | 195 | 97 |
