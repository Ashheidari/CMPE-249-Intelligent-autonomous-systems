# KITTI subset statistics

- images: 1500
- evaluated boxes: 6822
- ignore regions (iscrowd=1, counted once each): 3363
- mapping: `{'Car': 'car', 'Pedestrian': 'person', 'Person_sitting': 'person', 'Truck': 'truck', 'Van': 'ignore', 'Cyclist': 'ignore', 'Tram': 'ignore', 'Misc': 'ignore', 'DontCare': 'ignore'}`

## Evaluated boxes by class and COCO size (area < 32², < 96², else large)

| class | boxes | small | medium | large | from KITTI types |
|---|---|---|---|---|---|
| car | 5687 | 948 | 3046 | 1693 | Car 5687 |
| person | 936 | 209 | 475 | 252 | Pedestrian 906, Person_sitting 30 |
| truck | 199 | 31 | 117 | 51 | Truck 199 |
| **all** | 6822 | 1188 | 3638 | 1996 | |

## Ignore regions by KITTI type

| KITTI type | regions |
|---|---|
| Cyclist | 306 |
| DontCare | 2198 |
| Misc | 196 |
| Tram | 99 |
| Van | 564 |

## Evaluated boxes by occlusion level (0 visible, 1 partly, 2 largely, 3 unknown)

| class | occ 0 | occ 1 | occ 2 | occ 3 |
|---|---|---|---|---|
| car | 2685 | 1725 | 1113 | 164 |
| person | 561 | 212 | 144 | 19 |
| truck | 85 | 61 | 24 | 29 |
