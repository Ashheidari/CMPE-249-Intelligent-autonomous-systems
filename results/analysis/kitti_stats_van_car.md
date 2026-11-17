# KITTI subset statistics

- images: 1500
- evaluated boxes: 7386
- ignore regions (iscrowd=1, counted once each): 2799
- mapping: `{'Car': 'car', 'Pedestrian': 'person', 'Person_sitting': 'person', 'Truck': 'truck', 'Van': 'car', 'Cyclist': 'ignore', 'Tram': 'ignore', 'Misc': 'ignore', 'DontCare': 'ignore'}`

## Evaluated boxes by class and COCO size (area < 32², < 96², else large)

| class | boxes | small | medium | large | from KITTI types |
|---|---|---|---|---|---|
| car | 6251 | 1030 | 3358 | 1863 | Car 5687, Van 564 |
| person | 936 | 209 | 475 | 252 | Pedestrian 906, Person_sitting 30 |
| truck | 199 | 31 | 117 | 51 | Truck 199 |
| **all** | 7386 | 1270 | 3950 | 2166 | |

## Ignore regions by KITTI type

| KITTI type | regions |
|---|---|
| Cyclist | 306 |
| DontCare | 2198 |
| Misc | 196 |
| Tram | 99 |

## Evaluated boxes by occlusion level (0 visible, 1 partly, 2 largely, 3 unknown)

| class | occ 0 | occ 1 | occ 2 | occ 3 |
|---|---|---|---|---|
| car | 2828 | 1909 | 1246 | 268 |
| person | 561 | 212 | 144 | 19 |
| truck | 85 | 61 | 24 | 29 |
