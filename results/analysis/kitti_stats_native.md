# KITTI subset statistics

- images: 1500
- evaluated boxes: 7692
- ignore regions (iscrowd=1, counted once each): 2493
- mapping: `{'Car': 'car', 'Van': 'van', 'Truck': 'truck', 'Pedestrian': 'pedestrian', 'Person_sitting': 'pedestrian', 'Cyclist': 'cyclist', 'Tram': 'ignore', 'Misc': 'ignore', 'DontCare': 'ignore'}`

## Evaluated boxes by class and COCO size (area < 32², < 96², else large)

| class | boxes | small | medium | large | from KITTI types |
|---|---|---|---|---|---|
| car | 5687 | 948 | 3046 | 1693 | Car 5687 |
| van | 564 | 82 | 312 | 170 | Van 564 |
| truck | 199 | 31 | 117 | 51 | Truck 199 |
| pedestrian | 936 | 209 | 475 | 252 | Pedestrian 906, Person_sitting 30 |
| cyclist | 306 | 74 | 168 | 64 | Cyclist 306 |
| **all** | 7692 | 1344 | 4118 | 2230 | |

## Ignore regions by KITTI type

| KITTI type | regions |
|---|---|
| DontCare | 2198 |
| Misc | 196 |
| Tram | 99 |

## Evaluated boxes by occlusion level (0 visible, 1 partly, 2 largely, 3 unknown)

| class | occ 0 | occ 1 | occ 2 | occ 3 |
|---|---|---|---|---|
| car | 2685 | 1725 | 1113 | 164 |
| van | 143 | 184 | 133 | 104 |
| truck | 85 | 61 | 24 | 29 |
| pedestrian | 561 | 212 | 144 | 19 |
| cyclist | 195 | 47 | 11 | 53 |
