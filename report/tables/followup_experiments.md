# Follow-up experiments (from the M7 hypotheses)

## YOLO11s on KITTI: input size 640 vs. 1280

| setting | AP | APs | APm | APl | car | person | truck | latency median (ms) | peak driver MB |
|---|---|---|---|---|---|---|---|---|---|
| imgsz 640 (default) | 31.4 | 21.3 | 29.1 | 43.5 | 47.3 | 27.0 | 20.0 | 8.3 | 118 |
| imgsz 1280 | 35.9 | 23.9 | 35.4 | 44.3 | 51.4 | 28.3 | 28.1 | 16.5 | 1102 |

## Grounding DINO on COCO (first 200 images): 4 prompts of 20 classes vs. 1 prompt of 80

| setting | AP | AP50 | APs | APl | tokens per prompt | latency median (ms) | peak driver MB |
|---|---|---|---|---|---|---|---|
| 4 prompts x 20 classes (default) | 51.8 | 67.3 | 35.4 | 68.5 | [47, 56, 49, 49] | 2787 | 5409 |
| 1 prompt x 80 classes | 53.1 | 67.6 | 36.9 | 70.0 | [195] | 740 | 5384 |

AP on the first 200 COCO images only (the full-set AP of the default setting is 46.7). Latency from the M5 benchmark protocol on the same 200 timed images.
