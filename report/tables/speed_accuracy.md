# Speed-accuracy summary

| model | dataset | AP | latency median (ms) | FPS | peak driver mem (MB) | params (M) | typical input W×H | input Mpx (mean) | original Mpx (mean) | forward ms / input Mpx |
|---|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | COCO val2017 | 46.9 | 255.9 | 3.9 | 2250 | 43.7 | 1216×800 | 0.91 | 0.28 | 276 |
| Faster R-CNN R50-FPN v2 | KITTI subset | 37.8 | 201.0 | 5.0 | 1210 | 43.7 | 1344×416 | 0.56 | 0.46 | 349 |
| YOLO11s | COCO val2017 | 45.5 | 13.2 | 75.9 | 183 | 9.5 | 640×448 | 0.30 | 0.28 | 34 |
| YOLO11s | KITTI subset | 31.4 | 8.3 | 120.8 | 118 | 9.5 | 640×224 | 0.14 | 0.46 | 36 |
| Grounding DINO tiny | COCO val2017 | 46.7 | 2786.6 | 0.4 | 5409 | 172.3 | 1066×800 | 0.89 | 0.28 | 3123 |
| Grounding DINO tiny | KITTI subset | 36.5 | 1781.7 | 0.6 | 3336 | 172.3 | 1333×402 | 0.54 | 0.46 | 3285 |

AP: COCO = 80 classes, KITTI = car/person/truck (M4). Latency, FPS and memory: M5 benchmark (batch 1, MPS, last repeat). Input size is measured after each model's own resize/letterbox (padding included) on the 200 timed benchmark images. Grounding DINO's forward time covers 4 passes (one per prompt chunk), so its ms per input Mpx counts each pixel 4 times.
