# Speed and memory (batch 1, device mps)

Latency in ms per image (stage means; total = sum of stages per image).

| model | dataset | preprocess | forward | postprocess | total mean | total median | total p90 | FPS | run diff (mean) | run diff (median) | peak MPS alloc (MB) | peak driver (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN v2 | COCO val2017 | 2.7 | 250.9 | 3.9 | 257.5 | 255.9 | 271.7 | 3.9 | 0.2% | 0.6% | 167 | 2250 |
| Faster R-CNN R50-FPN v2 | KITTI subset | 2.5 | 194.9 | 3.7 | 201.0 | 201.0 | 201.7 | 5.0 | 0.1% | 0.4% | 167 | 1210 |
| YOLO11s | COCO val2017 | 1.3 | 10.3 | 1.6 | 13.2 | 13.2 | 14.6 | 75.9 | 32.2% | 0.6% | 36 | 183 |
| YOLO11s | KITTI subset | 1.5 | 5.2 | 1.5 | 8.3 | 8.3 | 8.3 | 120.8 | 1.1% | 1.1% | 36 | 118 |
| Grounding DINO tiny | COCO val2017 | 15.7 | 2789.6 | 23.2 | 2828.5 | 2786.6 | 3084.2 | 0.4 | 2.4% | 0.2% | 659 | 5409 |
| Grounding DINO tiny | KITTI subset | 7.3 | 1761.4 | 21.5 | 1790.3 | 1781.7 | 1815.1 | 0.6 | 0.2% | 0.3% | 659 | 3336 |

Protocol: 20 warm-up + 200 timed images (same list for every model), 2 repeats; table shows the last repeat. run diff = (max-min)/min across repeats of the mean or median total; the median is the stability criterion (<10%), because the mean of the first repeat includes one-off MPS kernel compilation for input shapes not seen during warm-up (see below). Image decoding is not timed. Device synchronized before every timestamp. Evaluation thresholds (score 0.001, 100 dets). NMS: Faster R-CNN final NMS in postprocess (proposal NMS inside forward); YOLO11s NMS in postprocess; Grounding DINO forward = 4 prompt-chunk passes, postprocess includes the cross-chunk NMS.

Memory: peak of torch.mps.current_allocated_memory() (tensors) and driver_allocated_memory() (all Metal allocations) after each image. Apple Silicon unified memory is shared with the CPU, so this is not equivalent to CUDA VRAM.

- Grounding DINO on COCO val2017: text encoder (BERT, 4 prompt chunks) alone takes 33.2 ms/img, about 1.2% of the forward time.
- Grounding DINO on KITTI subset: text encoder (BERT, 4 prompt chunks) alone takes 33.1 ms/img, about 1.9% of the forward time.

## First-seen input shapes on MPS

COCO images vary in size, so the network input shape varies; MPS compiles kernels the first time it sees a shape. KITTI frames all have one size. Measured after the same 20 warm-up images:

| model | dataset | distinct shapes | imgs with new shape | ms (new shape) | ms (seen shape) |
|---|---|---|---|---|---|
| YOLO11s | COCO val2017 | 18 | 10 | 140.9 | 13.5 |
| Faster R-CNN R50-FPN v2 | COCO val2017 | 26 | 17 | 281.9 | 243.1 |
| YOLO11s | KITTI subset | 1 | 0 | - | 8.5 |
| Faster R-CNN R50-FPN v2 | KITTI subset | 1 | 0 | - | 201.0 |
