# 2D Object Detection — Track A (Experimental)

Evaluation of three COCO-pretrained detectors (Faster R-CNN R50-FPN v2, YOLO11s, Grounding DINO tiny)
on COCO val2017 and a KITTI subset, without retraining. See `Implementation-plan.md` for the full plan.

## Reproduce

### 0. Environment and data
Requires Python 3.11 (`brew install python@3.11` on macOS).

```bash
scripts/m0_setup.sh
```
This creates `.venv`, installs the pinned `requirements.txt` (full lock in `requirements-lock.txt`),
writes `env.txt`, downloads COCO val2017 + annotations to `data/coco/`, and checks both datasets.

**KITTI must be downloaded by hand** (registration required). We use the labeled *training*
split (7,481 frames) with the **original 15-field KITTI labels** (needed for `DontCare`,
`truncated`, `occluded`). Two layouts are accepted under `data/kitti/`:

- Kaggle re-pack (used here): `data/kitti/{train,val}/{images,labels}/` — the train/val split is
  Kaggle's and is ignored; both halves together form the full training split. `test/` is unused.
- Official cvlibs zips (`data_object_image_2.zip` + `data_object_label_2.zip`):
  `data/kitti/training/{image_2,label_2}/`.

Do **not** use the Ultralytics `kitti.yaml` re-pack: its YOLO-format labels drop `DontCare` and the
occlusion/truncation fields, and its boxes are normalized by a fixed 1242×375 size, so frames of
other sizes get boxes shifted by up to ~18 px.

Verify with `.venv/bin/python scripts/check_kitti.py` (expects 7,481 paired frames, 15-field labels,
`DontCare` present).

Every command run through `scripts/log_cmd.sh` is appended to `logs/commands.log`.

### 1. Detector interface + smoke test
```bash
scripts/smoke_test.sh            # optional: --device cpu, --models yolo11s
```
Runs Faster R-CNN, YOLO11s and Grounding DINO on the first 5 COCO val2017 images (by id) and
writes visualizations to `results/smoke/<model>/` plus `results/smoke/summary.json` (device,
per-image time, detection counts, and the fraction of confident detections matching GT).
Weights download on first use: torchvision → `~/.cache/torch`, Ultralytics → `weights/yolo11s.pt`,
Hugging Face → `~/.cache/huggingface`. Per-model settings live in `configs/models.yaml`.
All models share the interface in `src/models/base.py` and return class *names*, boxes as
absolute-pixel `xyxy` in original image coordinates.

### 2. KITTI → COCO ground truth
```bash
scripts/m2_kitti.sh
```
- Uses the fixed 1,500-frame subset in `data/kitti/subset_1500.txt` (drawn once with seed 42 and
  committed; it is reused, never regenerated).
- Writes `data/kitti/annotations/kitti_subset_coco.json` with COCO's own category ids
  (person=1, car=3, truck=8), `area = w*h`, and KITTI `truncated` / `occluded` per box.
- Label policy: `configs/kitti_mapping.yaml`. `ignore` types become `iscrowd=1` ignore regions,
  written once per evaluated class because COCOeval matches per category. For another policy,
  run: `.venv/bin/python -m src.datasets.kitti_to_coco --mapping <yaml> --out <json> --stats <md>`.
- Outputs `results/kitti_stats.md` and 20 GT renders in `results/kitti_gt_check/`. It also checks
  that GT-as-predictions scores AP 1.0, with and without extra detections on ignore regions.

### 3. Shared COCO-style evaluation
```bash
scripts/m3_eval.sh            # or: scripts/m3_eval.sh limit | crosscheck
```
- `python -m src.eval.run_inference --model {faster_rcnn,yolo11s,gdino} --dataset {coco,kitti} [--limit N]`
  writes `results/<run_id>/predictions.json` (COCO results format, `bbox` = `[x, y, w, h]`) and
  `config.json`. `run_id` defaults to `<model>_<dataset>[_limit<N>]`. Every model gets the 80 COCO
  classes on both datasets. On KITTI, predictions outside car/person/truck are dropped and counted.
  An interrupted run resumes from `predictions.partial.jsonl`.
- `python -m src.eval.coco_eval --run results/<run_id> [--cat-ids ...]` runs pycocotools
  `COCOeval(bbox)` and writes `metrics.json`: the 12 standard stats plus per-class AP / AP50.
- `scripts/yolo_crosscheck.py` compares YOLO11s through our pipeline against Ultralytics' own
  validator on all 5,000 COCO images. It builds YOLO-format labels in `data/coco_yolo/` once.
  Output: `results/yolo_crosscheck/crosscheck.md`.

### 4. Full runs and results tables
```bash
scripts/m4_full_runs.sh                  # all 6 runs (~5.5 h on an M5, mostly Grounding DINO on COCO)
scripts/m4_full_runs.sh gdino kitti      # or one model x dataset at a time
.venv/bin/python -m src.analysis.tables  # regenerate tables from results/*/metrics.json
```
Writes `results/<model>_<dataset>/{predictions,config,metrics}.json` for the full sets, logs to
`logs/m4/`, and tables to `report/tables/`: `main_results.md/.csv`, `per_class_kitti.md`, and
`published_check.md` (our COCO AP vs. the published value for each checkpoint).

### 5. Speed and memory benchmark
```bash
scripts/m5_benchmark.sh                  # all 6 (~40 min on an M5), then report/tables/speed.md
scripts/m5_benchmark.sh yolo11s coco     # or one model x dataset (two separate arguments)
.venv/bin/python scripts/mps_shape_effect.py   # optional: first-seen input shape cost on MPS
```
`src/eval/benchmark.py` runs batch size 1 on a fixed image list: 20 warm-up and 200 timed images, repeated twice.
It times preprocess, forward and postprocess separately, synchronizing the device before every timestamp.
Each model wrapper has `preprocess` / `forward` / `postprocess` methods for this. A `--check` pass
confirms they give the same detections as `predict()`. Peak MPS memory is sampled after each image.
For Grounding DINO, the text encoder is also timed alone. Results are written to
`results/benchmark/<model>_<dataset>/benchmark.json` and `report/tables/speed.md`.
Stability is judged on the median: the mean of the first repeat includes one-off MPS kernel compilation
for COCO input shapes not seen during warm-up.
