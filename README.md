# 2D Object Detection — Track A (Experimental)

Evaluation of three COCO-pretrained detectors (Faster R-CNN R50-FPN v2, YOLO11s, Grounding DINO tiny)
on COCO val2017 and a KITTI subset, without retraining. See `Implementation-plan.md` for the full plan.

## Reproduce from a clean clone, in order

Times are for an Apple M5 (16 GB) on the MPS backend. Each step's details are in the numbered
sections below. Large outputs (`predictions.json`, datasets, weights) are git-ignored; every
`metrics.json`, `config.json`, `benchmark.json`, table and figure is committed.

| # | Step | Command | Time | Main outputs |
|---|---|---|---|---|
| 0 | Environment, COCO download, dataset checks | `scripts/m0_setup.sh` | ~5 min + download | `.venv`, `env.txt`, `data/coco/` |
| 0b | **KITTI by hand** (registration), then check | `.venv/bin/python scripts/check_kitti.py` | manual | `data/kitti/` |
| 1 | Smoke test (optional) | `scripts/smoke_test.sh` | ~1 min | `results/smoke/` |
| 2 | KITTI subset → COCO-format GT | `scripts/m2_kitti.sh` | ~1 min | `data/kitti/annotations/`, `results/kitti_stats.md` |
| 3 | Pipeline check + YOLO cross-check | `scripts/m3_eval.sh` | ~30 min | `results/*_limit200/`, `results/yolo_crosscheck/` |
| 4 | Full runs (3 models × 2 datasets) + tables | `scripts/m4_full_runs.sh` | ~5.5 h | `results/<model>_<dataset>/`, `report/tables/` |
| 5 | Speed + memory benchmark | `scripts/m5_benchmark.sh` | ~40 min | `results/benchmark/`, `report/tables/speed.md` |
| 6 | Analysis 1: generalization | see section 6 | ~2 min | `report/analysis_generalization.md`, figures |
| 7 | Analysis 2: speed–accuracy | see section 7 | ~2 min | `report/analysis_speed.md`, figures |
| 8 | Analysis 3 + ablations | `scripts/m8_open_vocab.sh` | ~50 min | `report/analysis_open_vocab.md` |
| 9 | Failure analysis | `.venv/bin/python -m src.analysis.failures` | ~5 min | `report/failure_cases.md`, `report/figures/failures/` |
| 10 | Report draft | `.venv/bin/python -m src.analysis.build_report` | seconds | `report/report_draft.md` |

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

### 6. Analysis 1: cross-dataset generalization
```bash
.venv/bin/python -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_van_car.yaml \
    --out data/kitti/annotations/kitti_subset_coco_van_car.json --stats results/analysis/kitti_stats_van_car.md
.venv/bin/python -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_cyclist_person.yaml \
    --out data/kitti/annotations/kitti_subset_coco_cyclist_person.json --stats results/analysis/kitti_stats_cyclist_person.md
.venv/bin/python -m src.analysis.generalization   # re-evaluates saved predictions only
.venv/bin/python -m src.analysis.plots
```
This re-evaluates COCO on car/person/truck (`results/<model>_coco/metrics_shared3.json`).
It also computes KITTI recall by occlusion and truncation with a greedy IoU ≥ 0.5 matcher, and
re-scores KITTI under the Van: car and Cyclist: person label policies. Outputs:
`results/analysis/generalization.json`, `report/tables/generalization_*.md`, figures in `report/figures/`,
and the write-up `report/analysis_generalization.md`.

### 7. Analysis 2: speed–accuracy tradeoff
```bash
.venv/bin/python -m src.analysis.speed_accuracy   # joins M4 AP + M5 latency; measures input sizes, params
.venv/bin/python -m src.analysis.plots            # adds speed_accuracy.png and latency_breakdown.png
```
Outputs: `results/analysis/speed_accuracy.json`, `report/tables/speed_accuracy.md`, the figures, and the
write-up `report/analysis_speed.md`.

### 8. Analysis 3 (bonus): closed-set vs. open-vocabulary, plus follow-up experiments
```bash
scripts/m8_open_vocab.sh    # ~50 min on an M5
```
This builds the KITTI-native GT (`configs/kitti_mapping_native.yaml`: car, van, truck, pedestrian and
cyclist as real classes) and runs Grounding DINO with the prompt `car . van . truck . pedestrian . cyclist .`,
plus "person" and "walking person" wordings of the pedestrian class.
- `run_inference --rename CLASS=TEXT` changes the wording of one class in the prompt.
- `run_inference --set KEY=VALUE` and `benchmark --set KEY=VALUE` override a model setting. They are used
  for two follow-up experiments: YOLO11s at `imgsz=1280` on KITTI, and Grounding DINO with one 80-class
  prompt (`chunk_size=80`) on COCO.

The script ends by running `python -m src.analysis.open_vocab`, which writes `report/tables/open_vocab.md`
and `report/tables/followup_experiments.md`. Write-ups: `report/analysis_open_vocab.md`, and the follow-up
section of `report/analysis_speed.md`.

### 9. Failure analysis
```bash
.venv/bin/python -m src.analysis.failures   # ~5 min (re-runs the models on frames with cyclists)
```
This classifies every KITTI detection at score ≥ 0.3 as TP, duplicate, inside an ignore region, or FP
(wrong class / localization / background), and finds missed GT boxes. It also re-runs all 80 classes on
frames containing cyclists, to count person + bicycle boxes on KITTI Cyclist boxes. It then renders
13 gallery images, 3 of them side-by-side comparisons of all models. Outputs: `report/figures/failures/`,
`report/tables/failure_counts.md`, `report/failure_cases.md` and `results/analysis/failures.json`.

### 10. Report draft
```bash
.venv/bin/python -m src.analysis.build_report
```
This builds `report/report_draft.md` from `report/report_template.md`. Each `{{include: ...}}` line in
the template is replaced by a generated table (`report/tables/*.md`, `results/kitti_stats.md`,
`env.txt`), so every number in the draft traces back to a `metrics.json` or `benchmark.json`. Write in
the template, not the draft, and rebuild after re-running any analysis. Places that need the author's
interpretation are marked `TODO(user)`.

`logs/commands.log` lists every command run through `scripts/log_cmd.sh`. It ends with a clearly marked,
reconstructed list of the test and re-render commands that were run without it.
