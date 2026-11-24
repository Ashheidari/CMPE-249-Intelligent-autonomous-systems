# From COCO to KITTI Without Fine-Tuning: Accuracy, Speed, and Vocabulary of Three Object Detector Families

This project evaluates three pretrained detectors from three design families, **without retraining**,
on COCO val2017 and a 1,500-frame KITTI subset. It then analyzes cross-dataset generalization, the
speed–accuracy tradeoff, and closed-set vs. open-vocabulary detection.

| Detector | Family | Checkpoint |
|---|---|---|
| Faster R-CNN R50-FPN v2 | two-stage, anchor-based | torchvision `fasterrcnn_resnet50_fpn_v2`, `COCO_V1` |
| YOLO11s | one-stage, anchor-free | Ultralytics `yolo11s.pt` |
| Grounding DINO tiny | open-vocabulary, query-based | Hugging Face `IDEA-Research/grounding-dino-tiny` |

**The report is [`report/submission/technical_report.pdf`](report/submission/technical_report.pdf).**
Every number in it comes from the files this README regenerates.

---

## 1. Requirements

- **Python 3.11** (on macOS: `brew install python@3.11`).
- **A GPU is recommended.** All results were produced on an Apple M5 (16 GB) with PyTorch's MPS backend.
  The code also runs on CUDA or CPU (`--device cuda|cpu`); see the notes at the end.
- **About 25 GB of free disk space:** COCO val2017 ≈ 1 GB, KITTI ≈ 12 GB, plus predictions.
- **Google Chrome**, only to render the report PDF in step 11.
- **Run every command from the repository root.**

## 2. Repository layout

```
configs/         models.yaml (per-model settings), kitti_mapping*.yaml (KITTI -> COCO label policies)
src/models/      common Detector interface + Faster R-CNN, YOLO11s, Grounding DINO wrappers
src/datasets/    dataset registry (coco.py), KITTI file index, KITTI -> COCO converter
src/eval/        run_inference.py (predictions), coco_eval.py (pycocotools metrics), benchmark.py (speed)
src/analysis/    tables, plots, generalization, speed_accuracy, open_vocab, failures, build_report
scripts/         Python helpers called below (environment record, dataset checks, cross-check, ...)
data/            datasets (git-ignored, except the fixed KITTI subset list data/kitti/subset_1500.txt)
results/         per-run config.json / metrics.json / benchmark.json (predictions.json are git-ignored)
report/          tables/, figures/, analysis write-ups, submission/technical_report.{html,pdf}
logs/            commands.log (every command run for the reported results) and run logs
```

## 3. Datasets

Both datasets go under `data/`, which is git-ignored except for the committed KITTI frame list
`data/kitti/subset_1500.txt`. Download them before running step 1 of section 4.

### 3.1 COCO val2017 (~1 GB, automatic)

Only the 5,000 validation images and `instances_val2017.json` are used. The annotation zip also contains
train-set files; they are harmless.

```bash
mkdir -p data/coco && cd data/coco
curl -L --fail -O http://images.cocodataset.org/zips/val2017.zip
curl -L --fail -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip
rm val2017.zip annotations_trainval2017.zip
cd ../..
```

### 3.2 KITTI 2D object detection (~12 GB, manual)

KITTI requires a free registration, so it cannot be downloaded by script. Only the **labeled training
split** (7,481 frames) is used, with its **original 15-field labels**. The code needs the `DontCare`,
`truncated` and `occluded` fields. Use either source; the code detects the layout automatically.

**Option A: official site.** Register at
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d> and download:

- "Download left color images of object data set" → `data_object_image_2.zip` (~12 GB)
- "Download training labels of object data set" → `data_object_label_2.zip` (~5 MB)

```bash
mkdir -p data/kitti
unzip -q data_object_image_2.zip -d data/kitti     # -> data/kitti/training/image_2/ (+ testing/, unused)
unzip -q data_object_label_2.zip -d data/kitti     # -> data/kitti/training/label_2/
```

**Option B: Kaggle re-pack (used for the reported results).** Download
<https://www.kaggle.com/datasets/hannanguyen24/kitti-object-detection-2d> (a Kaggle account is needed).
Inside the archive, the `Downloads/` folder contains `train/`, `val/` and `test/`; `train/` and `val/` each have
`images/` and `labels/`. Copy `train/` and `val/` into `data/kitti/`. The train/val split is Kaggle's own and is
ignored: together the two halves are the full 7,481-frame training split, and `test/` (no labels) is
not used. Check that a label file has 15 space-separated fields per line, e.g.
`Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62`.

Do **not** use the Ultralytics `kitti.yaml` re-pack: its YOLO-format labels drop `DontCare`, occlusion
and truncation, and its boxes are normalized by a fixed image size.

### 3.3 Expected layout

```
data/
├── coco/
│   ├── val2017/                       5,000 .jpg images
│   └── annotations/instances_val2017.json
└── kitti/
    ├── subset_1500.txt                committed: the fixed 1,500-frame subset (seed 42)
    ├── training/image_2/*.png         Option A: 7,481 images
    ├── training/label_2/*.txt         Option A: 7,481 labels
    ├── train/{images,labels}/         Option B: together with val/, 7,481 frames
    └── val/{images,labels}/           Option B
```

Folders created later by the pipeline: `data/kitti/annotations/` (COCO-format KITTI GT, step 3 and the
analyses) and `data/coco_yolo/` (YOLO-format COCO labels for the Ultralytics cross-check, step 5).

### 3.4 Check the data

```bash
python scripts/check_coco.py             # expects 5,000 images and a loadable instances_val2017.json
python scripts/check_kitti.py            # expects 7,481 paired frames with 15-field labels and DontCare
```

## 4. Reproduce step by step

Times are for the Apple M5 (MPS). The steps must run in this order, because later steps read
earlier outputs. Set up the shell once, then run each block:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pinned versions; full lock in requirements-lock.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1     # set it again in every new terminal
python scripts/record_env.py             # -> env.txt (versions, chip, RAM, MPS availability)
```

On macOS, prefix the long runs with `caffeinate -i` so the Mac does not sleep. Inference runs that
are interrupted **resume** when you rerun the same command. Finished runs are skipped; use `--overwrite`
to redo one.

### Step 1: Data (~5 min for COCO, plus the manual KITTI download)

Download and place both datasets as described in section 3, then check them:

```bash
python scripts/check_coco.py
python scripts/check_kitti.py
```

### Step 2: Smoke test (optional, ~1 min)

```bash
python scripts/smoke_test.py             # 3 models x 5 COCO images -> results/smoke/<model>/*.png
```

### Step 3: KITTI → COCO-format ground truth (~1 min)

Uses the fixed, committed frame list `data/kitti/subset_1500.txt` (1,500 frames drawn once with seed 42)
and the label policy in `configs/kitti_mapping.yaml`. Van, Cyclist, Tram, Misc and DontCare become COCO
ignore regions (`iscrowd=1`).

```bash
python -m src.datasets.kitti_to_coco     # -> data/kitti/annotations/kitti_subset_coco.json, results/kitti_stats.md
python scripts/kitti_gt_check.py         # 20 GT renders + checks that GT-as-predictions scores AP 1.0
```

### Step 4: Pipeline check on 200 images (~20 min)

```bash
for m in faster_rcnn yolo11s gdino; do
  for d in coco kitti; do
    python -m src.eval.run_inference --model $m --dataset $d --limit 200
    python -m src.eval.coco_eval --run results/${m}_${d}_limit200
  done
done
```

### Step 5: Full evaluation, 3 models × 2 datasets (~5.5 h; Grounding DINO on COCO alone is ~3 h 50 min)

```bash
for m in faster_rcnn yolo11s gdino; do
  for d in coco kitti; do
    python -m src.eval.run_inference --model $m --dataset $d      # -> results/<m>_<d>/predictions.json, config.json
    python -m src.eval.coco_eval --run results/${m}_${d}          # -> results/<m>_<d>/metrics.json
  done
done
python -m src.analysis.tables            # -> report/tables/main_results.{md,csv}, per_class_kitti.md, published_check.md
```

YOLO11s cross-check against Ultralytics' own validator. It needs the full `yolo11s_coco` run above and
builds YOLO-format COCO labels in `data/coco_yolo/` the first time it runs (~10 min):

```bash
python scripts/yolo_crosscheck.py        # -> results/yolo_crosscheck/crosscheck.md
```

### Step 6: Speed and memory benchmark (~40 min)

Batch 1; 20 warm-up and 200 timed images (the same list for every model), repeated twice.
Preprocessing, forward pass and postprocessing are timed separately. Run it with nothing else using
the GPU.

```bash
for m in faster_rcnn yolo11s gdino; do
  for d in coco kitti; do
    python -m src.eval.benchmark --model $m --dataset $d          # -> results/benchmark/<m>_<d>/benchmark.json
  done
done
python scripts/mps_shape_effect.py       # cost of first-seen input shapes on MPS -> results/benchmark/shape_effect.json
python -m src.analysis.tables            # adds report/tables/speed.md
```

### Step 7: Analysis 1, cross-dataset generalization (~2 min)

Two alternative KITTI label policies (Van → car, Cyclist → person) for the sensitivity test. These only
re-score the saved predictions; nothing is re-run.

```bash
python -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_van_car.yaml \
    --out data/kitti/annotations/kitti_subset_coco_van_car.json --stats results/analysis/kitti_stats_van_car.md
python -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_cyclist_person.yaml \
    --out data/kitti/annotations/kitti_subset_coco_cyclist_person.json --stats results/analysis/kitti_stats_cyclist_person.md
python -m src.analysis.generalization    # -> results/analysis/generalization.json, report/tables/generalization_*.md
```

### Step 8: Analysis 2, speed–accuracy (~2 min)

```bash
python -m src.analysis.speed_accuracy    # -> results/analysis/speed_accuracy.json, report/tables/speed_accuracy.md
python -m src.analysis.plots             # -> all figures in report/figures/
```

### Step 9: Analysis 3 (open vocabulary) and two ablations (~50 min)

Grounding DINO on KITTI with the KITTI-native prompt `car . van . truck . pedestrian . cyclist .`,
including two other wordings of the pedestrian class (`--rename`):

```bash
python -m src.datasets.kitti_to_coco --mapping configs/kitti_mapping_native.yaml \
    --out data/kitti/annotations/kitti_subset_native.json --stats results/analysis/kitti_stats_native.md
python -m src.eval.run_inference --model gdino --dataset kitti_native
python -m src.eval.coco_eval --run results/gdino_kitti_native
python -m src.eval.run_inference --model gdino --dataset kitti_native --rename pedestrian=person
python -m src.eval.coco_eval --run results/gdino_kitti_native_pedestrian-as-person
python -m src.eval.run_inference --model gdino --dataset kitti_native --rename "pedestrian=walking person"
python -m src.eval.coco_eval --run results/gdino_kitti_native_pedestrian-as-walking_person
```

Ablations (`--set` overrides one setting from `configs/models.yaml`): YOLO11s at 1280 px on KITTI, and
Grounding DINO with one 80-class prompt instead of 4 prompts of 20 (COCO, first 200 images):

```bash
python -m src.eval.run_inference --model yolo11s --dataset kitti --set imgsz=1280
python -m src.eval.coco_eval --run results/yolo11s_kitti_imgsz1280
python -m src.eval.benchmark --model yolo11s --dataset kitti --set imgsz=1280
python -m src.eval.run_inference --model gdino --dataset coco --limit 200 --set chunk_size=80
python -m src.eval.coco_eval --run results/gdino_coco_chunk_size80_limit200
python -m src.eval.benchmark --model gdino --dataset coco --set chunk_size=80
python -m src.analysis.open_vocab        # -> report/tables/open_vocab.md, followup_experiments.md
```

### Step 10: Failure analysis (~5 min)

```bash
python -m src.analysis.failures          # -> report/failure_cases.md, report/tables/failure_counts.md, report/figures/failures/
```

### Step 11: Report

Markdown draft with all generated tables included (optional):

```bash
python -m src.analysis.build_report      # report/report_template.md -> report/report_draft.md
```


## 5. Where the results are

| What | File |
|---|---|
| Main results (all metrics, model × dataset) | `report/tables/main_results.md` / `.csv` |
| COCO AP vs. published values | `report/tables/published_check.md` |
| Per-run settings, versions, timing | `results/<model>_<dataset>/config.json` |
| Per-run metrics (12 COCO stats, per-class AP / AP50 / AP75 / AP per IoU) | `results/<model>_<dataset>/metrics.json` |
| Speed and memory | `results/benchmark/<model>_<dataset>/benchmark.json`, `report/tables/speed.md` |
| Analyses (numbers + hypotheses) | `report/analysis_generalization.md`, `analysis_speed.md`, `analysis_open_vocab.md` |
| Failure cases | `report/failure_cases.md`, `report/figures/failures/` |
| Environment and commands actually run | `env.txt`, `logs/commands.log` |

## 6. Notes

- **Device:** `run_inference`, `benchmark`, `smoke_test.py` and `yolo_crosscheck.py` take
  `--device mps|cuda|cpu`. Without it, and in the analysis modules that re-run models, the default is
  MPS, then CUDA, then CPU. If a model fails on the GPU, it falls back to CPU for that model only, and `config.json` records
  the fallback.
- **One evaluator:** all mAP numbers come from pycocotools `COCOeval` (`src/eval/coco_eval.py`), with
  score threshold 0.001 and at most 100 detections per image. The only exception is the Ultralytics
  cross-check, which is reported for comparison.
- **Label space:** every model is asked for the 80 COCO classes on both datasets. On KITTI, predictions
  outside car/person/truck are dropped afterwards. Detectors return class *names*, and boxes as pixel
  `xyxy` in original-image coordinates (`src/models/base.py`).
- **Grounding DINO:** the 80 COCO names are split into 4 prompts of 20 and merged with class-wise NMS.
  Each box gets its class from its mean probability over that class name's tokens (`label_assignment:
  tokens`). The plan's phrase-matching rule is available as `label_assignment: phrase`.
- **Determinism:** seed 42. The KITTI subset list and the benchmark image lists are saved and reused,
  never regenerated.
- **Command log:** `logs/commands.log` records every command run for the reported results, with a
  timestamp. Those runs were launched through small shell wrappers that are no longer part of the
  repository; they ran exactly the commands listed in this README.
