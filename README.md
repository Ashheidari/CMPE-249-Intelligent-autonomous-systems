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
