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

**KITTI must be downloaded by hand** (registration required). Run `.venv/bin/python scripts/check_kitti.py`
for instructions; it passes once `data/kitti/training/{image_2,label_2}` hold 7,481 files each.

Every command run through `scripts/log_cmd.sh` is appended to `logs/commands.log`.
