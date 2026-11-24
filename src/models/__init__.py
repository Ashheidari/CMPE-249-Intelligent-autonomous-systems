"""Model registry. `build_detector` handles device choice and per-model CPU fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from ..devices import pick_device
from .base import Detector

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ["faster_rcnn", "yolo11s", "gdino"]


def load_model_cfg(name: str, path: Path = ROOT / "configs" / "models.yaml") -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())[name]
    if name == "yolo11s":  # resolve checkpoint path relative to the repo root
        cfg["checkpoint"] = str(ROOT / cfg["checkpoint"])
    return cfg


def _construct(name: str, device: str, cfg: dict[str, Any]) -> Detector:
    if name == "faster_rcnn":
        from .faster_rcnn import FasterRCNN
        return FasterRCNN(device, cfg)
    if name == "yolo11s":
        from .yolo import YOLO11
        return YOLO11(device, cfg)
    if name == "gdino":
        from .grounding_dino import GroundingDINO
        return GroundingDINO(device, cfg)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """['imgsz=1280', 'chunk_size=80'] -> {'imgsz': 1280, 'chunk_size': 80} (values parsed as YAML)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def build_detector(name: str, device: str | None = None, overrides: dict[str, Any] | None = None) -> Detector:
    """Build a detector on the preferred device and run one dummy prediction.

    If the dummy run fails on a GPU backend, rebuild on CPU and record why in
    `detector.stats['device_fallback']` (this ends up in config.json).
    """
    cfg = load_model_cfg(name)
    unknown = set(overrides or {}) - set(cfg)
    if unknown:
        raise ValueError(f"unknown {name} config keys: {unknown}")
    cfg.update(overrides or {})
    dev = pick_device(device)
    det = _construct(name, dev, cfg)
    try:
        det.predict(Image.new("RGB", (640, 480)), ["person"])
    except (RuntimeError, NotImplementedError) as e:
        if dev == "cpu":
            raise
        print(f"[{name}] failed on {dev} ({type(e).__name__}: {e}); falling back to cpu")
        det = _construct(name, "cpu", cfg)
        det.stats["device_fallback"] = f"{dev} -> cpu: {type(e).__name__}: {e}"
    if hasattr(det, "reset_stats"):
        det.reset_stats()  # don't count the dummy run
    return det
