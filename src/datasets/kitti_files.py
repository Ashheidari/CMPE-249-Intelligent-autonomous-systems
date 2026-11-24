"""Locate KITTI training images and labels (Kaggle train/val re-pack or official training/ layout)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KITTI_ROOT = ROOT / "data" / "kitti"
EXPECTED_FRAMES = 7481

# (image_dir, label_dir) pairs relative to the root, per supported layout.
_LAYOUTS = [
    [("train/images", "train/labels"), ("val/images", "val/labels")],  # Kaggle
    [("training/image_2", "training/label_2")],                         # official
]


def kitti_index(root: Path = DEFAULT_KITTI_ROOT) -> dict[str, tuple[Path | None, Path | None]]:
    """Map frame id -> (image_path, label_path) using the first layout found under root."""
    root = Path(root)
    for layout in _LAYOUTS:
        if not any((root / img).is_dir() for img, _ in layout):
            continue
        index: dict[str, list[Path | None]] = {}
        for img_dir, lbl_dir in layout:
            for p in (root / img_dir).glob("*.png"):
                index.setdefault(p.stem, [None, None])[0] = p
            for p in (root / lbl_dir).glob("*.txt"):
                index.setdefault(p.stem, [None, None])[1] = p
        return {k: (v[0], v[1]) for k, v in sorted(index.items())}
    raise FileNotFoundError(f"no KITTI layout found under {root}")
