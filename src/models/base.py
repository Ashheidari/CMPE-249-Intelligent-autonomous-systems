"""Common detector interface shared by all three models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image


class Detector(ABC):
    """A pretrained 2D detector.

    Every model returns class *names*, not IDs. The evaluator maps names to the dataset's
    category IDs, so the COCO label space and the KITTI mapping live in one place.
    """

    name: str

    def __init__(self, device: str, cfg: dict[str, Any]) -> None:
        self.device = device
        self.cfg = cfg
        # Run-level counters/notes, saved into config.json by the caller.
        self.stats: dict[str, Any] = {}

    @abstractmethod
    def predict(self, image: Image.Image, class_names: list[str]) -> dict:
        """Detect objects in one RGB image.

        Returns {'boxes':  np.ndarray[N, 4] float32, absolute-pixel xyxy in ORIGINAL image coords,
                 'scores': np.ndarray[N] float32, sorted descending,
                 'labels': list[str] of length N (class names)}.
        Closed-set models ignore `class_names` except for filtering their output to it;
        open-vocabulary models use it as the text prompt.
        """

    # Staged interface, used by the speed benchmark (src/eval/benchmark.py) to time each part.
    # postprocess(forward(preprocess(img, names)), ...) returns the same dict as predict().
    def preprocess(self, image: Image.Image, class_names: list[str]) -> dict:
        """PIL RGB image -> model inputs on self.device (resize, normalize, tokenize, ...)."""
        raise NotImplementedError

    def forward(self, inputs: dict) -> Any:
        """Run the network on the outputs of preprocess()."""
        raise NotImplementedError

    def postprocess(self, outputs: Any, inputs: dict) -> dict:
        """Raw network outputs -> the predict() result dict (decode, NMS, rescale, top-k)."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Settings to record in config.json."""
        return {"model": self.name, "device": self.device, **self.cfg, "stats": self.stats}


def make_result(boxes: np.ndarray, scores: np.ndarray, labels: list[str],
                class_names: list[str] | None = None, max_det: int | None = None) -> dict:
    """Package detections: optionally keep only `class_names`, sort by score, cap at `max_det`."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = np.arange(len(scores))
    if class_names is not None:
        allowed = set(class_names)
        keep = np.array([i for i in keep if labels[i] in allowed], dtype=int)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    if max_det is not None:
        keep = keep[:max_det]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": [labels[i] for i in keep]}
