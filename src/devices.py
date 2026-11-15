"""Device selection and synchronization helpers."""
from __future__ import annotations

import os

# Let unsupported MPS ops fall back to CPU instead of raising. Must be set before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def pick_device(requested: str | None = None) -> str:
    """Return `requested` if given, else the first available of mps, cuda, cpu."""
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until queued GPU work on `device` finishes (call before taking a timestamp)."""
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
