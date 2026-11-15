"""Record software versions and hardware info to env.txt (reproducibility record).

Usage: .venv/bin/python scripts/record_env.py [--out env.txt]
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["torch", "torchvision", "ultralytics", "transformers", "pycocotools",
            "numpy", "pillow", "pandas", "matplotlib", "pyyaml", "tqdm"]


def sh(cmd: list[str]) -> str:
    """Run a command and return stripped stdout, or 'unknown' on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "env.txt"))
    args = ap.parse_args()

    import torch  # imported here so a missing torch still gives a useful error above

    mem_bytes = sh(["sysctl", "-n", "hw.memsize"])
    ram_gb = f"{int(mem_bytes) / 2**30:.1f} GB" if mem_bytes.isdigit() else "unknown"

    lines = [
        f"python: {sys.version.split()[0]}",
        *[f"{p}: {version(p)}" for p in PACKAGES],
        f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}",
        f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
        f"os: macOS {sh(['sw_vers', '-productVersion'])} (build {sh(['sw_vers', '-buildVersion'])})",
        f"machine: {platform.machine()}",
        f"chip: {sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}",
        f"cpu_cores: {sh(['sysctl', '-n', 'hw.ncpu'])}",
        f"total_ram: {ram_gb}",
    ]
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
