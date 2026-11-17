"""All report figures. Reads results/*/metrics*.json and results/analysis/*.json only.

Usage: .venv/bin/python -m src.analysis.plots

M6 figures (report/figures/):
  per_class_ap.png         per-class AP, COCO vs. KITTI, one panel per model
  ap_by_size.png           APs / APm / APl on the shared classes, COCO vs. KITTI, per model
  recall_vs_occlusion.png  KITTI recall at IoU 0.5 by occlusion level, car and person
  ap_vs_iou.png            per-class AP at each IoU threshold 0.50..0.95, COCO vs. KITTI

Color always encodes the model (validated categorical slots 1-3); COCO is drawn in neutral gray
or dashed so color never means "dataset". Exact numbers are in the Markdown tables in report/tables/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG = ROOT / "report" / "figures"
MODELS = {"faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s", "gdino": "Grounding DINO"}
CLASSES = ["car", "person", "truck"]

# Reference palette, light mode (validated: scripts/validate_palette.js, all-pairs, 3 slots)
COLOR = {"faster_rcnn": "#2a78d6", "yolo11s": "#eb6834", "gdino": "#1baf7a"}
COCO_GRAY = "#aeaca4"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "axes.axisbelow": True,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(model: str, dataset: str, shared: bool = False) -> dict:
    name = "metrics_shared3.json" if shared else "metrics.json"
    return load(RESULTS / f"{model}_{dataset}" / name)


def paired_bars(ax, labels: list[str], coco: list[float], kitti: list[float], color: str) -> None:
    """Two thin bars per group (COCO gray, KITTI model color) with a 2 px surface gap between them."""
    x = np.arange(len(labels))
    w = 0.34
    kw = {"edgecolor": SURFACE, "linewidth": 2}
    ax.bar(x - w / 2, coco, w, color=COCO_GRAY, label="COCO", **kw)
    ax.bar(x + w / 2, kitti, w, color=color, label="KITTI", **kw)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 85)  # headroom so the legend never sits on a bar
    ax.legend(loc="upper right", handlelength=1, handleheight=1, ncols=2)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


def per_class_ap() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        c, k = metrics(m, "coco")["per_class"], metrics(m, "kitti")["per_class"]
        paired_bars(ax, CLASSES, [100 * c[x]["AP"] for x in CLASSES], [100 * k[x]["AP"] for x in CLASSES], COLOR[m])
        ax.set_title(MODELS[m])
    axes[0].set_ylabel("AP (%)")
    fig.suptitle("Per-class AP: COCO val2017 vs. KITTI subset", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "per_class_ap.png")


def ap_by_size() -> None:
    sizes = [("APs", "small"), ("APm", "medium"), ("APl", "large")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        c, k = metrics(m, "coco", shared=True)["stats"], metrics(m, "kitti")["stats"]
        paired_bars(ax, [s[1] for s in sizes], [100 * c[s[0]] for s in sizes], [100 * k[s[0]] for s in sizes], COLOR[m])
        ax.set_title(MODELS[m])
    axes[0].set_ylabel("AP (%)")
    fig.suptitle("AP by object size (COCO areas: <32², <96², larger), car + person + truck only",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "ap_by_size.png")


def recall_vs_occlusion() -> None:
    res = load(RESULTS / "analysis" / "generalization.json")
    levels = ["occ 0 fully visible", "occ 1 partly occluded", "occ 2 largely occluded"]
    xl = ["fully visible", "partly", "largely"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, cls in zip(axes, ["car", "person"]):
        for m in MODELS:
            for thr, style in (("0.3", "-"), ("0.001", "--")):
                r = res[m]["recall"][thr][cls]
                ax.plot(xl, [100 * r[l]["recall"] for l in levels], style, color=COLOR[m], marker="o",
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        alpha=1 if thr == "0.3" else 0.55,
                        label=f"{MODELS[m]}, score ≥ {thr}")
        n = res[next(iter(MODELS))]["recall"]["0.3"][cls]
        ax.set_xticks(range(3), [f"{x}\n(n={n[l]['n']})" for x, l in zip(xl, levels)])
        ax.set_title(cls)
        ax.set_ylim(0, 102)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("recall at IoU ≥ 0.5 (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, handlelength=3.5)
    fig.suptitle("KITTI recall by occlusion level", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "recall_vs_occlusion.png")


def ap_vs_iou() -> None:
    ious = np.linspace(0.5, 0.95, 10)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        for m in MODELS:
            for d, style, alpha in (("coco", "--", 0.55), ("kitti", "-", 1.0)):
                y = [100 * v for v in metrics(m, d)["per_class"][cls]["AP_by_iou"]]
                ax.plot(ious, y, style, color=COLOR[m], alpha=alpha, label=f"{MODELS[m]}, {d.upper()}")
        ax.set_title(cls)
        ax.set_xlabel("IoU threshold")
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    axes[0].set_ylabel("AP (%)")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, handlelength=3.5)
    fig.suptitle("AP vs. IoU threshold: dashed = COCO, solid = KITTI", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "ap_vs_iou.png")


def main() -> None:
    per_class_ap()
    ap_by_size()
    recall_vs_occlusion()
    ap_vs_iou()


if __name__ == "__main__":
    main()
