"""All report figures (report/figures/), from results/*/metrics*.json and results/analysis/*.json.

Color always encodes the model; COCO is drawn gray or dashed.
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

# Categorical palette, light mode: 3 hues checked for color-blind separation and contrast on the surface
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
                ax.plot(xl, [100 * r[lev]["recall"] for lev in levels], style, color=COLOR[m], marker="o",
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        alpha=1 if thr == "0.3" else 0.55,
                        label=f"{MODELS[m]}, score ≥ {thr}")
        n = res[next(iter(MODELS))]["recall"]["0.3"][cls]
        ax.set_xticks(range(3), [f"{x}\n(n={n[lev]['n']})" for x, lev in zip(xl, levels)])
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


def mix(hex_color: str, other: str, t: float) -> str:
    """Blend hex_color toward other by fraction t (0 = unchanged)."""
    a = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(other[i:i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(v)):02x}" for v in (1 - t) * a + t * b)


def speed_accuracy() -> None:
    res = load(RESULTS / "analysis" / "speed_accuracy.json")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.grid(True, axis="both", which="major")
    for m in MODELS:
        pts = {d: (res[m][d]["latency_total_ms_median"], 100 * res[m][d]["AP"], res[m][d]["peak_driver_mb"])
               for d in ("coco", "kitti")}
        ax.plot([pts["coco"][0], pts["kitti"][0]], [pts["coco"][1], pts["kitti"][1]], color=GRID, lw=1.5, zorder=1)
        for d, marker in (("coco", "o"), ("kitti", "s")):
            x, y, mem = pts[d]
            ax.scatter(x, y, s=90, marker=marker, color=COLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{mem:,.0f} MB", (x, y), xytext=(9, -4), textcoords="offset points",
                        fontsize=8, color=INK2)
        x, y, _ = pts["coco"]
        ax.annotate(MODELS[m], (x, y), xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(5, 6000)
    ax.set_ylim(25, 52)
    ax.set_xlabel("median latency per image, batch 1, Apple M5 MPS (ms, log scale)")
    ax.set_ylabel("AP (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=INK2, markersize=8, label="COCO val2017 (80 classes)"),
               Line2D([], [], marker="s", ls="", color=INK2, markersize=8, label="KITTI subset (car/person/truck)")]
    handles += [Line2D([], [], marker="o", ls="", color=COLOR[m], markersize=8, label=MODELS[m]) for m in MODELS]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Speed vs. accuracy (labels next to points: peak MPS driver memory)", loc="left")
    fig.tight_layout()
    save(fig, "speed_accuracy.png")


def latency_breakdown() -> None:
    res = load(RESULTS / "analysis" / "speed_accuracy.json")
    stages = ["preprocess", "forward", "postprocess"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, m in zip(axes, MODELS):
        shades = [mix(COLOR[m], "#ffffff", 0.6), COLOR[m], mix(COLOR[m], "#000000", 0.4)]
        top = 0
        for i, d in enumerate(("coco", "kitti")):
            bottom = 0.0
            lat = res[m][d]["latency_ms"]
            for st, c in zip(stages, shades):
                ax.bar(i, lat[st], 0.5, bottom=bottom, color=c, edgecolor=SURFACE, linewidth=2,
                       label=st if i == 0 else None)
                bottom += lat[st]
            top = max(top, bottom)
            ax.annotate(f"{bottom:.1f} ms\npre {lat['preprocess']:.1f} · post {lat['postprocess']:.1f}",
                        (i, bottom), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_ylim(0, top * 1.25)
        ax.set_xticks([0, 1], ["COCO", "KITTI"])
        ax.set_xlim(-0.6, 1.6)
        ax.tick_params(axis="x", length=0)
        ax.set_title(MODELS[m])
        ax.legend(loc="upper right", fontsize=8, handlelength=1, handleheight=1)
    axes[0].set_ylabel("latency per image (ms)")
    fig.suptitle("Latency breakdown (mean, batch 1, MPS) — note each panel has its own scale",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "latency_breakdown.png")


def main() -> None:
    per_class_ap()
    ap_by_size()
    recall_vs_occlusion()
    ap_vs_iou()
    speed_accuracy()
    latency_breakdown()


if __name__ == "__main__":
    main()
