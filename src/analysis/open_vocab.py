"""M8 analysis 3 (bonus): closed-set vs. open-vocabulary, plus two follow-up experiments from M7.

Usage: .venv/bin/python -m src.analysis.open_vocab   (after the runs in README step 9)

Grounding DINO was run on the KITTI subset with the KITTI-native prompt
"car . van . truck . pedestrian . cyclist ." (dataset kitti_native, category ids 101-105).
The same predictions are scored two ways:
  (a) mapped:  car + van -> car, pedestrian -> person, truck -> truck, cyclist dropped, against the
               default KITTI GT (car / person / truck, with Van and Cyclist as ignore regions).
               Compared with the COCO-80-prompt run from M4 (results/gdino_kitti). A "car only"
               variant drops van detections instead of merging them, to isolate the van merge.
  (b) native:  per-class AP on all 5 KITTI classes (Van and Cyclist are real classes here).
               COCO-trained closed-set models have no van or cyclist class, so they cannot do this.
Prompt sensitivity: the pedestrian class sent as "pedestrian", "person" or "walking person".
Follow-ups: YOLO11s on KITTI at imgsz 1280 vs. 640; Grounding DINO on COCO (first 200 images)
with one 80-class prompt vs. 4 prompts of 20.
Writes results/analysis/open_vocab.json, report/tables/open_vocab.md, report/tables/followup_experiments.md.
"""
from __future__ import annotations

import json

from ..datasets.coco import ROOT, load_gt, select_image_ids
from ..eval.coco_eval import evaluate
from .tables import md_table

RESULTS = ROOT / "results"
TABLES = ROOT / "report" / "tables"
NATIVE = ["car", "van", "truck", "pedestrian", "cyclist"]
SHARED = ["car", "person", "truck"]
# native category id -> COCO category id (None = drop)
MAPPED = {"car+van -> car": {101: 3, 102: 3, 103: 8, 104: 1, 105: None},
          "car only (van dropped)": {101: 3, 102: None, 103: 8, 104: 1, 105: None}}
WORDINGS = {"pedestrian": "gdino_kitti_native",
            "person": "gdino_kitti_native_pedestrian-as-person",
            "walking person": "gdino_kitti_native_pedestrian-as-walking_person"}


def load(path):
    return json.loads(path.read_text())


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def remap(preds: list[dict], table: dict[int, int | None]) -> list[dict]:
    return [{**p, "category_id": table[p["category_id"]]} for p in preds if table[p["category_id"]] is not None]


def main() -> None:
    kitti_gt = load_gt("kitti")
    ids = select_image_ids(kitti_gt)
    res: dict = {"wordings": {}}

    coco_prompt = load(RESULTS / "gdino_kitti" / "metrics.json")
    res["coco80_prompt"] = {"AP": coco_prompt["stats"]["AP"], **{c: coco_prompt["per_class"][c]["AP"] for c in SHARED}}
    res["closed_set_kitti"] = {m: {"AP": (x := load(RESULTS / f"{m}_kitti" / "metrics.json"))["stats"]["AP"],
                                   **{c: x["per_class"][c]["AP"] for c in SHARED}}
                               for m in ("faster_rcnn", "yolo11s")}
    for word, run in WORDINGS.items():
        d = RESULTS / run
        if not (d / "metrics.json").exists():
            print(f"skip {run}: not evaluated")
            continue
        native = load(d / "metrics.json")
        preds = load(d / "predictions.json")
        entry = {"run": run, "native": {"AP": native["stats"]["AP"], "AP50": native["stats"]["AP50"],
                                        **{c: native["per_class"][c]["AP"] for c in NATIVE},
                                        "AP50_per_class": {c: native["per_class"][c]["AP50"] for c in NATIVE},
                                        "n_gt": {c: native["per_class"][c]["n_gt"] for c in NATIVE}},
                 "mapped": {}}
        for name, table in MAPPED.items():
            m = evaluate(kitti_gt, remap(preds, table), ids)
            entry["mapped"][name] = {"AP": m["stats"]["AP"], **{c: m["per_class"][c]["AP"] for c in SHARED}}
        cfg = load(d / "config.json")
        entry["ms_per_image"] = cfg["timing"]["predict_ms_mean"]
        entry["prompt"] = cfg["prompt_classes"]
        res["wordings"][word] = entry
    res["followups"] = followups()
    out = RESULTS / "analysis" / "open_vocab.json"
    out.write_text(json.dumps(res, indent=2))
    write_tables(res)
    for f in ("open_vocab.md", "followup_experiments.md"):
        print((TABLES / f).read_text())


def followups() -> dict:
    """YOLO11s imgsz 1280 vs 640 on KITTI; Grounding DINO 1 vs 4 prompt chunks on COCO (first 200)."""
    out = {}
    pairs = {
        "yolo11s_kitti": [("imgsz 640 (default)", "yolo11s_kitti", "yolo11s_kitti"),
                          ("imgsz 1280", "yolo11s_kitti_imgsz1280", "yolo11s_kitti_imgsz1280")],
        "gdino_coco_200": [("4 prompts x 20 classes (default)", "gdino_coco_limit200", "gdino_coco"),
                           ("1 prompt x 80 classes", "gdino_coco_chunk_size80_limit200", "gdino_coco_chunk_size80")],
    }
    for exp, rows in pairs.items():
        out[exp] = {}
        for label, run, bench in rows:
            mp, bp = RESULTS / run / "metrics.json", RESULTS / "benchmark" / bench / "benchmark.json"
            if not mp.exists():
                continue
            m = load(mp)
            r = {"run": run, **{k: m["stats"][k] for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")}}
            if m["dataset"] == "kitti":
                r |= {c: m["per_class"][c]["AP"] for c in SHARED}
            if bp.exists():
                b = load(bp)
                r["latency_median_ms"] = b["runs"][-1]["total"]["median"]
                r["forward_ms"] = b["runs"][-1]["forward"]["mean"]
                r["peak_driver_mb"] = b["memory_mb"]["peak_during_timed"]["driver_allocated_mb"]
            cfg = load(RESULTS / run / "config.json")
            r["chunk_token_counts"] = cfg["detector"]["stats"].get("chunk_token_counts")
            out[exp][label] = r
    return out


def write_tables(res: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    w = res["wordings"]
    lines = ["# Closed-set vs. open-vocabulary on KITTI (Grounding DINO tiny)", "",
             "## (a) Mapped to car / person / truck (default KITTI GT, Van and Cyclist ignored)", ""]
    rows = [["Grounding DINO, COCO-80 prompt (M4)", pct(res["coco80_prompt"]["AP"]),
             *[pct(res["coco80_prompt"][c]) for c in SHARED]]]
    if "pedestrian" in w:
        for name, v in w["pedestrian"]["mapped"].items():
            rows.append([f"Grounding DINO, KITTI prompt, {name}", pct(v["AP"]), *[pct(v[c]) for c in SHARED]])
    names = {"faster_rcnn": "Faster R-CNN (closed-set, M4)", "yolo11s": "YOLO11s (closed-set, M4)"}
    rows += [[names[m], pct(v["AP"]), *[pct(v[c]) for c in SHARED]] for m, v in res["closed_set_kitti"].items()]
    lines += [md_table(["model / prompt", "AP (3 cls)", "car", "person", "truck"], rows), ""]
    lines += ["KITTI prompt: `car . van . truck . pedestrian . cyclist .`. Cyclist detections are dropped in (a).", "",
              "## (b) Native KITTI classes (Van and Cyclist are real classes)", ""]
    if "pedestrian" in w:
        n = w["pedestrian"]["native"]
        lines += [md_table(["", "AP (5 cls)", *NATIVE],
                           [["AP", pct(n["AP"]), *[pct(n[c]) for c in NATIVE]],
                            ["AP50", pct(n["AP50"]), *[pct(n["AP50_per_class"][c]) for c in NATIVE]],
                            ["GT boxes", "", *[str(n["n_gt"][c]) for c in NATIVE]]]), "",
                  "Faster R-CNN and YOLO11s cannot be scored here: COCO has no van or cyclist class.", ""]
    lines += ["## Prompt wording for the pedestrian class", ""]
    rows = []
    for word, e in w.items():
        rows.append([f'"{word}"', pct(e["native"]["pedestrian"]), pct(e["native"]["AP50_per_class"]["pedestrian"]),
                     pct(e["mapped"]["car+van -> car"]["person"]), pct(e["native"]["AP"]),
                     *[pct(e["native"][c]) for c in ("car", "van", "truck", "cyclist")], f"{e['ms_per_image']:.0f}"])
    lines += [md_table(["pedestrian sent as", "pedestrian AP (native)", "pedestrian AP50", "person AP (mapped)",
                        "AP (5 cls)", "car", "van", "truck", "cyclist", "ms/img"], rows), "",
              "Only the pedestrian wording changes; the other four names stay the same. All prompt words share one "
              "text encoding, so the other classes can move too.", ""]
    (TABLES / "open_vocab.md").write_text("\n".join(lines))

    f = res["followups"]
    lines = ["# Follow-up experiments (from the M7 hypotheses)", ""]
    if f.get("yolo11s_kitti"):
        rows = [[k, pct(v["AP"]), pct(v["APs"]), pct(v["APm"]), pct(v["APl"]), pct(v["car"]), pct(v["person"]),
                 pct(v["truck"]), f"{v.get('latency_median_ms', float('nan')):.1f}",
                 f"{v.get('peak_driver_mb', float('nan')):.0f}"] for k, v in f["yolo11s_kitti"].items()]
        lines += ["## YOLO11s on KITTI: input size 640 vs. 1280", "",
                  md_table(["setting", "AP", "APs", "APm", "APl", "car", "person", "truck", "latency median (ms)",
                            "peak driver MB"], rows), ""]
    if f.get("gdino_coco_200"):
        rows = [[k, pct(v["AP"]), pct(v["AP50"]), pct(v["APs"]), pct(v["APl"]), str(v["chunk_token_counts"]),
                 f"{v.get('latency_median_ms', float('nan')):.0f}", f"{v.get('peak_driver_mb', float('nan')):.0f}"]
                for k, v in f["gdino_coco_200"].items()]
        lines += ["## Grounding DINO on COCO (first 200 images): 4 prompts of 20 classes vs. 1 prompt of 80", "",
                  md_table(["setting", "AP", "AP50", "APs", "APl", "tokens per prompt", "latency median (ms)",
                            "peak driver MB"], rows), "",
                  "AP on the first 200 COCO images only (the full-set AP of the default setting is 46.7). Latency from "
                  "the M5 benchmark protocol on the same 200 timed images.", ""]
    (TABLES / "followup_experiments.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
