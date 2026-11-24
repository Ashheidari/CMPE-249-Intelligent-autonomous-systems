# Analysis 3 (bonus): Closed-set vs. open-vocabulary

Grounding DINO tiny was run again on the same 1,500 KITTI frames, this time with a **KITTI-native
prompt**, `car . van . truck . pedestrian . cyclist .` (one prompt, 12 tokens), instead of the 80 COCO
class names. Nothing was retrained. Numbers come from `results/gdino_kitti_native*/metrics.json` and
`results/analysis/open_vocab.json`. Regenerate them with `scripts/m8_open_vocab.sh`; the full tables are in
[tables/open_vocab.md](tables/open_vocab.md). AP is in %.

## (a) Mapped back to car / person / truck: the effect of the prompt

The same KITTI ground truth as in M4: Van and Cyclist are ignore regions. The native predictions are mapped
car + van → car, pedestrian → person, truck → truck, and cyclist detections are dropped.

| model / prompt | AP (3 cls) | car | person | truck |
|---|---|---|---|---|
| Grounding DINO, COCO-80 prompt (M4) | 36.5 | 48.4 | 29.7 | 31.3 |
| Grounding DINO, KITTI prompt | 39.0 | 50.8 | 22.1 | 44.3 |
| Grounding DINO, KITTI prompt with "person" instead of "pedestrian" | **40.7** | 49.3 | 28.0 | 44.8 |
| Faster R-CNN, closed-set (M4) | 37.8 | 50.2 | 29.9 | 33.3 |
| YOLO11s, closed-set (M4) | 31.4 | 47.3 | 27.0 | 20.0 |

- **A task-specific prompt raises Grounding DINO from 36.5 to 39.0 AP.** That moves it above Faster R-CNN
  (37.8). With "person" as the wording it reaches 40.7. The weights are the same; only the prompt changed.
- **The gain is mostly truck (+13.0) and car (+2.4).** Person drops 7.6 points when the class is called
  "pedestrian".
- **Merging van detections into car is worth +2.2 car AP** (50.8 vs. 48.6 when van detections are dropped).
  Under this GT, van boxes are ignore regions, so a "van" detection that is really a car would otherwise
  be lost.
- **The KITTI prompt is also about 4.5× faster:** 385 ms per image, against 1,730 ms with the four 20-class
  COCO prompts. It needs one forward pass instead of four.

## (b) Native KITTI classes: what the closed-set models cannot do

GT where Van (564 boxes) and Cyclist (306 boxes) are real classes. Person_sitting is counted as pedestrian.

|  | AP (5 cls) | car | van | truck | pedestrian | cyclist |
|---|---|---|---|---|---|---|
| AP | 27.6 | 48.4 | 24.7 | 36.2 | 18.1 | 10.9 |
| AP50 | 47.5 | 81.0 | 36.9 | 54.5 | 37.0 | 27.9 |

- **Grounding DINO detects vans and cyclists zero-shot**, at 24.7 and 10.9 AP (36.9 and 27.9 AP50).
  Faster R-CNN and YOLO11s have no van or cyclist class at all. The closest they get is "car"/"truck" for a
  van, or a separate "person" and "bicycle" for a cyclist, and neither matches KITTI's one cyclist box.
- **Those zero-shot scores are low next to car**, which COCO-style training data covers well.

## Prompt wording for one class

Only the pedestrian name changes; the other four stay the same.

| pedestrian sent as | pedestrian AP (native) | pedestrian AP50 | person AP (mapped) | AP (5 cls) |
|---|---|---|---|---|
| "pedestrian" | 18.1 | 37.0 | 22.1 | 27.6 |
| "person" | **24.0** | **56.3** | **28.0** | 27.6 |
| "walking person" | 12.0 | 25.4 | 14.4 | 26.2 |

The wording of one class moves its AP by 12 points: the best wording doubles the worst. It also shifts
other classes slightly; for example, car is 48.4 with "pedestrian" and 46.2 with "person", because all
prompt words are encoded together.

## Observations (hypotheses to verify, not conclusions)

- **H1, the vocabulary helps by narrowing the label space, not by adding knowledge.** With 5 classes instead
  of 80, a truck box no longer competes with "bus", "car" or "train" for its score. That could explain the
  +13 truck AP. *To verify:* check how many truck GT boxes are matched by a "bus"/"car" detection under the
  COCO-80 prompt. TODO(user)
- **H2, the model knows "person" better than "pedestrian".** Grounding DINO's grounding data (O365, GoldG,
  Cap4M) probably uses "person" far more often than "pedestrian". "Walking person" may hurt because KITTI's
  Pedestrian class also contains standing people. TODO(user)
- **H3, open vocabulary is a tradeoff, not a free win.** The fixed-class AP is now competitive (39–41 vs.
  37.8). But van and cyclist AP are low, the result depends on the exact words, and the model is about 47×
  slower than YOLO11s even with a short prompt (385 vs. 8.3 ms). TODO(user): your judgement on when this
  flexibility is worth it.
- TODO(user): whether the prompt choice ("person" vs. "pedestrian") counts as tuning on the test data, and
  how you will report it fairly. The prompt was not chosen on a held-out split.
