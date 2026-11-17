"""Grounding DINO tiny (open-vocabulary, query-based) from Hugging Face transformers.

Prompt: lowercase class names joined by " . " and ending in " ." (e.g. "car . person . truck .").
COCO's 80 names are split into chunks (cfg.chunk_size) with one forward pass per chunk; detections
are concatenated and, if there was more than one chunk, class-wise NMS (cfg.chunk_nms_iou) is
applied before keeping the top cfg.max_det. All 80 names are ~195 BERT tokens, under the
256-token limit, so chunking is a design choice rather than a hard requirement here.

The text encoder runs inside every forward pass, so text encoding is part of the timed path.

Label assignment (cfg.label_assignment):
  "tokens" (default): the model outputs, per query q, a probability p[q, t] = sigmoid(logit)
      for every prompt token t. The score of class c for query q is the mean of p[q, t] over the
      tokens of c's name (this is the positive-map scoring used in the official Grounding DINO
      COCO evaluation). Every (query, class) pair with score > box_threshold is a candidate
      detection, so one query can yield several class hypotheses, as in DETR-style COCO eval.
      text_threshold is not used in this mode.
  "phrase": the plan's original rule, using HF post_process_grounded_object_detection, which
      keeps queries whose max token prob > box_threshold and decodes all tokens with
      prob > text_threshold into a phrase. The phrase is mapped to a class by
        1. exact match after lowercase/strip,
        2. else the longest class name that is a substring of the phrase or vice versa,
        3. else dropped (counted in stats['dropped_unmatched_labels']).
      With low text thresholds, phrases often span several classes ("wine glass cup potted
      plant") or subword fragments ("gife"), so rule 2 can pick the wrong class; phrases matching
      more than one class are counted in stats['ambiguous_phrases'].
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .base import Detector, make_result


def build_prompt(names: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Return the prompt and the (start, end) character span of each name in it."""
    parts, spans, pos = [], [], 0
    for n in names:
        n = n.lower().strip()
        spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n) + len(" . ")
    return " . ".join(parts) + " .", spans


def match_phrase(phrase: str, names: list[str]) -> tuple[str | None, int]:
    """Map a decoded phrase to a class name (phrase mode). Returns (name or None, #candidates)."""
    p = phrase.lower().strip()
    lowered = {n.lower().strip(): n for n in names}
    if p in lowered:
        return lowered[p], 1
    cands = [n for n in lowered if p and (n in p or p in n)]
    if not cands:
        return None, 0
    return lowered[max(cands, key=len)], len(cands)


class GroundingDINO(Detector):
    name = "gdino"

    def __init__(self, device: str, cfg: dict[str, Any]) -> None:
        super().__init__(device, cfg)
        self.processor = AutoProcessor.from_pretrained(cfg["checkpoint"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg["checkpoint"]).eval().to(device)
        self._chunks: dict[tuple[str, ...], list[dict]] = {}  # tokenized prompts per class list
        self.reset_stats()

    def reset_stats(self) -> None:
        """Clear cached prompts and counters (stats describe one run)."""
        self._chunks.clear()
        self.stats.update(chunk_token_counts=[], dropped_unmatched_labels=0, ambiguous_phrases=0)

    def _prepare_chunks(self, class_names: list[str]) -> list[dict]:
        """Tokenize each chunk's prompt once; record token spans of every class name."""
        key = tuple(class_names)
        if key in self._chunks:
            return self._chunks[key]
        size = self.cfg["chunk_size"]
        chunks = []
        for i in range(0, len(class_names), size):
            names = class_names[i:i + size]
            prompt, spans = build_prompt(names)
            enc = self.processor.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].tolist()  # [(char_start, char_end)] per token
            n_tok = enc["input_ids"].shape[1]
            if n_tok > self.cfg["max_text_tokens"]:
                raise ValueError(f"prompt chunk has {n_tok} tokens > {self.cfg['max_text_tokens']}")
            # token indices of each class name (special tokens have offset (0, 0) and are skipped)
            tok_ids = [[t for t, (s, e) in enumerate(offsets) if e > s and s >= a and e <= b]
                       for a, b in spans]
            chunks.append({"names": names, "prompt": prompt, "enc": enc, "class_tokens": tok_ids})
            self.stats["chunk_token_counts"].append(n_tok)
            print(f"[gdino] chunk {len(chunks)}: {len(names)} classes, {n_tok} tokens")
        self._chunks[key] = chunks
        return chunks

    def _assign_tokens(self, out, chunk: dict, W: int, H: int):
        """Token-span scoring. Returns boxes [M,4] xyxy px, scores [M], labels (list[str])."""
        probs = out.logits[0].sigmoid()                      # [Q=900, 256]
        cls_scores = torch.stack([probs[:, t].mean(-1) for t in chunk["class_tokens"]], dim=1)  # [Q, C]
        q, c = torch.nonzero(cls_scores > self.cfg["box_threshold"], as_tuple=True)
        cx, cy, w, h = out.pred_boxes[0][q].unbind(-1)       # normalized cxcywh
        boxes = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)
        return boxes, cls_scores[q, c], [chunk["names"][i] for i in c.tolist()]

    def _assign_phrase(self, out, chunk: dict, W: int, H: int):
        """HF phrase decoding + string matching (plan's original rule)."""
        r = self.processor.post_process_grounded_object_detection(
            out, chunk["enc"]["input_ids"].to(self.device), threshold=self.cfg["box_threshold"],
            text_threshold=self.cfg["text_threshold"], target_sizes=[(H, W)])[0]
        keep, labels = [], []
        for i, phrase in enumerate(r["text_labels"]):
            name, n_cands = match_phrase(phrase, chunk["names"])
            if name is None:
                self.stats["dropped_unmatched_labels"] += 1
                continue
            self.stats["ambiguous_phrases"] += n_cands > 1
            keep.append(i)
            labels.append(name)
        keep_t = torch.tensor(keep, dtype=torch.long, device=r["boxes"].device)
        return r["boxes"][keep_t], r["scores"][keep_t], labels

    @torch.no_grad()
    def predict(self, image: Image.Image, class_names: list[str]) -> dict:
        inputs = self.preprocess(image, class_names)
        return self.postprocess(self.forward(inputs), inputs)

    @torch.no_grad()
    def preprocess(self, image: Image.Image, class_names: list[str]) -> dict:
        """Resize + normalize the image; tokenized prompt chunks (cached after the first call) to device."""
        image = image.convert("RGB")
        chunks = self._prepare_chunks(class_names)
        pix = self.processor.image_processor(image, return_tensors="pt").to(self.device)  # resized + normalized
        encs = [c["enc"].to(self.device) for c in chunks]
        return {"pix": pix, "encs": encs, "chunks": chunks, "size": image.size, "class_names": class_names}

    @torch.no_grad()
    def forward(self, inputs: dict) -> list:
        """One forward pass per prompt chunk; the text encoder (BERT) runs inside each pass."""
        return [self.model(**inputs["pix"], **enc) for enc in inputs["encs"]]

    @torch.no_grad()
    def postprocess(self, outputs: list, inputs: dict) -> dict:
        """Assign classes per chunk, merge chunks with class-wise NMS, clamp, keep top max_det."""
        W, H = inputs["size"]
        class_names = inputs["class_names"]
        assign = self._assign_tokens if self.cfg["label_assignment"] == "tokens" else self._assign_phrase
        all_boxes, all_scores, all_labels = [], [], []
        for out, chunk in zip(outputs, inputs["chunks"]):
            b, s, l = assign(out, chunk, W, H)
            all_boxes.append(b.float())
            all_scores.append(s.float())
            all_labels += l
        boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
        if len(inputs["chunks"]) > 1 and self.cfg.get("chunk_nms_iou") and len(scores):
            idx = torch.tensor([class_names.index(n) for n in all_labels], device=boxes.device)
            keep = batched_nms(boxes, scores, idx, self.cfg["chunk_nms_iou"]).tolist()
            boxes, scores, all_labels = boxes[keep], scores[keep], [all_labels[i] for i in keep]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        return make_result(boxes.cpu().numpy(), scores.cpu().numpy(), all_labels,
                           None, self.cfg["max_det"])
