"""Run each candidate detection model against the sampled Media URLs.

Reads data/media_urls.json (produced by fetch_media_urls.py), fetches
each image, runs it through every enabled model, and writes per-image
per-model metrics to data/eval_results.csv.

The models it can run (each is independently gate-able via CLI flag):

  --coco               YOLOv8x with COCO weights (80 everyday classes).
                       Baseline — same model the production yolo_microservice
                       runs today.

  --fashionpedia       YOLOv8 trained on Fashionpedia (46 fashion classes:
                       shoes, dresses, tops, bottoms, outerwear, bags,
                       accessories, jewelry). Community checkpoint; the
                       exact URL is picked via --fashionpedia-model. Any
                       ultralytics-loadable .pt file works — swap models
                       without editing this script.

  --grounding-dino     Open-vocabulary detection. Prompted with the Media's
                       CatalogProduct.category text (e.g. "shoes",
                       "skincare bottle", "activewear"). One model
                       covers every brand vertical.

  --all                Shorthand for --coco --fashionpedia --grounding-dino.

For each (media, model) pair the row includes:
  - detection_count       — how many bboxes returned
  - top_confidence        — highest-confidence detection's score
  - top_class             — class name of the top detection
  - top_bbox_area_frac    — top bbox area / image area
                            (proxy for "did we find the whole product or
                            just a spurious corner?")
  - top_class_matches_cat — fuzzy match between top class and
                            CatalogProduct.category (0/1, best-effort)
  - runtime_ms            — wall time for this model call
  - detections_json       — the full raw list for post-hoc analysis

Analyze with `python analyze.py` after the run.

Usage:
    python run_eval.py --all
    python run_eval.py --coco --fashionpedia
    python run_eval.py --coco --grounding-dino --limit 5
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).parent


# ── Model wrappers ─────────────────────────────────────────────────
#
# Each wrapper takes (image_pil, context_dict) and returns a list of
# dicts with a stable shape: {class_name, confidence, x1, y1, x2, y2}.
# Callers don't need to know which model produced the list.


class YoloWrapper:
    """Ultralytics YOLOv8 for both COCO and Fashionpedia. Pass any
    .pt path (local file or HF cache identifier)."""

    def __init__(self, weights: str, conf: float = 0.2, imgsz: int = 960):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def detect(self, image_pil, context):
        import numpy as np
        arr = np.array(image_pil.convert("RGB"))
        r = self.model.predict(arr, conf=self.conf, imgsz=self.imgsz,
                                iou=0.6, agnostic_nms=True,
                                verbose=False, augment=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        out = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append({
                "class_name": self.names.get(int(k), "object"),
                "confidence": float(c),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
            })
        return out


class GroundingDinoWrapper:
    """Open-vocabulary via HuggingFace transformers pipeline. Prompt
    text comes from the per-image context (CatalogProduct.category /
    title / brand fallback)."""

    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, model_id: str = None, conf: float = 0.25):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self.conf = conf
        self.torch = torch

    def _prompt_for(self, context: dict) -> str:
        """Build a Grounding DINO text prompt from CatalogProduct context.
        Multi-label format: 'shoe. sneaker. sandal.' (period-separated).
        Falls back through category → brand → generic when data thin."""
        cat = (context.get("product_category") or "").strip()
        title = (context.get("product_title") or "").strip()
        parts = []
        if cat:
            # Some products have compound categories like "Women > Shoes > Sneakers"
            parts.extend([p.strip().lower() for p in re.split(r"[>|/,;]", cat) if p.strip()])
        # Also add title-derived tokens (rough heuristic — take last noun-ish word)
        if title:
            # Take last 1–2 words as a noun candidate; strip color/size adjectives
            title_tokens = re.findall(r"[A-Za-z]+", title.lower())
            if title_tokens:
                parts.append(title_tokens[-1])
                if len(title_tokens) >= 2:
                    parts.append(" ".join(title_tokens[-2:]))
        # Always include broad fallbacks so the model has something even
        # when catalog data is sparse.
        parts.extend(["product", "object"])
        # Grounding DINO expects period-separated class strings.
        seen = set()
        deduped = []
        for p in parts:
            p = p.strip().lower()
            if p and p not in seen and len(p) <= 30:
                seen.add(p)
                deduped.append(p)
        return ". ".join(deduped[:8]) + "."

    def detect(self, image_pil, context):
        prompt = self._prompt_for(context)
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image_pil.size[::-1]])  # (H,W)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.conf, text_threshold=self.conf,
            target_sizes=target_sizes
        )[0]
        out = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            out.append({
                "class_name": label if isinstance(label, str) else str(label),
                "confidence": float(score),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        # Sort by confidence descending so top-1 metrics are stable.
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out


# ── Utility ────────────────────────────────────────────────────────


def fetch_image(url: str, timeout: int = 30) -> Image.Image:
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def fuzzy_class_matches_category(class_name: str, category: str) -> int:
    """Best-effort semantic match. Returns 1 if any token in class_name
    substring-matches any token in category (case-insensitive), else 0.
    Not perfect but good enough to flag totally-wrong classifications."""
    if not class_name or not category:
        return 0
    class_tokens = set(re.findall(r"[a-z]+", class_name.lower()))
    cat_tokens = set(re.findall(r"[a-z]+", category.lower()))
    for c in class_tokens:
        for k in cat_tokens:
            if c == k or (len(c) >= 4 and (c in k or k in c)):
                return 1
    return 0


def top_metrics(detections: list, img_w: int, img_h: int, category: str):
    if not detections:
        return {
            "detection_count": 0,
            "top_confidence": 0.0,
            "top_class": None,
            "top_bbox_area_frac": 0.0,
            "top_class_matches_cat": 0,
        }
    top = max(detections, key=lambda d: d["confidence"])
    area = max(0.0, top["x2"] - top["x1"]) * max(0.0, top["y2"] - top["y1"])
    frac = area / max(1.0, img_w * img_h)
    return {
        "detection_count": len(detections),
        "top_confidence": round(float(top["confidence"]), 3),
        "top_class": top["class_name"],
        "top_bbox_area_frac": round(float(frac), 4),
        "top_class_matches_cat": fuzzy_class_matches_category(top["class_name"], category or ""),
    }


# ── Main ───────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(HERE / "data" / "media_urls.json"),
                    help="Input JSON from fetch_media_urls.py.")
    ap.add_argument("--out",   type=str, default=str(HERE / "data" / "eval_results.csv"),
                    help="Output CSV path.")
    ap.add_argument("--coco",           action="store_true", help="Run YOLOv8x-COCO baseline.")
    ap.add_argument("--fashionpedia",   action="store_true", help="Run Fashionpedia YOLOv8.")
    ap.add_argument("--grounding-dino", action="store_true", help="Run Grounding DINO open-vocab.")
    ap.add_argument("--all",            action="store_true", help="Enable all three.")
    ap.add_argument("--fashionpedia-model", type=str, default=os.environ.get("FASHIONPEDIA_MODEL", "kesimeg/yolov8n-fashion"),
                    help="Fashionpedia model path or HuggingFace repo id.")
    ap.add_argument("--grounding-dino-model", type=str, default=os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"),
                    help="Grounding DINO HuggingFace model id.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit total images (for smoke testing).")
    args = ap.parse_args()

    if args.all:
        args.coco = args.fashionpedia = args.grounding_dino = True
    if not (args.coco or args.fashionpedia or args.grounding_dino):
        print("ERROR: enable at least one of --coco / --fashionpedia / --grounding-dino / --all", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        rows = json.load(f)
    if args.limit:
        rows = rows[:args.limit]
    print(f"📥 loaded {len(rows)} media rows from {args.input}")

    # Lazy-load each model so a run with just --coco doesn't pay Grounding
    # DINO's ~200MB download.
    models = {}
    if args.coco:
        print("🎯 loading YOLOv8x-COCO ...")
        models["coco"] = YoloWrapper("yolov8x.pt", conf=0.2)
    if args.fashionpedia:
        print(f"👗 loading Fashionpedia: {args.fashionpedia_model} ...")
        try:
            models["fashionpedia"] = YoloWrapper(args.fashionpedia_model, conf=0.25)
        except Exception as e:
            print(f"⚠️  Fashionpedia load failed ({args.fashionpedia_model}): {e}", file=sys.stderr)
    if args.grounding_dino:
        print(f"💬 loading Grounding DINO: {args.grounding_dino_model} ...")
        try:
            models["grounding-dino"] = GroundingDinoWrapper(args.grounding_dino_model, conf=0.25)
        except Exception as e:
            print(f"⚠️  Grounding DINO load failed ({args.grounding_dino_model}): {e}", file=sys.stderr)

    if not models:
        print("ERROR: no models loaded — nothing to run", file=sys.stderr)
        sys.exit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = [
        "media_id", "brand_name", "product_id", "product_category",
        "model", "detection_count", "top_confidence", "top_class",
        "top_bbox_area_frac", "top_class_matches_cat", "runtime_ms",
        "detections_json"
    ]
    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            url = row.get("file_url")
            if not url:
                continue
            try:
                img = fetch_image(url)
            except Exception as e:
                print(f"[{i}/{len(rows)}] ⚠️  fetch failed {row['media_id']}: {e}", file=sys.stderr)
                continue
            w, h = img.size
            cat = row.get("product_category")
            print(f"[{i}/{len(rows)}] {row['brand_name']}/{row['media_id']} ({w}x{h})")

            for model_name, model in models.items():
                t0 = time.time()
                try:
                    dets = model.detect(img, row)
                except Exception as e:
                    print(f"    ⚠️  {model_name} failed: {e}", file=sys.stderr)
                    dets = []
                elapsed_ms = int((time.time() - t0) * 1000)
                m = top_metrics(dets, w, h, cat or "")
                writer.writerow({
                    "media_id":              row["media_id"],
                    "brand_name":            row["brand_name"],
                    "product_id":            row.get("product_id"),
                    "product_category":      cat,
                    "model":                 model_name,
                    "detection_count":       m["detection_count"],
                    "top_confidence":        m["top_confidence"],
                    "top_class":             m["top_class"],
                    "top_bbox_area_frac":    m["top_bbox_area_frac"],
                    "top_class_matches_cat": m["top_class_matches_cat"],
                    "runtime_ms":            elapsed_ms,
                    "detections_json":       json.dumps(dets[:10]),  # cap to first 10
                })
                fp.flush()  # partial results survive if we ctrl-c
                print(f"    {model_name:16s} dets={m['detection_count']:3d}  top_conf={m['top_confidence']:.2f}  top={m['top_class']}  frac={m['top_bbox_area_frac']:.2f}  match={m['top_class_matches_cat']}  {elapsed_ms}ms")

    print(f"\n✓  wrote CSV → {args.out}")
    print("   analyze with: python analyze.py")


if __name__ == "__main__":
    main()
