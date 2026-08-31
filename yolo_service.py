# Multi-stage product detector. Restored from the pre-rewrite version that gave
# the best recall on real inventory photos, extended with EXIF handling, unified
# response shape, and the /detect-video endpoint we added later.
#
# Detection stages (image):
#   1. YOLOv8x with tiled inference (big model + sliding window catches small items)
#   2. OpenCV rectangle proposals (boxes/pouches/packs via contour shape)
#   3. OpenAI gpt-4o-mini box fallback (only when YOLO+rects look sparse)
#   4. NMS merge + confidence filter

import os, io, json, base64, cv2, numpy as np, torch, tempfile, traceback
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, request, jsonify
from ultralytics import YOLO
from torchvision.ops import nms

# ── YOLO knobs ────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("YOLO_MODEL", "yolov8x.pt")
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU      = float(os.getenv("YOLO_IOU", "0.60"))
YOLO_IMGSZ    = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_MAXDET   = int(os.getenv("YOLO_MAX_DET", "300"))

# ── Open-vocabulary detection (Grounding DINO) ────────────────────
# YOLOv8x-COCO has 80 everyday-object classes — no shoes, no most apparel,
# no cosmetics-specific labels. Measured 86% empty refinedProducts on
# Soludos and mislabelled ("vase", "skateboard") on the ones it did
# detect. Fashionpedia was considered as a second YOLO head; a local eval
# (yolo_microservice/eval/) showed Grounding DINO with the product's own
# CatalogProduct.category as the text prompt hits 100% detection at 100%
# correct labels across Soludos, Pelagic Gear, and Gymshark — strictly
# dominates Fashionpedia. So we shipped Grounding DINO instead.
#
# HOW IT'S GATED:
#   POST /detect with 'prompt' field  →  Grounding DINO ONLY (skip COCO)
#   POST /detect without 'prompt'     →  existing pipeline (COCO + OpenCV
#                                        + OAI fallback), unchanged
#
# This keeps UGC (no prompt) behaviour byte-identical and gives catalog
# (has prompt from CatalogProduct.category/title) the specialized path.
# Not merged with COCO because Grounding DINO alone hits 100% — running
# both would ~3.5× the latency (measured on CPU) for zero recall gain.
#
# MEMORY: Grounding DINO tiny is ~200MB, on top of YOLOv8x's ~500MB.
# Both load at boot; with GUNICORN_PRELOAD_APP=true they COW-share across
# workers. Standard Plus (4GB) fits 5-6 workers.
#
# LATENCY: ~15-20s per image on CPU (measured, ~3.5× COCO's 5s). Async
# ingest job so this doesn't block anything user-facing.
OPEN_VOCAB_ENABLED = os.getenv("YOLO_OPEN_VOCAB_ENABLED", "true").lower() == "true"
OPEN_VOCAB_MODEL   = os.getenv("YOLO_OPEN_VOCAB_MODEL", "IDEA-Research/grounding-dino-tiny")
OPEN_VOCAB_CONF    = float(os.getenv("YOLO_OPEN_VOCAB_CONF", "0.25"))
OPEN_VOCAB_TEXT_CONF = float(os.getenv("YOLO_OPEN_VOCAB_TEXT_CONF", "0.25"))

# ── Tiled inference ───────────────────────────────────────────────
USE_TILING    = os.getenv("YOLO_TILING", "1") == "1"
TILE          = int(os.getenv("YOLO_TILE", "1024"))
OVERLAP       = float(os.getenv("YOLO_TILE_OVERLAP", "0.35"))

# ── OpenCV rectangle proposals ────────────────────────────────────
FALLBACK_RECT      = os.getenv("FALLBACK_RECT", "1") == "1"
RECT_MIN_AREA_FRAC = float(os.getenv("RECT_MIN_AREA_FRAC", "0.01"))
RECT_MAX           = int(os.getenv("RECT_MAX", "12"))

# ── OpenAI gpt-4o-mini box fallback (triggers only when recall is weak) ──
OAI_BOX_FALLBACK      = os.getenv("OAI_BOX_FALLBACK", "1") == "1"
OAI_MODEL             = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TRIGGER_MIN_DETS  = int(os.getenv("OAI_TRIGGER_MIN_DETS", "6"))
OAI_TRIGGER_MIN_COVER = float(os.getenv("OAI_TRIGGER_MIN_COVER", "0.22"))
OAI_TIMEOUT           = int(os.getenv("OAI_TIMEOUT", "30"))

# ── Output / video ────────────────────────────────────────────────
CONF_THRESHOLD   = float(os.getenv("CONF_THRESHOLD", "0.25"))  # final floor on output confidence
IOU_DEDUP        = float(os.getenv("IOU_DEDUP", "0.3"))
VIDEO_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "2"))
VERBOSE          = os.getenv("YOLO_VERBOSE", "true").lower() == "true"

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

app = Flask(__name__)
model = YOLO(MODEL_PATH)
print(f"🎯 primary model loaded: {MODEL_PATH} ({len(model.names) if hasattr(model, 'names') else '?'} classes)", flush=True)

# Load Grounding DINO (open-vocabulary detection) when enabled. Failing to
# load is NOT fatal — the /detect endpoint falls back to COCO-only for
# every request (as if no prompt were provided). This keeps the service
# resilient to a bad HuggingFace fetch or a transformers install glitch.
gd_model = None
gd_processor = None
gd_torch = None
if OPEN_VOCAB_ENABLED:
    try:
        import torch as _torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gd_torch = _torch
        gd_processor = AutoProcessor.from_pretrained(OPEN_VOCAB_MODEL)
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(OPEN_VOCAB_MODEL)
        gd_model.eval()
        print(f"💬 open-vocab (Grounding DINO) loaded: {OPEN_VOCAB_MODEL} (box_conf≥{OPEN_VOCAB_CONF}, text_conf≥{OPEN_VOCAB_TEXT_CONF})", flush=True)
    except Exception as e:
        print(f"⚠️  open-vocab load failed ({OPEN_VOCAB_MODEL}): {e} — /detect will ignore prompt field and run COCO-only", flush=True)
        gd_model = None
        gd_processor = None
        gd_torch = None
else:
    print("💬 open-vocab DISABLED (YOLO_OPEN_VOCAB_ENABLED=false)", flush=True)

# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────
def safe_crop(image_np, box):
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_np[y1:y2, x1:x2]
    # image_np is RGB (PIL → np.array on the /detect path; explicit BGR→RGB
    # conversion on the /detect-video path at line 347). cv2.imencode assumes
    # BGR ordering, so without this conversion the encoded JPEG has R and B
    # swapped — every cropped detection sent to downstream identification was
    # being read with skin/hair/clothing colors flipped (e.g. blue bikini
    # rendered as brown/yellow). The hero frame path is unaffected because
    # frame_to_base64_jpeg is called with the raw cv2 frame (already BGR).
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8") if ok else None

def frame_to_base64_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode('utf-8')

def _run_yolo(model_obj, img_np, conf=None):
    """Model-agnostic YOLOv8 predict. Returns [[x1,y1,x2,y2,conf,cls_idx], ...].
    Callers pass the specific model + optional confidence override; the rest
    of the tuning (IOU, IMGSZ, MAXDET) stays global.
    """
    r = model_obj.predict(
        img_np, conf=(conf if conf is not None else YOLO_CONF), iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAXDET, agnostic_nms=True, augment=False, verbose=False
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf_arr = r.boxes.conf.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy()
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), int(k)]
            for b, c, k in zip(xyxy, conf_arr, cls)]

def run_yolo(img_np):
    """Primary YOLO (COCO). Thin wrapper preserved so existing callers work
    unchanged."""
    return _run_yolo(model, img_np)

def tile_infer(img_np):
    """Primary YOLO tiled inference. Used when USE_TILING=1 (default) so
    small items on large images survive."""
    H, W = img_np.shape[:2]
    if max(H, W) <= TILE:
        return run_yolo(img_np)

    stride = max(1, int(TILE * (1 - OVERLAP)))
    boxes, confs, classes = [], [], []

    y = 0
    while True:
        x = 0
        y2 = min(y + TILE, H)
        while True:
            x2 = min(x + TILE, W)
            patch = img_np[y:y2, x:x2]
            for bx1, by1, bx2, by2, c, k in run_yolo(patch):
                boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                confs.append(c)
                classes.append(k)
            if x2 >= W: break
            x += stride
        if y2 >= H: break
        y += stride

    if not boxes: return []
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(confs, dtype=torch.float32)
    keep = nms(b, s, YOLO_IOU).tolist()
    return [[*boxes[i], float(confs[i]), int(classes[i])] for i in keep]

# ── Open-vocab optimizations (Tier 6 from the scaling ladder) ─────
# Cap the longer edge before feeding Grounding DINO. The processor
# auto-resizes internally but doing it CPU-side first saves ~30% wall
# per image on 2000x2000 catalog photos (measured locally). The model's
# recall is unaffected — 800px is well above what Grounding DINO's
# internal receptive field can use anyway. Set to 0 to disable.
GD_MAX_LONG_EDGE = max(0, int(os.getenv("YOLO_OPEN_VOCAB_MAX_LONG_EDGE", "800")))

def _downscale_for_gd(image_pil):
    """Scale-preserving resize so the longer edge <= GD_MAX_LONG_EDGE.
    Returns (image_pil, scale_x, scale_y) so bbox coordinates can be
    projected back to the ORIGINAL image space before returning to the
    caller."""
    if GD_MAX_LONG_EDGE <= 0:
        return image_pil, 1.0, 1.0
    w, h = image_pil.size
    long_edge = max(w, h)
    if long_edge <= GD_MAX_LONG_EDGE:
        return image_pil, 1.0, 1.0
    scale = GD_MAX_LONG_EDGE / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image_pil.resize((new_w, new_h), Image.BILINEAR)
    # Return the INVERSE scale so callers multiply bbox coords back up.
    return resized, (w / new_w), (h / new_h)

def run_grounding_dino(image_pil, prompt: str):
    """Open-vocabulary detection. Returns a list of dicts (NOT the
    [x1,y1,x2,y2,conf,cls_idx] tuple shape used by YOLO/OpenCV/OAI) because
    Grounding DINO's labels are TEXT extracted from the prompt, not integer
    class indices. Skipping merge_nms for these is intentional — Grounding
    DINO alone is the primary detector when it fires, and it already emits
    tight per-object bboxes without needing NMS from a second source.

    Returns [] on any failure (model not loaded, empty prompt, runtime
    error) so callers can safely fall back to the YOLO path.
    """
    if gd_model is None or gd_processor is None or gd_torch is None:
        return []
    if not prompt or not prompt.strip():
        return []
    # Downscale first — bboxes in the model's output are in the resized
    # image's coord space, which we project back before returning.
    scaled_pil, scale_x, scale_y = _downscale_for_gd(image_pil)
    try:
        inputs = gd_processor(images=scaled_pil, text=prompt, return_tensors="pt")
        with gd_torch.no_grad():
            outputs = gd_model(**inputs)
        target_sizes = gd_torch.tensor([scaled_pil.size[::-1]])  # (H, W)
        # transformers >=4.44 collapsed `box_threshold`+`text_threshold`
        # into a single `threshold`. Older releases had both. Try the new
        # signature first, fall back for older transformers.
        try:
            results = gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=OPEN_VOCAB_CONF, text_threshold=OPEN_VOCAB_TEXT_CONF,
                target_sizes=target_sizes
            )[0]
        except TypeError:
            results = gd_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=OPEN_VOCAB_CONF, text_threshold=OPEN_VOCAB_TEXT_CONF,
                target_sizes=target_sizes
            )[0]
    except Exception as e:
        print(f"⚠️  Grounding DINO runtime error: {e}", flush=True)
        return []

    # Return list of dicts directly, sorted by confidence descending. Each
    # detection carries `label` (text, from the prompt) and `bbox` in
    # (x1,y1,x2,y2) pixel coords — projected back to the ORIGINAL image
    # space (multiply by scale_x/scale_y since we downscaled before infer).
    out = []
    for score, label, box in zip(results.get("scores", []), results.get("labels", []) or results.get("text_labels", []), results.get("boxes", [])):
        try:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        except Exception:
            continue
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        out.append({
            "label":      label if isinstance(label, str) else str(label),
            "confidence": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out

def propose_rectangles(image_np):
    """Shape-based box proposals — catches products that YOLO isn't trained on."""
    H, W = image_np.shape[:2]
    min_area = RECT_MIN_AREA_FRAC * H * W
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    props = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area: continue
        ar = w / max(1.0, h)
        if ar < 0.3 or ar > 3.5: continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if box_area <= 0 or area / box_area < 0.65: continue
        # Class -1 marks non-YOLO origin so we can label it 'object' downstream.
        props.append([float(x), float(y), float(x + w), float(y + h), 0.35, -1])

    props.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return props[:RECT_MAX]

def oai_box_proposals(image_bytes, H, W):
    """Ask gpt-4o-mini for product bounding boxes. Used only when YOLO+rects look sparse."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        sys = (
            "You output bounding boxes for every retail product visible. "
            "Use normalized [x1,y1,x2,y2] in 0..1. Confidence 0..1. "
            "Labels should be broad types like 'bottle','jar','box','pouch','bag','carton','tube'."
        )
        user_text = "Return strict JSON with key 'detections'. Avoid duplicates; be exhaustive."
        resp = client.chat.completions.create(
            model=OAI_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=OAI_TIMEOUT,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        out = []
        for d in data.get("detections", []):
            box = d.get("box", [])
            conf = float(d.get("confidence", 0.55))
            if not (isinstance(box, list) and len(box) == 4): continue
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(1.0, float(x1))) * W
            y1 = max(0.0, min(1.0, float(y1))) * H
            x2 = max(0.0, min(1.0, float(x2))) * W
            y2 = max(0.0, min(1.0, float(y2))) * H
            # Class -2 marks OpenAI origin.
            out.append([x1, y1, x2, y2, conf, -2])
        return out
    except Exception as e:
        app.logger.warning(f"OpenAI fallback failed: {e}")
        return []

def merge_nms(*box_lists, iou_intra=0.55, iou_cross=0.80):
    """Class-aware NMS across multiple proposal sources.

    Boxes carry source identity in cls_idx:
      cls_idx >= 0 → YOLO (real COCO class)
      cls_idx == -1 → OpenCV rectangle proposal
      cls_idx == -2 → gpt-4o-mini box fallback

    Two-pass merge:
      1. Intra-source NMS at iou_intra (tight, default 0.55) — drops
         duplicate proposals from the same source covering the same region.
      2. Cross-source NMS at iou_cross (loose, default 0.80) — only
         suppresses cross-source overlaps when boxes truly cover the same
         pixels. Sub-region boxes (e.g. an OpenCV bikini-top rect inside
         a YOLO 'person' box) naturally have low IoU and survive both
         passes; near-duplicate cross-source boxes (e.g. YOLO 'tie' at
         IoU 0.85 with an OpenCV 'object' rect on the same tie) get
         deduped, with the higher-confidence proposal winning.

    Preserves cls_idx from each surviving box so downstream
    class_name_for resolves to the right source label.
    """
    merged = [b for lst in box_lists for b in (lst or [])]
    if not merged: return []

    def source_of(box):
        ci = box[5]
        if ci == -1: return 'opencv'
        if ci == -2: return 'oai'
        return 'yolo'

    # Pass 1 — intra-source NMS, tight threshold per source bucket.
    groups = {}
    for box in merged:
        groups.setdefault(source_of(box), []).append(box)

    survivors = []
    for src_boxes in groups.values():
        if len(src_boxes) <= 1:
            survivors.extend(src_boxes)
            continue
        b = torch.tensor([m[:4] for m in src_boxes], dtype=torch.float32)
        s = torch.tensor([m[4] for m in src_boxes], dtype=torch.float32)
        keep = nms(b, s, iou_intra).tolist()
        survivors.extend([src_boxes[i] for i in keep])

    # Pass 2 — cross-source NMS, loose threshold so sub-region OpenCV /
    # gpt-4o-mini boxes inside YOLO containers survive (low IoU) but
    # near-duplicate cross-source proposals still dedup (IoU > 0.80).
    if len(survivors) <= 1:
        return survivors
    b = torch.tensor([m[:4] for m in survivors], dtype=torch.float32)
    s = torch.tensor([m[4] for m in survivors], dtype=torch.float32)
    keep = nms(b, s, iou_cross).tolist()
    return [survivors[i] for i in keep]

def total_coverage(boxes, H, W):
    area = 0.0
    for b in boxes:
        area += max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    return area / float(H * W) if H and W else 0.0

def class_name_for(cls_idx):
    if cls_idx == -1: return "object"          # from OpenCV rects
    if cls_idx == -2: return "product"         # from OpenAI fallback
    # Grounding DINO doesn't route through class_name_for — its detections
    # skip merge_nms and land in the response with their text labels
    # attached directly. So no open-vocab range here.
    return model.names.get(int(cls_idx), "object") if hasattr(model, "names") else "object"

def run_full_detection(image_np, raw_bytes_for_oai=None, label="image"):
    """The COCO + rects + OAI pipeline. Fires when no prompt is provided
    on /detect (UGC path). Returns list of [x1,y1,x2,y2,conf,cls_idx].
    """
    H, W = image_np.shape[:2]

    # 1. YOLO COCO (tiled).
    preds = tile_infer(image_np) if USE_TILING else run_yolo(image_np)
    yolo_count = len(preds)

    # 2. OpenCV rectangles.
    rect_count = 0
    if FALLBACK_RECT:
        rects = propose_rectangles(image_np)
        rect_count = len(rects)
        # Class-aware merge — OpenCV sub-region boxes (e.g. clothing items
        # inside a YOLO 'person' container) survive at the loose cross-source
        # threshold; intra-source duplicates still get tight dedup.
        preds = merge_nms(preds, rects)

    # 3. OpenAI fallback — only when recall looks weak. Kept for the UGC
    #    path (no prompt); catalog Media hit the open-vocab path instead
    #    and never reach this stage.
    oai_count = 0
    if OAI_BOX_FALLBACK and raw_bytes_for_oai is not None:
        cov = total_coverage(preds, H, W)
        if len(preds) < OAI_TRIGGER_MIN_DETS or cov < OAI_TRIGGER_MIN_COVER:
            oai = oai_box_proposals(raw_bytes_for_oai, H, W)
            oai_count = len(oai)
            if oai:
                preds = merge_nms(preds, oai)

    if VERBOSE:
        print(f"🔎 {label}: yolo={yolo_count} rects={rect_count} openai={oai_count} merged={len(preds)}", flush=True)
    return preds

def _source_model_for(cls_idx):
    """Which producer emitted this detection. Written into the response so
    the backend can log per-source recall by brand without another call.
    Backend doesn't read this field yet — it's an observability channel."""
    if cls_idx == -1: return 'opencv'
    if cls_idx == -2: return 'openai'
    return 'coco'

def make_detection(image_np, pred, img_w, img_h, first_seen_sec=None):
    x1, y1, x2, y2, conf, cls_idx = pred
    b64 = safe_crop(image_np, (x1, y1, x2, y2))
    if not b64:
        return None
    det = {
        'base64':       b64,
        'confidence':   round(float(conf), 3),
        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
        'class_name':   class_name_for(cls_idx),
        'source_model': _source_model_for(cls_idx),
        'img_width':    img_w,
        'img_height':   img_h,
    }
    if first_seen_sec is not None:
        det['first_seen_sec'] = round(first_seen_sec, 2)
    return det

def iou_box(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def is_duplicate(box, cls, seen):
    for sb, sc in seen:
        if sc == cls and iou_box(box, sb) > IOU_DEDUP:
            return True
    return False

# ──────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return "ok", 200

def _make_gd_detection(image_np, gd_det, img_w, img_h):
    """Package a Grounding DINO detection into the same response shape as
    make_detection (used by the COCO/OpenCV/OAI path). Backend consumers
    treat both shapes identically."""
    b64 = safe_crop(image_np, (gd_det["x1"], gd_det["y1"], gd_det["x2"], gd_det["y2"]))
    if not b64:
        return None
    return {
        'base64':       b64,
        'confidence':   round(float(gd_det["confidence"]), 3),
        'x1': int(gd_det["x1"]), 'y1': int(gd_det["y1"]),
        'x2': int(gd_det["x2"]), 'y2': int(gd_det["y2"]),
        'class_name':   gd_det["label"] or "product",
        'source_model': 'open-vocab',
        'img_width':    img_w,
        'img_height':   img_h,
    }


# ── Image decode with STRUCTURED errors (not Flask HTML 500) ─────
# Wraps PIL open so callers can distinguish "the URL you handed us
# was not an image" (permanent — should be marked and never retried)
# from a genuine service-side failure (transient — retry-safe). Before
# this, PIL.UnidentifiedImageError bubbled all the way to Flask's
# default HTML 500 handler, which the backend then classified as a
# generic http-500 and threw away 2 retry attempts per bad URL.
#
# Callers are expected to catch _DecodeError and translate it to a
# response — the class carries `code`, `http_status`, and the byte
# count for observability. `traceback.print_exc()` fires before the
# exception is raised so Render logs still carry the full trace even
# when the response body is a clean JSON envelope.
class _DecodeError(Exception):
    def __init__(self, code, http_status, message, bytes_len):
        self.code = code
        self.http_status = http_status
        self.message = message
        self.bytes_len = bytes_len
        super().__init__(message)

def _decode_image_or_raise(raw):
    """Decode a byte string into an RGB PIL image + EXIF transpose.
    Raises _DecodeError with an actionable code on any failure; the
    caller returns that as a JSON response with the right status."""
    if not raw:
        raise _DecodeError('empty-body', 400, 'zero-byte image body', 0)
    try:
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        return ImageOps.exif_transpose(image)
    except UnidentifiedImageError:
        # Most common failure in prod: media.fileUrl points at a Cloudinary
        # asset that no longer exists or is behind an HTML "not found" page.
        # Backend gets a HTTP 200 body of HTML, forwards it here verbatim.
        # PERMANENT — backend must mark the Media so it never re-queues.
        print(f"⚠️  decode: unidentified image ({len(raw)} bytes)", flush=True)
        raise _DecodeError('unidentified-image', 400, 'cannot decode image bytes', len(raw))
    except (OSError, ValueError) as e:
        # Truncated JPEG, malformed PNG chunk, decompression bomb refusal.
        # PERMANENT — the bytes are what they are.
        print(f"⚠️  decode: bad image ({len(raw)} bytes) — {e}", flush=True)
        raise _DecodeError('decode-error', 400, f'image decode failed: {str(e)[:200]}', len(raw))


# ── Last-resort Flask errorhandler ────────────────────────────────
# Anything an inline try/except missed lands here and returns JSON,
# never the Flask default HTML page. Backend's _callYolo classifies
# `err.response.data` as JSON — an HTML body throws it into the
# generic "http-500" bucket and re-queues the caller.
@app.errorhandler(Exception)
def _json_uncaught(e):
    # 4xx exceptions (Werkzeug abort) keep their own status; only
    # actual 5xx / uncaught land here as HTTP 500.
    status = getattr(e, 'code', 500) if hasattr(e, 'code') else 500
    if not isinstance(status, int) or status < 100 or status > 599:
        status = 500
    if status >= 500:
        # Print the trace to Render logs — the JSON body is deliberately
        # short so backend can log it without pollution.
        traceback.print_exc()
    return jsonify({
        'error': str(e)[:400] or 'internal error',
        'code':  'internal-error' if status >= 500 else 'client-error'
    }), status


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required', 'code': 'missing-image'}), 400

    raw = request.files['image'].read()
    try:
        image = _decode_image_or_raise(raw)
    except _DecodeError as de:
        return jsonify({
            'error': de.message,
            'code':  de.code,
            'bytes': de.bytes_len
        }), de.http_status
    img_w, img_h = image.size

    # Open-vocab fork — when a prompt is provided AND Grounding DINO is
    # loaded, run it INSTEAD OF the COCO+rects+OAI pipeline. The eval
    # (yolo_microservice/eval/) showed Grounding DINO alone hits 100%
    # detection at 100% correct labels on our catalog; running COCO in
    # parallel would ~3.5× latency for zero recall gain.
    prompt = (request.form.get('prompt') or '').strip()
    if prompt and gd_model is not None:
        gd_dets = run_grounding_dino(image, prompt)
        image_np = np.array(image)
        detections = []
        dropped = 0
        for gd in gd_dets:
            if gd["confidence"] < CONF_THRESHOLD:
                dropped += 1
                if VERBOSE:
                    print(f"   [gd drop conf<{CONF_THRESHOLD}] {gd.get('label')} conf={gd['confidence']:.3f}", flush=True)
                continue
            d = _make_gd_detection(image_np, gd, img_w, img_h)
            if d:
                detections.append(d)
                if VERBOSE:
                    print(f"   [gd keep] {d['class_name']} conf={d['confidence']:.3f} box=({d['x1']},{d['y1']})→({d['x2']},{d['y2']})", flush=True)
        print(f"💬 /detect (open-vocab) prompt='{prompt[:80]}' returning {len(detections)} detection(s), dropped {dropped}", flush=True)
        return jsonify({'width': img_w, 'height': img_h, 'detections': detections})

    # Fall-through (no prompt, or open-vocab not available) — existing
    # COCO+rects+OAI pipeline. Byte-identical behavior to pre-Grounding-DINO
    # deploys, so UGC callers (productMatchService, etc.) are unaffected.
    image_np = np.array(image)
    preds = run_full_detection(image_np, raw_bytes_for_oai=raw, label="/detect")

    detections = []
    dropped = 0
    for pred in preds:
        if pred[4] < CONF_THRESHOLD:
            if VERBOSE:
                print(f"   [drop conf<{CONF_THRESHOLD}] {class_name_for(pred[5])} conf={pred[4]:.3f}", flush=True)
            dropped += 1
            continue
        d = make_detection(image_np, pred, img_w, img_h)
        if d:
            detections.append(d)
            if VERBOSE:
                print(f"   [keep] {d['class_name']} conf={d['confidence']:.3f} box=({d['x1']},{d['y1']})→({d['x2']},{d['y2']})", flush=True)

    print(f"🎯 /detect returning {len(detections)} detection(s), dropped {dropped} below threshold {CONF_THRESHOLD}", flush=True)
    return jsonify({'width': img_w, 'height': img_h, 'detections': detections})


@app.route('/detect-batch', methods=['POST'])
def detect_batch():
    """Batch endpoint for high-throughput ingest paths. Accepts N images
    (all under the multipart field name 'image') and N optional prompts
    (form field 'prompts', JSON-encoded array parallel to the image list).
    Response: {results: [{width, height, detections}, ...]} in the same
    order as the images.

    Per-image errors are isolated: one image that fails to decode or
    inference doesn't fail the whole batch — that slot returns an empty
    detections array with an 'error' field so the caller can retry
    individually if it cares.

    Batch value: amortizes HTTP + Flask + Python invocation overhead
    (~30% wall reduction per image on the observed CPU box) and lets
    Grounding DINO's transformers processor batch model.forward() when
    all images resize to the same target shape (a further ~10-15%).

    Caller-side batching lives in the backend (services/yoloService.js
    detectBatch + services/catalogYoloDetectionService per-product
    batching). Batch size is bounded there — the microservice will
    accept whatever it's given up to a memory-safe ceiling.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'At least one image is required'}), 400

    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': 'At least one image is required'}), 400

    # Parse prompts (JSON-encoded array parallel to files). Missing or
    # empty → all-empty prompts, which routes each image through the
    # COCO+rects+OAI pipeline (same behaviour as /detect without prompt).
    prompts_raw = request.form.get('prompts', '')
    prompts = []
    if prompts_raw:
        try:
            prompts = json.loads(prompts_raw)
            if not isinstance(prompts, list):
                return jsonify({'error': "'prompts' must be a JSON array"}), 400
        except json.JSONDecodeError:
            return jsonify({'error': "'prompts' is not valid JSON"}), 400
    # Pad/truncate to match file count. Extra prompts = ignored;
    # missing prompts = empty (COCO path for that slot).
    if len(prompts) < len(files):
        prompts = prompts + [''] * (len(files) - len(prompts))
    prompts = prompts[:len(files)]

    results = []
    open_vocab_count = 0
    coco_count = 0
    error_count = 0
    for i, (f, prompt) in enumerate(zip(files, prompts)):
        try:
            raw = f.read()
            try:
                image = _decode_image_or_raise(raw)
            except _DecodeError as de:
                # Per-slot permanent failure — do NOT let the caller retry.
                # The `code` field is what backend/detectYoloForMediaBatch
                # reads to stamp the Media as bad-source and skip re-queue.
                error_count += 1
                results.append({
                    'width': 0, 'height': 0, 'detections': [],
                    'error': de.message, 'code': de.code, 'bytes': de.bytes_len
                })
                continue
            img_w, img_h = image.size

            prompt_clean = (prompt or '').strip()
            if prompt_clean and gd_model is not None:
                # Open-vocab path — skip base64 crop generation. Backend
                # catalog synthesis path doesn't consume cropBuffer; UGC
                # never comes here because it has no prompt. Saves ~200ms
                # per image with detections. Set the empty base64 field
                # to preserve response shape backwards compat.
                gd_dets = run_grounding_dino(image, prompt_clean)
                detections = []
                for gd in gd_dets:
                    if gd["confidence"] < CONF_THRESHOLD:
                        continue
                    detections.append({
                        'base64':       '',   # deliberately skipped for open-vocab
                        'confidence':   round(float(gd["confidence"]), 3),
                        'x1': int(gd["x1"]), 'y1': int(gd["y1"]),
                        'x2': int(gd["x2"]), 'y2': int(gd["y2"]),
                        'class_name':   gd["label"] or "product",
                        'source_model': 'open-vocab',
                        'img_width':    img_w,
                        'img_height':   img_h,
                    })
                results.append({'width': img_w, 'height': img_h, 'detections': detections})
                open_vocab_count += 1
            else:
                # No prompt or open-vocab unavailable — legacy COCO path.
                image_np = np.array(image)
                preds = run_full_detection(image_np, raw_bytes_for_oai=raw, label=f"/detect-batch#{i}")
                detections = []
                for pred in preds:
                    if pred[4] < CONF_THRESHOLD:
                        continue
                    d = make_detection(image_np, pred, img_w, img_h)
                    if d:
                        detections.append(d)
                results.append({'width': img_w, 'height': img_h, 'detections': detections})
                coco_count += 1
        except Exception as e:
            # Something we DIDN'T anticipate. Print the trace so Render
            # logs preserve the shape (unlike the per-image _DecodeError
            # branch above, this is genuinely unexpected).
            error_count += 1
            traceback.print_exc()
            print(f"⚠️  /detect-batch item #{i} failed: {e}", flush=True)
            results.append({
                'width': 0, 'height': 0, 'detections': [],
                'error': str(e)[:200], 'code': 'inference-error'
            })

    print(f"📦 /detect-batch returning {len(results)} result(s) — open_vocab={open_vocab_count} coco={coco_count} errors={error_count}", flush=True)
    return jsonify({'results': results})


@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Video file is required'}), 400

    video_file = request.files['video']
    suffix = os.path.splitext(video_file.filename or '.mp4')[1] or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        video_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(video_fps / VIDEO_SAMPLE_FPS))
        img_w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        seen, detections, frame_idx = [], [], 0
        best_frame       = None
        best_frame_count = -1
        best_frame_sec   = 0.0
        hero_reason      = 'fallback'

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            # cv2 frame is BGR; YOLO/ultralytics handles it, but the rectangle proposals
            # and crop encoding expect RGB for PIL consistency elsewhere. Keep as-is for
            # speed (cv2.imencode in safe_crop handles the native format fine).
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = run_full_detection(rgb_frame, raw_bytes_for_oai=None, label=f"frame@{frame_idx/video_fps:.1f}s")
            t_sec = frame_idx / video_fps

            frame_det_count = 0
            for pred in preds:
                if pred[4] < CONF_THRESHOLD:
                    continue
                frame_det_count += 1
                if is_duplicate(pred[:4], pred[5], seen):
                    continue
                seen.append((pred[:4], pred[5]))
                d = make_detection(rgb_frame, pred, img_w, img_h, first_seen_sec=t_sec)
                if d:
                    detections.append(d)

            if frame_det_count > best_frame_count:
                best_frame_count = frame_det_count
                best_frame = frame.copy()
                best_frame_sec = t_sec
                hero_reason = f'highest-detection-count ({frame_det_count})'

            frame_idx += 1

        cap.release()

        if best_frame is None:
            cap2 = cv2.VideoCapture(tmp_path)
            total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            mid = total // 2
            cap2.set(cv2.CAP_PROP_POS_FRAMES, mid)
            _, best_frame = cap2.read()
            best_frame_sec = mid / video_fps
            hero_reason = 'middle-frame fallback (no detections)'
            cap2.release()

        hero_frame_b64 = frame_to_base64_jpeg(best_frame) if best_frame is not None else None

        return jsonify({
            'width': img_w,
            'height': img_h,
            'detections': detections,
            'hero_frame': hero_frame_b64,
            'hero_frame_sec': round(best_frame_sec, 2),
            'hero_reason': hero_reason,
            'video_duration_sec': round(frame_idx / video_fps, 2)
        })

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
