# Multi-stage product detector. Restored from the pre-rewrite version that gave
# the best recall on real inventory photos, extended with EXIF handling, unified
# response shape, and the /detect-video endpoint we added later.
#
# Detection stages (image):
#   1. YOLOv8x with tiled inference (big model + sliding window catches small items)
#   2. OpenCV rectangle proposals (boxes/pouches/packs via contour shape)
#   3. OpenAI gpt-4o-mini box fallback (only when YOLO+rects look sparse)
#   4. NMS merge + confidence filter

import os, io, json, base64, cv2, numpy as np, torch, tempfile
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
from ultralytics import YOLO
from torchvision.ops import nms

# ── YOLO knobs ────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("YOLO_MODEL", "yolov8x.pt")
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU      = float(os.getenv("YOLO_IOU", "0.60"))
YOLO_IMGSZ    = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_MAXDET   = int(os.getenv("YOLO_MAX_DET", "300"))

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

def run_yolo(img_np):
    """YOLOv8 on a full image (or tile). Returns list of [x1,y1,x2,y2,conf,cls_idx]."""
    r = model.predict(
        img_np, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAXDET, agnostic_nms=True, augment=False, verbose=False
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy()
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), int(k)]
            for b, c, k in zip(xyxy, conf, cls)]

def tile_infer(img_np):
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
    return model.names.get(int(cls_idx), "object") if hasattr(model, "names") else "object"

def run_full_detection(image_np, raw_bytes_for_oai=None, label="image"):
    """The four-stage pipeline. Returns list of [x1,y1,x2,y2,conf,cls_idx]."""
    H, W = image_np.shape[:2]

    # 1. YOLO (tiled)
    preds = tile_infer(image_np) if USE_TILING else run_yolo(image_np)
    yolo_count = len(preds)

    # 2. OpenCV rectangles
    rect_count = 0
    if FALLBACK_RECT:
        rects = propose_rectangles(image_np)
        rect_count = len(rects)
        # Class-aware merge — OpenCV sub-region boxes (e.g. clothing items
        # inside a YOLO 'person' container) survive at the loose cross-source
        # threshold; intra-source duplicates still get tight dedup.
        preds = merge_nms(preds, rects)

    # 3. OpenAI fallback — only when recall looks weak
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

def make_detection(image_np, pred, img_w, img_h, first_seen_sec=None):
    x1, y1, x2, y2, conf, cls_idx = pred
    b64 = safe_crop(image_np, (x1, y1, x2, y2))
    if not b64:
        return None
    det = {
        'base64':     b64,
        'confidence': round(float(conf), 3),
        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
        'class_name': class_name_for(cls_idx),
        'img_width':  img_w,
        'img_height': img_h,
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

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    raw = request.files['image'].read()
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    image = ImageOps.exif_transpose(image)
    img_w, img_h = image.size
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
