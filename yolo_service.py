import os, io, json, base64
import cv2, numpy as np, torch
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
from torchvision.ops import nms

# =================== ENV KNOBS ===================
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8x.pt")
YOLO_CONF  = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU   = float(os.getenv("YOLO_IOU", "0.60"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_MAXDET= int(os.getenv("YOLO_MAX_DET", "300"))

USE_TILING = os.getenv("YOLO_TILING", "1") == "1"
TILE       = int(os.getenv("YOLO_TILE", "1024"))
OVERLAP    = float(os.getenv("YOLO_TILE_OVERLAP", "0.35"))

# CV rectangle proposals (for boxes/pouches/packs)
FALLBACK_RECT = os.getenv("FALLBACK_RECT", "1") == "1"
RECT_MIN_AREA_FRAC = float(os.getenv("RECT_MIN_AREA_FRAC", "0.006"))
RECT_MAX_AREA_FRAC = float(os.getenv("RECT_MAX_AREA_FRAC", "0.85"))
RECT_MIN_AR = float(os.getenv("RECT_MIN_AR", "0.45"))
RECT_MAX_AR = float(os.getenv("RECT_MAX_AR", "2.8"))
RECT_MIN_RECTANGULARITY = float(os.getenv("RECT_MIN_RECTANGULARITY", "0.78"))
RECT_MAX = int(os.getenv("RECT_MAX", "12"))

# OpenAI box fallback (slow; only if recall looks weak)
OAI_BOX_FALLBACK = os.getenv("OAI_BOX_FALLBACK", "1") == "1"
OAI_MODEL = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TRIGGER_MIN_DETS = int(os.getenv("OAI_TRIGGER_MIN_DETS", "6"))
OAI_TRIGGER_MIN_COVER = float(os.getenv("OAI_TRIGGER_MIN_COVER", "0.22"))
OAI_TIMEOUT = int(os.getenv("OAI_TIMEOUT", "30"))

# Open-vocab (OWL-ViT) fallback (CPU-OK; optional)
OV_ENABLED = os.getenv("OV_FALLBACK", "0") == "1"
OV_MODEL_ID = os.getenv("OV_MODEL_ID", "google/owlvit-base-patch32")
OV_CONF = float(os.getenv("OV_CONF", "0.20"))
OV_PROMPTS = os.getenv("OV_PROMPTS",
    "product, box, pouch, pack, bag, carton, tube, toothpaste, diaper pack, detergent pack")

# Ultralytics config dir (avoid container write warnings)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

# =================== APP/MODELS ===================
app = Flask(__name__)
model = YOLO(MODEL_PATH)

# Lazy OWL-ViT load
_ov_proc = _ov_model = None
def ensure_owlvit_loaded():
    global _ov_proc, _ov_model
    if _ov_proc is None:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        _ov_proc = OwlViTProcessor.from_pretrained(OV_MODEL_ID)
        _ov_model = OwlViTForObjectDetection.from_pretrained(OV_MODEL_ID)

# =================== HELPERS ===================
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
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8") if ok else None

def run_yolo(img_np):
    r = model.predict(
        img_np, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAXDET, agnostic_nms=True, augment=False, verbose=False
    )[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)]
            for b, c in zip(xyxy, conf)]

def tile_infer(img_np):
    H, W = img_np.shape[:2]
    if max(H, W) <= TILE:
        return run_yolo(img_np)

    stride = max(1, int(TILE * (1 - OVERLAP)))
    boxes, confs = [], []
    y = 0
    while True:
        x = 0
        y2 = min(y + TILE, H)
        while True:
            x2 = min(x + TILE, W)
            patch = img_np[y:y2, x:x2]
            for bx1, by1, bx2, by2, c in run_yolo(patch):
                boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                confs.append(c)
            if x2 >= W: break
            x += stride
        if y2 >= H: break
        y += stride

    if not boxes: return []
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(confs, dtype=torch.float32)
    keep = nms(b, s, YOLO_IOU).tolist()
    return [[*boxes[i], float(confs[i])] for i in keep]

def propose_rectangles(image_np):
    """Heuristic proposals for flat packages/boxes (no 'whole frame')."""
    H, W = image_np.shape[:2]
    min_area = RECT_MIN_AREA_FRAC * H * W
    max_area = RECT_MAX_AREA_FRAC * H * W

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 8)
    edges = cv2.Canny(gray, 60, 180)
    mask = cv2.bitwise_or(thr, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    props = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area or area > max_area:
            continue
        ar = w / max(1.0, h)
        if not (RECT_MIN_AR <= ar <= RECT_MAX_AR):
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if box_area <= 0 or (area / box_area) < RECT_MIN_RECTANGULARITY:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 10:
            continue

        props.append([float(x), float(y), float(x+w), float(y+h), 0.42])

    props.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return props[:RECT_MAX]

def merge_nms(*box_lists, iou=0.55):
    merged = [b for lst in box_lists for b in (lst or [])]
    if not merged: return []
    b = torch.tensor([m[:4] for m in merged], dtype=torch.float32)
    s = torch.tensor([m[4] for m in merged], dtype=torch.float32)
    keep = nms(b, s, iou).tolist()
    return [merged[i] for i in keep]

def total_coverage(boxes, H, W):
    area = 0.0
    for x1,y1,x2,y2,_ in boxes:
        area += max(0.0, (x2-x1)) * max(0.0, (y2-y1))
    return area / float(H*W) if H and W else 0.0

def oai_box_proposals(image_bytes, H, W):
    """OpenAI fallback → list [[x1,y1,x2,y2,conf], ...]"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"
        sys = ("You output bounding boxes for every retail product visible. "
               "Use normalized [x1,y1,x2,y2] in 0..1. Confidence 0..1. "
               "Labels should be broad types like 'bottle','jar','box','pouch','bag','carton','tube'.")
        user_text = "Return strict JSON {\"detections\":[{label,confidence,box:[x1,y1,x2,y2]}]} only."
        resp = client.chat.completions.create(
            model=OAI_MODEL,
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=OAI_TIMEOUT,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":[
                    {"type":"text","text":user_text},
                    {"type":"image_url","image_url":{"url": data_url}}
                ]}
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        dets = data.get("detections", [])
        out = []
        for d in dets:
            box = d.get("box", [])
            conf = float(d.get("confidence", 0.55))
            if not (isinstance(box, list) and len(box) == 4): continue
            x1,y1,x2,y2 = box
            x1 = max(0.0, min(1.0, float(x1))) * W
            y1 = max(0.0, min(1.0, float(y1))) * H
            x2 = max(0.0, min(1.0, float(x2))) * W
            y2 = max(0.0, min(1.0, float(y2))) * H
            out.append([x1,y1,x2,y2, conf])
        return out
    except Exception as e:
        app.logger.warning(f"OpenAI fallback failed: {e}")
        return []

def ov_boxes(image_np):
    """Open-vocab detector (OWL-ViT) fallback."""
    if not OV_ENABLED:
        return []
    try:
        ensure_owlvit_loaded()
        from transformers import OwlViTProcessor  # type: ignore
        inputs = _ov_proc(text=[OV_PROMPTS], images=Image.fromarray(image_np), return_tensors="pt")
        with torch.no_grad():
            outputs = _ov_model(**inputs)
        target_sizes = torch.tensor([image_np.shape[:2]])
        results = _ov_proc.post_process_object_detection(outputs, threshold=OV_CONF, target_sizes=target_sizes)[0]
        boxes, scores = results["boxes"], results["scores"]
        out = []
        for (x1,y1,x2,y2), s in zip(boxes.tolist(), scores.tolist()):
            out.append([float(x1), float(y1), float(x2), float(y2), float(s)])
        return out
    except Exception as e:
        app.logger.warning(f"OWL-ViT fallback failed: {e}")
        return []

# =================== ROUTES ===================
@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    raw = request.files["image"].read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # 1) YOLO (with optional tiling)
    preds = tile_infer(image_np) if USE_TILING else run_yolo(image_np)

    # 2) Rectangle proposals (CV)
    if FALLBACK_RECT:
        preds = merge_nms(preds, propose_rectangles(image_np), iou=0.55)

    # 3) If recall looks weak, try Open-vocab (OWL-ViT)
    cov = total_coverage(preds, H, W)
    if OV_ENABLED and (len(preds) < OAI_TRIGGER_MIN_DETS or cov < OAI_TRIGGER_MIN_COVER):
        preds = merge_nms(preds, ov_boxes(image_np), iou=0.55)

    # 4) If still weak, try OpenAI box proposals
    cov = total_coverage(preds, H, W)
    if OAI_BOX_FALLBACK and (len(preds) < OAI_TRIGGER_MIN_DETS or cov < OAI_TRIGGER_MIN_COVER):
        preds = merge_nms(preds, oai_box_proposals(raw, H, W), iou=0.55)

    # 5) Build response with crops
    out = []
    for x1,y1,x2,y2,conf in preds:
        if conf < YOLO_CONF:
            continue
        b64 = safe_crop(image_np, (x1,y1,x2,y2))
        if not b64: continue
        out.append({
            "base64": b64,
            "confidence": round(float(conf), 3),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return jsonify(out)

if __name__ == "__main__":
    # local dev only; in production use Gunicorn and $PORT
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
