import os, io, json, base64, cv2, numpy as np, torch
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
from torchvision.ops import nms

# -------- Env knobs --------
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8x.pt")
YOLO_CONF  = float(os.getenv("YOLO_CONF", "0.20"))
YOLO_IOU   = float(os.getenv("YOLO_IOU", "0.60"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_MAXDET= int(os.getenv("YOLO_MAX_DET", "300"))

USE_TILING = os.getenv("YOLO_TILING", "1") == "1"
TILE       = int(os.getenv("YOLO_TILE", "1024"))
OVERLAP    = float(os.getenv("YOLO_TILE_OVERLAP", "0.35"))

# CV rectangle proposals (good for boxes/pouches/packs)
FALLBACK_RECT = os.getenv("FALLBACK_RECT", "1") == "1"
RECT_MIN_AREA_FRAC = float(os.getenv("RECT_MIN_AREA_FRAC", "0.01"))
RECT_MAX = int(os.getenv("RECT_MAX", "12"))

# OpenAI box fallback (slow; only used if recall looks poor)
OAI_BOX_FALLBACK = os.getenv("OAI_BOX_FALLBACK", "1") == "1"
OAI_MODEL = os.getenv("OAI_MODEL", "gpt-4o-mini")  # cheap & fast; use gpt-4o if you want
OAI_TRIGGER_MIN_DETS = int(os.getenv("OAI_TRIGGER_MIN_DETS", "6"))
OAI_TRIGGER_MIN_COVER = float(os.getenv("OAI_TRIGGER_MIN_COVER", "0.22"))  # 22% area
OAI_TIMEOUT = int(os.getenv("OAI_TIMEOUT", "30"))  # seconds

# Ultralytics config dir (avoid write warnings in containers)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

app = Flask(__name__)
model = YOLO(MODEL_PATH)

# ---------- helpers ----------
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
    H, W = image_np.shape[:2]
    min_area = RECT_MIN_AREA_FRAC * H * W
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    props = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < min_area: continue
        ar = w / max(1.0, h)
        if ar < 0.3 or ar > 3.5: continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if box_area <= 0 or area/box_area < 0.65: continue
        props.append([float(x), float(y), float(x+w), float(y+h), 0.35])

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

# ---------- OpenAI fallback ----------
def oai_box_proposals(image_bytes, H, W):
    """Return list of [x1,y1,x2,y2,conf] from OpenAI (normalized -> pixels)."""
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
        user_text = ("Return strict JSON with key 'detections'. Avoid duplicates; be exhaustive.")
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
            # clamp and project to pixels
            x1 = max(0.0, min(1.0, float(x1))) * W
            y1 = max(0.0, min(1.0, float(y1))) * H
            x2 = max(0.0, min(1.0, float(x2))) * W
            y2 = max(0.0, min(1.0, float(y2))) * H
            out.append([x1,y1,x2,y2, conf])
        return out
    except Exception as e:
        app.logger.warning(f"OpenAI fallback failed: {e}")
        return []

# ---------- routes ----------
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

    # 1) YOLO (with tiling)
    preds = tile_infer(image_np) if USE_TILING else run_yolo(image_np)

    # 2) Rectangle proposals (CV)
    if FALLBACK_RECT:
        preds = merge_nms(preds, propose_rectangles(image_np), iou=0.55)

    # 3) OpenAI fallback — only if YOLO recall looks weak
    if OAI_BOX_FALLBACK:
        cov = total_coverage(preds, H, W)
        if len(preds) < OAI_TRIGGER_MIN_DETS or cov < OAI_TRIGGER_MIN_COVER:
            oai_boxes = oai_box_proposals(raw, H, W)
            if oai_boxes:
                preds = merge_nms(preds, oai_boxes, iou=0.55)

    # 4) Build response with crops
    out = []
    for x1,y1,x2,y2,conf in preds:
        if conf < YOLO_CONF:  # keep a simple confidence floor
            continue
        b64 = safe_crop(image_np, (x1,y1,x2,y2))
        if not b64: continue
        out.append({
            "base64": b64,
            "confidence": round(float(conf), 3),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return jsonify(out)
