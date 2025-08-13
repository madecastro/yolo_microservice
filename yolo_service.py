from flask import Flask, request, jsonify
from ultralytics import YOLO
import os, cv2, base64, numpy as np, torch
from PIL import Image
from torchvision.ops import nms

app = Flask(__name__)

# -------- Tunables via env (sane defaults) --------
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8x.pt")
CONF      = float(os.getenv("YOLO_CONF", "0.35"))
IOU       = float(os.getenv("YOLO_IOU", "0.50"))
IMGSZ     = int(os.getenv("YOLO_IMGSZ", "960"))
MAX_DET   = int(os.getenv("YOLO_MAX_DET", "300"))
USE_TILING = os.getenv("YOLO_TILING", "1") == "1"           # set 0 to disable tiling
TILE       = int(os.getenv("YOLO_TILE", "960"))             # tile size in px
OVERLAP    = float(os.getenv("YOLO_TILE_OVERLAP", "0.25"))  # 25% overlap

model = YOLO(MODEL_PATH)

def safe_crop(image_np, box):
    """Clamp to image bounds and JPEG-encode. Returns base64 or None."""
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
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")

def run_yolo(img):
    """Run a single pass and return list of [x1,y1,x2,y2,conf]."""
    res = model.predict(
        img,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device="cpu",
        augment=True,          # small TTA helps edges
        agnostic_nms=True,
        max_det=MAX_DET,
        verbose=False
    )[0]

    if res.boxes is None or len(res.boxes) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    return [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)] for b, c in zip(xyxy, conf)]

def tile_infer(image_np):
    """Slide a window over the whole image, merge all boxes with NMS."""
    H, W = image_np.shape[:2]
    stride = max(1, int(TILE * (1 - OVERLAP)))
    all_boxes, all_confs = [], []

    y = 0
    while True:
        x = 0
        y2 = min(y + TILE, H)
        while True:
            x2 = min(x + TILE, W)
            patch = image_np[y:y2, x:x2]
            for bx1, by1, bx2, by2, c in run_yolo(patch):
                all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                all_confs.append(c)
            if x2 >= W:
                break
            x += stride
        if y2 >= H:
            break
        y += stride

    if not all_boxes:
        return []
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    confs_t = torch.tensor(all_confs, dtype=torch.float32)
    keep = nms(boxes_t, confs_t, IOU)
    return [[*all_boxes[i], float(all_confs[i])] for i in keep.tolist()]

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    image = Image.open(request.files['image'].stream).convert('RGB')
    image_np = np.array(image)

    preds = tile_infer(image_np) if USE_TILING and max(image_np.shape[:2]) > TILE else run_yolo(image_np)

    detections = []
    for x1, y1, x2, y2, conf in preds:
        if conf < CONF:
            continue
        b64 = safe_crop(image_np, (x1, y1, x2, y2))
        if not b64:
            continue
        detections.append({
            'base64': b64,
            'confidence': round(float(conf), 3),
            'box': [int(x1), int(y1), int(x2), int(y2)]  # helpful for debugging / visualization
        })

    return jsonify(detections)

if __name__ == '__main__':
    # For local dev only; Render will run via your CMD
    app.run(host='0.0.0.0', port=5000)
