import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Allow overriding via env; default to the most robust model
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8x.pt")

# (Optional) keep Ultralytics config files writable in containers
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

# Preload model at import time
model = YOLO(MODEL_NAME)

def crop_and_encode(image_np, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
    crop = image_np[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".jpg", crop)
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    # Read and normalize to RGB
    image = Image.open(request.files["image"].stream).convert("RGB")
    image_np = np.array(image)

    # 🔧 Accuracy‑oriented inference settings (from rec #2)
    # - larger imgsz helps small parts (try 960–1024 on CPU)
    # - slightly lower conf to avoid missing borderline objects
    # - higher IoU for tighter NMS, class-agnostic to reduce dupes
    results = model.predict(
        image_np,
        imgsz=960,
        conf=0.20,
        iou=0.60,
        agnostic_nms=True,
        verbose=False
    )

    if not results:
        return jsonify([])

    r0 = results[0]
    boxes = r0.boxes
    names = getattr(getattr(model, "model", None), "names", None) or getattr(model, "names", None)

    detections = []
    for b in boxes:
        # xyxy, conf, class
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        cls_id = int(b.cls[0]) if b.cls is not None else -1
        label = (names[cls_id] if names and 0 <= cls_id < len(names) else None)

        # You can still filter if you want an extra safety margin
        if conf < 0.20:
            continue

        crop_b64 = crop_and_encode(image_np, (x1, y1, x2, y2))
        detections.append({
            "base64": crop_b64,
            "confidence": round(conf, 3),
            "cls": cls_id,
            "label": label
        })

    return jsonify(detections)

if __name__ == "__main__":
    # Local dev only; in production Gunicorn will run the app and bind $PORT
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
