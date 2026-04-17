from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

CONF_THRESHOLD = 0.4
IOU_DEDUP_THRESHOLD = 0.3
VIDEO_SAMPLE_FPS = 2  # frames to sample per second of video


def crop_and_encode(image_np, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image_np[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', crop)
    return base64.b64encode(buffer).decode('utf-8')


def iou(box1, box2):
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def is_duplicate(box, cls, seen):
    """Return True if this box overlaps significantly with a seen box of the same class."""
    for seen_box, seen_cls in seen:
        if seen_cls == cls and iou(box, seen_box) > IOU_DEDUP_THRESHOLD:
            return True
    return False


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    results = model(image)

    image_np = np.array(image)
    detections = []

    for *box, conf, cls in results.xyxy[0].tolist():
        if conf < CONF_THRESHOLD:
            continue
        crop_base64 = crop_and_encode(image_np, box)
        detections.append({
            'base64': crop_base64,
            'confidence': round(conf, 3)
        })

    return jsonify(detections)


@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Video file is required'}), 400

    video_file = request.files['video']

    # Save to temp file — cv2.VideoCapture requires a file path
    suffix = os.path.splitext(video_file.filename or '.mp4')[1] or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(video_fps / VIDEO_SAMPLE_FPS))

        seen = []       # list of (box, cls) for deduplication
        detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            results = model(pil_image)

            for *box, conf, cls in results.xyxy[0].tolist():
                if conf < CONF_THRESHOLD:
                    continue
                if is_duplicate(box, cls, seen):
                    continue
                seen.append((box, cls))
                crop_base64 = crop_and_encode(frame, box)
                detections.append({
                    'base64': crop_base64,
                    'confidence': round(conf, 3)
                })

            frame_idx += 1

        cap.release()
        return jsonify(detections)

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
