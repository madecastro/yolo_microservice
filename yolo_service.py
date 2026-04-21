from flask import Flask, request, jsonify
import torch
from PIL import Image, ImageOps
import io
import base64
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Env-tunable so we can adjust without redeploying.
# Lower threshold = more (but noisier) detections.
CONF_THRESHOLD    = float(os.environ.get('CONF_THRESHOLD', 0.4))
IOU_DEDUP         = float(os.environ.get('IOU_DEDUP', 0.3))
VIDEO_SAMPLE_FPS  = float(os.environ.get('VIDEO_SAMPLE_FPS', 2))
VERBOSE_DETECTIONS = os.environ.get('YOLO_VERBOSE', 'true').lower() == 'true'


def crop_and_encode(image_np, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image_np[y1:y2, x1:x2]
    _, buf = cv2.imencode('.jpg', crop)
    return base64.b64encode(buf).decode('utf-8')


def frame_to_base64_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode('utf-8')


def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def is_duplicate(box, cls, seen):
    for sb, sc in seen:
        if sc == cls and iou(box, sb) > IOU_DEDUP:
            return True
    return False


def make_detection(image_np, box, conf, cls, img_w, img_h, first_seen_sec=None):
    x1, y1, x2, y2 = map(int, box)
    det = {
        'base64':     crop_and_encode(image_np, box),
        'confidence': round(conf, 3),
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'class_name': model.names[int(cls)],
        'img_width':  img_w,
        'img_height': img_h,
    }
    if first_seen_sec is not None:
        det['first_seen_sec'] = round(first_seen_sec, 2)
    return det


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    # PIL does NOT apply EXIF orientation by default, but Cloudinary's secure_url
    # auto-rotates when served. Normalize here so YOLO's (img_w, img_h) and
    # detection bboxes live in the same coord system the browser will render.
    image = Image.open(request.files['image'].stream).convert('RGB')
    image = ImageOps.exif_transpose(image)
    img_w, img_h = image.size
    results   = model(image)
    image_np  = np.array(image)
    raw       = results.xyxy[0].tolist()
    detections = []
    kept = 0
    dropped = 0

    for *box, conf, cls in raw:
        name = model.names[int(cls)]
        if conf < CONF_THRESHOLD:
            if VERBOSE_DETECTIONS:
                print(f"   [drop conf<{CONF_THRESHOLD}]  {name} conf={conf:.3f}")
            dropped += 1
            continue
        if VERBOSE_DETECTIONS:
            print(f"   [keep] {name} conf={conf:.3f} box=({int(box[0])},{int(box[1])})→({int(box[2])},{int(box[3])})")
        kept += 1
        detections.append(make_detection(image_np, box, conf, cls, img_w, img_h))

    print(f"🔎 /detect raw={len(raw)} kept={kept} dropped={dropped} (threshold={CONF_THRESHOLD})")
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
        best_frame        = None
        best_frame_count  = -1
        best_frame_sec    = 0.0
        hero_reason       = 'fallback'

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(pil_img)
            t_sec   = frame_idx / video_fps

            frame_det_count = 0
            for *box, conf, cls in results.xyxy[0].tolist():
                if conf < CONF_THRESHOLD:
                    continue
                frame_det_count += 1
                if is_duplicate(box, cls, seen):
                    continue
                seen.append((box, cls))
                detections.append(make_detection(frame, box, conf, cls, img_w, img_h, first_seen_sec=t_sec))

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
