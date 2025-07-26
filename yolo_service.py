from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # lightweight YOLOv8 model

def crop_and_encode(image_np, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image_np[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', crop)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    image = Image.open(request.files['image'].stream).convert('RGB')
    image_np = np.array(image)
    results = model(image_np)[0]

    detections = []
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        conf = float(box[4]) if len(box) > 4 else 0.5
        if conf < 0.4:
            continue
        crop_b64 = crop_and_encode(image_np, (x1, y1, x2, y2))
        detections.append({ 'base64': crop_b64, 'confidence': round(conf, 3) })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
