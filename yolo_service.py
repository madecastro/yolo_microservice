from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import cv2
import numpy as np

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def crop_and_encode(image_np, box):
    x1, y1, x2, y2 = map(int, box)
    crop = image_np[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', crop)
    return base64.b64encode(buffer).decode('utf-8')

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
        if conf < 0.4:
            continue
        crop_base64 = crop_and_encode(image_np, box)
        detections.append({
            'base64': crop_base64,
            'confidence': round(conf, 3)
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
