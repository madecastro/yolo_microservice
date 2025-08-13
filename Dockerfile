FROM python:3.9-slim

ENV YOLO_CONFIG_DIR=/tmp \
    YOLO_MODEL=yolov8x.pt \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY yolo_service.py .

# Runtime deps (no -dev; libgl1 replaces libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir flask pillow opencv-python-headless numpy gunicorn && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics

# Pre-download weights to avoid cold-start
RUN python - <<'PY'
from ultralytics import YOLO
import os
YOLO(os.getenv("YOLO_MODEL","yolov8x.pt"))
PY

EXPOSE 5000

# Use Gunicorn in production; preload loads the model once
CMD ["bash","-lc","exec gunicorn --preload -w 2 -k gthread --threads 2 -t 180 -b 0.0.0.0:${PORT:-5000} yolo_service:app"]
