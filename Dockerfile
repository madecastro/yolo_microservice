FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install flask pillow opencv-python-headless numpy ultralytics openai pandas seaborn \
    && rm -rf /var/lib/apt/lists/*

# Pre-download YOLOv8x weights at build time so first request doesn't cold-start.
ENV YOLO_CONFIG_DIR=/tmp
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"

EXPOSE 5000
# -u disables stdout buffering so print() output reaches Render logs in real time.
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "yolo_service.py"]
