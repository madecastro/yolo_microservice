FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install flask gunicorn pillow opencv-python-headless numpy ultralytics openai pandas seaborn \
    && rm -rf /var/lib/apt/lists/*

# Pre-download YOLOv8x weights at build time so first request doesn't cold-start.
ENV YOLO_CONFIG_DIR=/tmp
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"

EXPOSE 5000
# Stdout unbuffered so gunicorn / Flask print() output streams to Render logs in real time.
ENV PYTHONUNBUFFERED=1

# Production WSGI via gunicorn. Single worker because YOLOv8x holds
# ~500MB+ resident; multiple workers would each load their own model
# copy and OOM the instance. Single-threaded because PyTorch models
# aren't reliably thread-safe — concurrent requests queue inside
# gunicorn rather than racing inside Python. Long timeout for video
# frame extraction (can take 30-60s on large clips).
CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "1", \
     "--timeout", "300", \
     "--bind", "0.0.0.0:5000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "yolo_service:app"]
