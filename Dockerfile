FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py gunicorn.conf.py ./

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install flask gunicorn pillow opencv-python-headless numpy ultralytics openai pandas seaborn transformers \
    && rm -rf /var/lib/apt/lists/*

# Pre-download YOLOv8x weights at build time so first request doesn't cold-start.
ENV YOLO_CONFIG_DIR=/tmp
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"

# ── Pre-download Grounding DINO at build time ────────────────────────
# Open-vocabulary detection model. When YOLO_OPEN_VOCAB_ENABLED=true at
# runtime, the service loads this at boot and serves prompt-driven
# detection on /detect for any request that includes a `prompt` form
# field. Pre-loading here saves ~200MB of first-request cold-start.
#
# Model choice is env-configurable via YOLO_OPEN_VOCAB_MODEL. Default is
# grounding-dino-tiny (~200MB, fastest); grounding-dino-base is ~700MB
# with higher recall on cluttered images — swap via dashboard env.
ARG YOLO_OPEN_VOCAB_MODEL="IDEA-Research/grounding-dino-tiny"
RUN python -c "\
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
p = AutoProcessor.from_pretrained('$YOLO_OPEN_VOCAB_MODEL'); \
m = AutoModelForZeroShotObjectDetection.from_pretrained('$YOLO_OPEN_VOCAB_MODEL'); \
print(f'✓ pre-loaded {\"$YOLO_OPEN_VOCAB_MODEL\"}')"

EXPOSE 5000
# Stdout unbuffered so gunicorn / Flask print() output streams to Render logs in real time.
ENV PYTHONUNBUFFERED=1

# Production WSGI via gunicorn, configured by gunicorn.conf.py which
# reads from GUNICORN_* env vars. Tunables (workers / threads / timeout /
# max-requests / preload) can be changed in the Render dashboard without
# rebuilding the Docker image.
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "yolo_service:app"]
