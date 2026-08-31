# python:3.11-slim (was python:3.9-slim through 2026-08-30). Modern
# transformers releases require Python >=3.10 for full API compatibility
# with the Grounding DINO processor; installing on 3.9 either failed
# outright or picked up an older transformers with a broken post-process
# signature. 3.11 is stable and still slim. If you need to pin lower for
# an unrelated reason, pin transformers to <5.0.0 explicitly.
FROM python:3.11-slim

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
# Simple form — no f-string quote nesting to trip syntax quirks across
# Python versions. Shell expands $YOLO_OPEN_VOCAB_MODEL before Python sees
# the source.
RUN python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; AutoProcessor.from_pretrained('$YOLO_OPEN_VOCAB_MODEL'); AutoModelForZeroShotObjectDetection.from_pretrained('$YOLO_OPEN_VOCAB_MODEL'); print('pre-loaded: $YOLO_OPEN_VOCAB_MODEL')"

EXPOSE 5000
# Stdout unbuffered so gunicorn / Flask print() output streams to Render logs in real time.
ENV PYTHONUNBUFFERED=1

# Production WSGI via gunicorn, configured by gunicorn.conf.py which
# reads from GUNICORN_* env vars. Tunables (workers / threads / timeout /
# max-requests / preload) can be changed in the Render dashboard without
# rebuilding the Docker image.
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "yolo_service:app"]
