FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py gunicorn.conf.py ./

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

# ── Pre-download apparel/DeepFashion2 weights at build time ──────────
# YOLO_APPAREL_MODEL_URL is a build ARG (not baked env). When provided,
# the checkpoint is downloaded into /app/yolov8x-df2.pt and ultralytics
# loads it once to prime any lazy caches. When empty, this step is a
# no-op — the runtime service will detect the missing file, log an INFO
# line, and continue COCO-only. Owner-selected checkpoint via
# --build-arg YOLO_APPAREL_MODEL_URL=<url> at `docker build`, or set on
# the Render dashboard's Build Command → Build Environment tab.
#
# Typical values would point at a HuggingFace or GitHub release URL for
# a YOLOv8 checkpoint trained on apparel data (DeepFashion2, Fashionpedia,
# etc.). Keeping this out of the source lets us swap checkpoints without
# a code change once the recall of a specific model is measured.
ARG YOLO_APPAREL_MODEL_URL=""
RUN if [ -n "$YOLO_APPAREL_MODEL_URL" ]; then \
      echo "📥 Downloading apparel model from $YOLO_APPAREL_MODEL_URL" && \
      python -c "import urllib.request; urllib.request.urlretrieve('$YOLO_APPAREL_MODEL_URL', '/app/yolov8x-df2.pt')" && \
      python -c "from ultralytics import YOLO; YOLO('/app/yolov8x-df2.pt')" && \
      ls -lh /app/yolov8x-df2.pt ; \
    else \
      echo "ℹ️  YOLO_APPAREL_MODEL_URL not set — apparel model will not be baked in" ; \
    fi

EXPOSE 5000
# Stdout unbuffered so gunicorn / Flask print() output streams to Render logs in real time.
ENV PYTHONUNBUFFERED=1

# Production WSGI via gunicorn, configured by gunicorn.conf.py which
# reads from GUNICORN_* env vars. Tunables (workers / threads / timeout /
# max-requests / preload) can be changed in the Render dashboard without
# rebuilding the Docker image.
CMD ["gunicorn", "--config", "/app/gunicorn.conf.py", "yolo_service:app"]
