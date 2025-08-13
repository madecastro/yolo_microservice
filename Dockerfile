FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY yolo_service.py .

# NOTE: libgl1-mesa-glx → libgl1, -dev → runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir flask pillow opencv-python-headless numpy && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics

EXPOSE 5000
CMD ["python", "yolo_service.py"]
