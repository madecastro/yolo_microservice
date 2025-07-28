FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev && \
    pip install --upgrade pip && \
    pip install flask pillow opencv-python-headless numpy && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install ultralytics

EXPOSE 5000
CMD ["python", "yolo_service.py"]