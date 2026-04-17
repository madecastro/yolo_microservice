FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py .

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt \
    && pip install flask opencv-python-headless ultralytics \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000
CMD ["python", "yolo_service.py"]
