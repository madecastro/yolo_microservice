FROM python:3.9-slim

WORKDIR /app
COPY yolo_service.py .

RUN apt-get update && apt-get install -y libgl1-mesa-glx && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install flask pillow opencv-python-headless numpy

EXPOSE 5000
CMD ["python", "yolo_service.py"]
