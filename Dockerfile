FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/tmp/ultralytics PYTHONUNBUFFERED=1
WORKDIR /srv

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake weights into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY app.py index.html ./
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]