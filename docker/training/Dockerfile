# Training image: mount host ``data/`` with labels + JP2s, or copy JSONL + scenes into the image.
# Example:
#   docker build -f docker/training/Dockerfile -t vessel-detection-train .
#   docker run --rm -v %cd%/data:/app/data vessel-detection-train

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vessel_detection ./vessel_detection
COPY scripts ./scripts

ENV PYTHONPATH=/app

CMD ["python", "scripts/train_all_models.py"]
