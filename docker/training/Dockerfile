# Training image: mount host ``data/`` with labels + JP2s, or copy JSONL + scenes into the image.
# Example:
#   docker build -f docker/training/Dockerfile -t aquaforge-train .
#   docker run --rm -v %cd%/data:/app/data aquaforge-train

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY aquaforge ./aquaforge
COPY scripts ./scripts

ENV PYTHONPATH=/app

CMD ["python", "scripts/train_aquaforge.py", "--epochs", "4", "--batch-size", "2"]
