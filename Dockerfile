# Multi-stage lightweight production container for Crop Leaf Disease Classification API
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal OS dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code, configuration, metadata, and models
COPY config.yaml /app/
COPY data/supported_classes.json /app/data/
COPY data/class_weights.json /app/data/
COPY data/samples/ /app/data/samples/
COPY src/ /app/src/
COPY api/ /app/api/
COPY frontend/ /app/frontend/
COPY models/ /app/models/

# Expose standard FastAPI port
EXPOSE 8000

# Healthcheck to monitor API responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch production server with Uvicorn (supports Render dynamic $PORT and single worker for memory optimization)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
