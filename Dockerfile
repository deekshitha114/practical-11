# ============================================================
# Dockerfile — Student Result Prediction API  (Practical-11)
# ============================================================
# Multi-stage-friendly, single-stage for classroom simplicity.
# Logs go to stdout so Render / any container platform captures them.

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="student-result-prediction"
LABEL description="Student Result Prediction FastAPI service - Practical 11"
LABEL version="1.0.0"

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY train_model.py .
COPY app.py .

# Train and persist the model at image-build time
RUN python train_model.py

# Expose the port uvicorn listens on
EXPOSE 8000

# Health check — Render / Docker will poll /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
