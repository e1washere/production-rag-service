# Production Dockerfile for RAG Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/simple_api.py ./app/
COPY app/logging.py ./app/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=3s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# Start application
CMD ["uvicorn", "app.simple_api:app", "--host", "0.0.0.0", "--port", "8000"]
