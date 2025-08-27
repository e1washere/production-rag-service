# Builder
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install --no-cache-dir -e .
RUN mkdir -p /app/.cache/index
COPY .cache/index/ /app/.cache/index/

# Final
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/sample/ ./data/sample/
COPY data/docs/ ./data/docs/
RUN mkdir -p /app/.cache/index
COPY --from=builder /app/.cache/index/ /app/.cache/index/
ENV PYTHONUNBUFFERED=1
ENV INDEX_DIR=/app/.cache/index
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000"]
