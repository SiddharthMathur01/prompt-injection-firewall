# Multi-stage Dockerfile for LLM Prompt Injection Firewall
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Copy project files
COPY config.yaml .
COPY rules/ rules/
COPY policy/ policy/
COPY classifier/ classifier/
COPY adjudicator/ adjudicator/
COPY service/ service/
COPY main.py .

# Create logs directory
RUN mkdir -p logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8080') + '/health')"]

CMD ["sh", "-c", "python -m service.main --host 0.0.0.0 --port ${PORT:-8080}"]
