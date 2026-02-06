FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (for building some python packages if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run commands
# 1. Ingest data (if not persisted volume, or handling via init container)
#    For this exercise, we'll try to run ingest on build or start if DB invalid, but usually bad practice for build.
#    We will assume data is ingested or volume mounted. 
#    Actually, per brief "Industrialisation", better to separate ingest, but we'll bundle for simplicity of the single pod exercise.

CMD ["sh", "-c", "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
