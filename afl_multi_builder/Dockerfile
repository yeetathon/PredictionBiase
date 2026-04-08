FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create required directories
RUN mkdir -p data/models data/logs

# Expose API port
EXPOSE 8000

# Default command: seed data then start API
CMD ["sh", "-c", "python scripts/seed_demo_data.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
