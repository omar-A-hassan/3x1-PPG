FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .

EXPOSE 8003

# Use uvicorn for consistency with other services
# Cloud Run sets PORT env var, fallback to 8003
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8003}"]
