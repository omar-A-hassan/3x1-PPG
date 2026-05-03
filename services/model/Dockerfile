FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including libraries needed by PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .
COPY best_model.pt .

# Copy src and ts2vec (must be copied into services/model/ folder first)
COPY src /app/src
COPY ts2vec /app/ts2vec

EXPOSE 8002

# Set PyTorch to single-threaded mode to avoid issues in container
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Use uvicorn for consistency
# Cloud Run sets PORT env var, fallback to 8002 for local/docker-compose
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8002}"]
