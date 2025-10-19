FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including libraries needed by PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY services/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/model/main.py .

# Copy project files (src, ts2vec, best_model.pt)
COPY src /app/src
COPY ts2vec /app/ts2vec
COPY services/model/best_model.pt /app/best_model.pt

EXPOSE 8002

# Set PyTorch to single-threaded mode to avoid issues in container
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["python", "-u", "main.py"]
