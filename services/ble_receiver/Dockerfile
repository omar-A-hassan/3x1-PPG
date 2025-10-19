FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Bluetooth
RUN apt-get update && apt-get install -y \
    bluez \
    bluetooth \
    libbluetooth-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Run with Bluetooth access
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
