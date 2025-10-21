# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install any system dependencies you might still need (none for Wi-Fi)
RUN apt-get update && apt-get install -y \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main Wi-Fi communication file
COPY Wifi_main.py .

# Expose port 8000 (FastAPI)
EXPOSE 8000

# Run the FastAPI Wi-Fi receiver
CMD ["uvicorn", "Wifi_main:app", "--host", "0.0.0.0", "--port", "8000"]