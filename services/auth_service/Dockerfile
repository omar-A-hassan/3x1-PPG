FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy service code and env file
COPY auth_service.py .
COPY .env .

EXPOSE 8004

# Run the auth service
CMD ["sh", "-c", "uvicorn auth_service:app --host 0.0.0.0 --port ${PORT:-8004}"]
