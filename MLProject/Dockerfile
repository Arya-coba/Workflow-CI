FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file
COPY . .

# Expose port untuk serving
EXPOSE 5001

# Default command — jalankan inference API
CMD ["python", "inference.py"]
