FROM python:3.10-slim

# Prevent Python buffering (important for logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV / YOLO)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose HF port
EXPOSE 7860

# Run via module (OpenEnv compliant)
CMD ["python", "-m", "server.app"]
