FROM python:3.10-slim

# =========================
# ENV
# =========================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# =========================
# SYSTEM DEPENDENCIES
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# =========================
# COPY FILES
# =========================
COPY . .

# =========================
# PYTHON DEPENDENCIES
# =========================
RUN pip install --upgrade pip

# ⚡ Install torch separately (more stable on HF)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install rest
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# PORT
# =========================
EXPOSE 7860

# =========================
# RUN SERVER
# =========================
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
