FROM python:3.10-slim

# =========================
# ENV
# =========================
ENV PYTHONUNBUFFERED=1

# =========================
# WORKDIR
# =========================
WORKDIR /app

# =========================
# SYSTEM DEPENDENCIES (IMPORTANT FOR YOLO + CV2)
# =========================
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# COPY PROJECT
# =========================
COPY . .

# =========================
# PIP UPGRADE
# =========================
RUN pip install --upgrade pip

# =========================
# REQUIREMENTS
# =========================
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# EXPOSE HF PORT
# =========================
EXPOSE 7860

# =========================
# START SERVER (IMPORTANT FIX)
# =========================
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
