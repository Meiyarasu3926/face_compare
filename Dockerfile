# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV CUDA_VISIBLE_DEVICES="-1"
ENV PORT=4000
ENV DEEPFACE_HOME=/app/.deepface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create directories for deepface
RUN mkdir -p /app/.deepface/weights

# Download models during build
RUN wget -q https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 -O /app/.deepface/weights/facenet512_weights.h5

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create temp directory for file uploads
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose dynamic port
EXPOSE ${PORT}

# Set memory limit for Python
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRIM_THRESHOLD_=100000

# Command to run the application with optimized settings
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2 --limit-concurrency 10 --backlog 8
