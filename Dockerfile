# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for TensorFlow
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV CUDA_VISIBLE_DEVICES="-1"
ENV PORT=4000

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

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

# Command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
