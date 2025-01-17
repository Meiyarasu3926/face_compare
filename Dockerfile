# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for TensorFlow
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV CUDA_VISIBLE_DEVICES="-1"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=4000

# Disable CUDA completely
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=""
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false
ENV CUDA_CACHE_DISABLE=1

# Install system dependencies and cleanup
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/cuda* \
    && rm -rf /usr/local/nvidia*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with optimizations and ensure no GPU packages
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cudnn-cu11 nvidia-cuda-cupti-cu11 nvidia-cublas-cu11 nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 || true

# Copy the rest of the application
COPY . .

# Create temp directory for file uploads
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose dynamic port
EXPOSE ${PORT}

# Add initialization script
COPY <<EOF init.sh
#!/bin/sh
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=""
exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
EOF

RUN chmod +x init.sh

# Command to run the application with dynamic port
CMD ["init.sh"]
