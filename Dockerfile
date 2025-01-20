# Use NVIDIA CUDA development image which includes CUDA toolkit
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install basic Python packages
RUN python3 -m pip install --upgrade setuptools pip wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create temp directory for file uploads
RUN mkdir -p /tmp && chmod 777 /tmp

# CUDA paths are already set in the base image
# but we can verify them with:
RUN nvcc --version

# Expose port
EXPOSE 4000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
