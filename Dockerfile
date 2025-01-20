# Use NVIDIA CUDA base image instead of plain Python
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python 3.9 and required system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
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

# Expose port
EXPOSE 4000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
