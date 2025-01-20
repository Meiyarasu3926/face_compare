# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install cuda-toolkit

RUN python3 -m pip install --upgrade setuptools pip wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to disable GPU
ENV CUDA_VISIBLE_DEVICES="-1"

# Copy the rest of the application
COPY . .

# Create temp directory for file uploads
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose port
EXPOSE 4000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
