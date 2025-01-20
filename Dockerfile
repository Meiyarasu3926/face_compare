# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Add CUDA repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    software-properties-common \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && add-apt-repository "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cuda \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb

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

# Add CUDA to PATH
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Expose port
EXPOSE 4000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
