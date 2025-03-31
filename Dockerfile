# Use an NVIDIA CUDA base image compatible with recent PyTorch versions
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies
# - Python 3.10 and pip
# - Git (needed by some pip installs or if cloning repos)
# - X11 libraries for GUI forwarding
# - OpenCV dependencies (including GUI components like highgui)
# - Other common libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libice6 \
    libsm6 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy necessary files
# Copy requirements first for layer caching
COPY requirements.txt .
# Copy the application code and local unik3d module
COPY unik3d ./unik3d
COPY live_slam_viewer.py .
# Copy hubconf.py if it's needed by the model loading process
COPY hubconf.py .

# Install Python dependencies
# Install PyTorch matching the CUDA version first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install xformers (try pre-built wheel first)
RUN pip install --no-cache-dir xformers
# Install remaining requirements
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# Command to run the application
CMD ["python", "live_slam_viewer.py"]