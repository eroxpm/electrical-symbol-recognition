# Electrical Symbol Recognition with SAM3
# Docker image with CUDA support for GPU inference

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA support FIRST
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create data directories
RUN mkdir -p data/input/raw data/output/visualizations data/output/masks models/huggingface

# Set HuggingFace cache to local models directory
ENV HF_HOME=/app/models/huggingface
ENV TRANSFORMERS_CACHE=/app/models/huggingface

# Default command
# Default command
ENTRYPOINT ["python3", "scripts/mosaic_inference.py"]
CMD ["--help"]
