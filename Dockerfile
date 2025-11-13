# Base: CUDA 12.6 runtime with Ubuntu 22.04
# FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04  
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# nvidia/cuda:12.4.1-cudnn8-runtime-ubuntu22.04


# Basic setup
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
    git wget curl ffmpeg sox libsndfile1 \
    libhdf5-dev libfreetype6-dev libfontconfig1-dev libffi-dev \
    build-essential cython3 && \
    rm -rf /var/lib/apt/lists/*


# Copy project
WORKDIR /workspace
# COPY . /workspace


# Install project dependencies
RUN pip install --upgrade pip
# RUN python3.10 -m pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    "numpy<2.0" \
    h5py \
    pillow \
    opencv-python \
    focal-frequency-loss \
    fonttools \
    freetype-py \
    iopath \
    imutils \
    pynvml \
    imageio \
    termcolor \
    attrs \
    PyYAML \
    hydra-core \
    pytorch-lightning \
    einops \
    accelerate \
    pybind11
RUN pip install --no-build-isolation nemo_toolkit[all]

COPY . /workspace
RUN pip install -e .


# Set environment vars
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace


# Default command
CMD ["/bin/bash"]
