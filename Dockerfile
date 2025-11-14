# Base: CUDA 12.6 runtime with Ubuntu 22.04
# FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04  
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


# Basic setup
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libnuma-dev git pkg-config \
    cuda-command-line-tools-12-4 || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
    wget curl ffmpeg sox libsndfile1 \
    libhdf5-dev libfreetype6-dev libfontconfig1-dev libffi-dev \
    cython3 && \
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
    nvidia-ml-py \
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

# install Transformer Engine (stable) and ModelOpt with torch extras
ENV NVTE_FRAMEWORK=pytorch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      "git+https://github.com/NVIDIA/TransformerEngine.git@stable" \
      "nvidia-modelopt[all]" \
      --prefer-binary


# NVIDIA apex (if NeMo / Megatron needs it)
RUN pip install --no-cache-dir git+https://github.com/NVIDIA/apex.git

COPY . /workspace
RUN pip install -e .


# Set environment vars
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace


# Default command
CMD ["/bin/bash"]
