# Copyright (c) 2024 Jacob Schmieder
# SPDX-License-Identifier: Apache 2.0

# NOTE: To build this you will need a docker version >= 19.03 and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies
WORKDIR /app

# Environment dependencies
ENV SYCL_DEVICE_FILTER=level_zero:gpu
ENV IPEX_XPU_ONEDNN_LAYOUT=1
ENV SCRAIBE_TORCH_DEVICE=xpu
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV AUTOT_CACHE=/app/models
ENV PYANNOTE_CACHE="/app/models/pyannote"
ARG HF_TOKEN

# Installing system dependencies, including ffmpeg for audio processing, and python venv
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common && \
    apt-key adv --refresh-keys --keyserver hkp://keyserver.ubuntu.com:80 && \
    apt update -y && apt upgrade -y && \
    apt install -y libsm6 libxrender1 libfontconfig1 ffmpeg python3 python3-pip python3-dev python3-venv && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create and activate Python virtual environment early
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    python -m pip --no-cache-dir install --upgrade pip setuptools psutil

# Set PATH to use the virtual environment's bin directory by default
ENV PATH="/app/venv/bin:$PATH"

# Copy application files and requirements
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY scraibe /app/scraibe
COPY pyproject.toml poetry.lock* /app/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Labels (moved to the final stage)

# Labels (move labels to the final stage)
LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.1.1.dev"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"
ARG IPEX_VERSION=2.7.0
ARG TORCHCCL_VERSION=2.7.0
ARG PYTORCH_VERSION=2.7.0
ARG TORCHAUDIO_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0

# Install PyTorch and Intel Extension for PyTorch within the virtual environment
RUN . /app/venv/bin/activate && \
    pip install --no-cache-dir torch==2.7.0+cpu torchvision==0.22.0+cpu torchaudio==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir intel_extension_for_pytorch==2.7.0 oneccl_bind_pt==2.7.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ && \
    pip install intel-openmp && \
    pip install --no-cache-dir -r requirements.txt

# Install ScrAIbe itself within the virtual environment
RUN . /app/venv/bin/activate && \
    pip install .


# Download default model from huggingface using the HF_TOKEN environment variable (will be downloaded inside the container)
ARG HF_TOKEN
RUN . /app/venv/bin/activate && \
    python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='openai/whisper-medium', filename='pytorch_model.bin', cache_dir='/app/models', token=os.environ.get('HF_TOKEN'))"

# Entrypoint script will activate the venv and source setvars.sh
# The custom docker-entrypoint.sh will be modified to handle this
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
