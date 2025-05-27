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

# Add Intel repositories and install graphics drivers, compute runtime, and Level Zero Loader
# Install dependencies and configure the environment
RUN set -eux && \
    apt-get update && \
    #
    # Update and install basic dependencies and prerequisites for Intel repo
    apt-get install -y --no-install-recommends \
      curl git sudo libunwind8-dev vim less gnupg gpg-agent software-properties-common wget && \
    \
    # Add Intel GPU repository and key
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
        tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
    \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-opencl-icd \
        intel-level-zero-gpu \
        level-zero && \
    \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment early
RUN python3 -m venv /app/venv &&     . /app/venv/bin/activate &&     python -m pip --no-cache-dir install --upgrade pip setuptools psutil

# Set PATH to use the virtual environment's bin directory by default
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies explicitly within the virtual environment
RUN . /app/venv/bin/activate && \
    pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu

RUN . /app/venv/bin/activate && \
    pip install --no-cache-dir \
    intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

RUN . /app/venv/bin/activate && \
    pip install --no-cache-dir intel-openmp

# Install other dependencies from requirements.txt, grouping them

# Copy application files
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE
COPY scraibe /app/scraibe
COPY pyproject.toml poetry.lock* /app/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.1.1.dev"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"
RUN . /app/venv/bin/activate && \
    pip install .


# Download default model from huggingface using the HF_TOKEN environment variable (will be downloaded inside the container)
ARG HF_TOKEN
RUN . /app/venv/bin/activate && \
    python3 -c "from huggingface_hub import hf_hub_download; import os; hf_hub_download(repo_id='openai/whisper-medium', filename='pytorch_model.bin', cache_dir='/app/models', token=os.environ.get('HF_TOKEN'))"

# Entrypoint script will activate the venv and source setvars.sh
# The custom docker-entrypoint.sh will be modified to handle this
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# Modify the docker-entrypoint.sh script
RUN sed -i '/source \/app\/venv\/lib\/python3.\/site-packages\/torch\/lib\/..\/..\/..\/..\/oneccl_bindings_for_pytorch\/env\/setvars.sh/d' /usr/local/bin/docker-entrypoint.sh && \
    sed -i '/# Ensure the oneAPI environment is sourced/i # Ensure the oneAPI environment is sourced\nsource \/opt\/intel\/oneapi\/setvars.sh || true\n\n# Activate the virtual environment\nsource \/app\/venv\/bin\/activate' /usr/local/bin/docker-entrypoint.sh


# Declare /opt/intel/oneapi as a volume to be mounted from the host
VOLUME /opt/intel/oneapi
