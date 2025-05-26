ARG HF_TOKEN

# Builder stage
FROM intel/intel-extension-for-pytorch:2.7.10-xpu as builder
# Labels
# Labels will be in the final stage

# Install dependencies
WORKDIR /app
#Enviorment dependencies
ENV SYCL_DEVICE_FILTER=level_zero:gpu
ENV IPEX_XPU_ONEDNN_LAYOUT=1
ENV SCRAIBE_TORCH_DEVICE=xpu
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV AUTOT_CACHE=/app/models
ENV PYANNOTE_CACHE=/app/models/pyannote

# Installing system dependencies, including ffmpeg for audio processing
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common && \
    apt-key adv --refresh-keys --keyserver hkp://keyserver.ubuntu.com:80 && \
    apt update -y && apt upgrade -y && \
    apt install -y libsm6 libxrender1 libfontconfig1 ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up Intel APT repository and install necessary runtimes
RUN apt-get update && \
    apt install -y gpg-agent wget && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-runtime-tbb intel-oneapi-mkl && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all necessary files
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY scraibe /app/scraibe

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt


# Download the "medium" Whisper model from Hugging Face
ARG HF_TOKEN
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='openai/whisper-medium', filename='pytorch_model.bin', cache_dir='/app/models', token='${HF_TOKEN}')"
# Final stage
# Use a smaller base image for the final runtime
FROM ubuntu:jammy

# Labels (move labels to the final stage)
LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.1.1.dev"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"

# Copy necessary files from the builder stage
WORKDIR /app

# Copy installed Python packages from the builder
COPY --from=builder /usr/local/lib/python*/dist-packages /usr/local/lib/python/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code and other necessary files
COPY --from=builder /app/scraibe /app/scraibe
COPY --from=builder /app/README.md /app/README.md

# Copy models if they are downloaded during the build (adjust path if necessary)
COPY --from=builder /app/models /app/models

# Install runtime dependencies in the final stage
RUN apt-get update && \
    apt-get install -y libsm6 libxrender1 libfontconfig1 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENTRYPOINT ["python3", "-m", "scraibe.cli"]