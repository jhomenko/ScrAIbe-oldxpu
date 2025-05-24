# 🧱 Base image with Intel Arc GPU support and IPEX preinstalled
FROM intel/intel-extension-for-pytorch:2.7.10-xpu

# Labels
LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.1.1.dev"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"

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
#Copy all necessary files 

#Installing all necessary dependencies and running the application with a personalised Hugging-Face-Token
RUN apt update -y && apt upgrade -y && \
    apt install -y libsm6 libxrender1 libfontconfig1 ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy all necessary files
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md
COPY scraibe /app/scraibe

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Run the application

ENTRYPOINT ["python3", "-m", "scraibe.cli"]