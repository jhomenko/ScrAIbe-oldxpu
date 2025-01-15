#pytorch Image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Labels

LABEL maintainer="Jacob Schmieder"
LABEL email="Jacob.Schmieder@dbfz.de"
LABEL version="0.0.0"
LABEL description="Scraibe is a tool for automatic speech recognition and speaker diarization. \
                    It is based on the Hugging Face Transformers library and the Pyannote library. \
                    It is designed to be used with the Whisper model, a lightweight model for automatic \
                    speech recognition and speaker diarization."
LABEL url="https://github.com/JSchmie/ScrAIbe"

# Install dependencies
WORKDIR /app
#Enviorment dependencies
ENV AUTOT_CACHE=/data/models/
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/"

#Copy all necessary files 
COPY README.md /app/src/README.md
COPY scraibe /app/src/scraibe
COPY pyproject.toml /app/src/pyproject.toml
COPY LICENSE /app/src/LICENSE

RUN mkdir -p /data

#Installing all necessary Dependencies and Running the Application with a personalised Hugging-Face-Token
RUN apt update -y && apt upgrade -y && \
    apt install -y libsm6 libxrender1 libfontconfig1 git ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN conda update --all -y && conda install -y -c conda-forge libsndfile && \ 
    conda clean --all -y

RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir ./src

# Run the application
ENTRYPOINT ["python3", "-m",  "scraibe.cli"]  