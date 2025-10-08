# Dockerfile for NeuTTS Air
# Build args:
#   PYTHON_VERSION (default 3.11)
# Usage examples:
#  docker build -t neutts-air:latest .
#  docker run --rm -it -v $(pwd):/workspace -v /path/to/models:/models neutts-air:latest /bin/bash

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  LANG=C.UTF-8 \
  DEBIAN_FRONTEND=noninteractive

# Install system dependencies (espeak-ng and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  wget \
  curl \
  ca-certificates \
  pkg-config \
  libsndfile1 \
  libsndfile1-dev \
  libssl-dev \
  libffi-dev \
  ffmpeg \
  cmake \
  python3-dev \
  espeak-ng \
  libespeak-ng1 \
  libespeak-ng-dev \
  # portaudio19-dev \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG UNAME=neutts
ARG UID=1000
RUN groupadd -g ${UID} ${UNAME} || true \
  && useradd -m -u ${UID} -g ${UID} -s /bin/bash ${UNAME}

WORKDIR /workspace

# Copy repo files into container (assumes you build from repo root)
# This will allow Docker cache to be used: copy only requirements first if present
COPY requirements.txt requirements.txt
# Some projects pin a fairly large set of packages; install requirements first
RUN pip install --upgrade pip setuptools wheel \
  && pip install -r requirements.txt \
  && pip install "llama-cpp-python" \
  && pip install "onnxruntime" 
  # && pip install "pyaudio"

# Copy the rest of the project into the image
# (This keeps installed pip packages cached from earlier step)
COPY . /workspace

# Ensure correct ownership
RUN chown -R ${UNAME}:${UNAME} /workspace

# Switch to non-root user
USER ${UNAME}
ENV PATH="/home/${UNAME}/.local/bin:${PATH}" \
  PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so \
  PHONEMIZER_ESPEAK_PATH=/usr/lib/x86_64-linux-gnu \
  HF_HOME=/home/${UNAME}/.cache/huggingface \
  NUMBA_CACHE_DIR=/tmp/numba_cache

RUN mkdir -p /tmp/numba_cache && chmod -R 777 /tmp/numba_cache

# Default workdir for running examples or scripts
WORKDIR /workspace

# Expose nothing by default — NeuTTS Air is a library / CLI scripts
# Provide a small entrypoint helper script that users can override
# If you want the container to run an example by default, change ENTRYPOINT/CMD
# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["python", "-m", "examples.wyoming_server", \
  "--uri", "tcp://0.0.0.0:10600", \
  "--voice", "name=joi,ref_codes=samples/joi.pt,ref_text=samples/joi.txt", \
  "--backbone", "neuphonic/neutts-air-q4-gguf", \
  "--codec", "neuphonic/neucodec-onnx-decoder" ]
# CMD ["-c", "echo 'Container ready. Mount your models and run python -m examples.basic_example ...' && /bin/bash"]
