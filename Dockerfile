# Dockerfile for NeuTTS Air

ARG PYTHON_VERSION=3.13
# ==================================================================
# BUILDER STAGE
# Installs build dependencies, downloads/builds python packages
# ==================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    libsndfile1-dev \
    libssl-dev \
    libffi-dev \
    ffmpeg \
    cmake \
    python3-dev \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the workspace and copy requirements
WORKDIR /app
COPY requirements.txt .

# Install uv and use it in the SAME RUN layer
# uv installs to $HOME/.local/bin (which is /root/.local/bin for root user)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --prefix="/install" \
    -r requirements.txt \
    "llama-cpp-python" \
    "onnxruntime" \
    "nltk"


# ==================================================================
# FINAL STAGE
# A clean, small image with only runtime dependencies
# ==================================================================
FROM python:${PYTHON_VERSION}-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install ONLY runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libespeak-ng-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG UNAME=neutts
ARG UID=1000
RUN groupadd -g ${UID} ${UNAME} || true \
    && useradd -m -u ${UID} -g ${UID} -s /bin/bash ${UNAME}

# Copy installed python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code (will respect .dockerignore)
WORKDIR /workspace
COPY . /workspace
RUN chown -R ${UNAME}:${UNAME} /workspace

# Switch to non-root user
USER ${UNAME}
ENV PATH="/home/${UNAME}/.local/bin:${PATH}" \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so \
    PHONEMIZER_ESPEAK_PATH=/usr/lib/x86_64-linux-gnu \
    HF_HOME=/home/${UNAME}/.cache/huggingface \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN mkdir -p /tmp/numba_cache && chmod -R 777 /tmp/numba_cache

# Define mount points for models and data (good practice)
VOLUME ["/models"]

# Default workdir
WORKDIR /workspace

# UPDATED CMD: Points to the /models and /samples volumes
CMD ["python", "-m", "examples.wyoming_server", \
    "--uri", "tcp://0.0.0.0:10200", \
    "--debug", \
    "--voice", "name=joi,ref_codes=samples/joi.pt,ref_text=samples/joi.txt", \
    "--backbone", "models/neutts-air-q4-gguf.gguf", \
    "--codec", "models/neucodec-onnx-decoder.onnx" ]