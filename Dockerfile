FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_XET=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      python3 \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.batch.txt /tmp/requirements.batch.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 && \
    python3 -m pip install -r /tmp/requirements.batch.txt

COPY . /app

# Batch tasks can override these at submit time.
ENV HF_HOME=/tmp/hf_home \
    HF_HUB_CACHE=/tmp/hf_home/hub \
    HF_DATASETS_CACHE=/tmp/hf_home/datasets \
    TRANSFORMERS_CACHE=/tmp/hf_home/models

RUN mkdir -p /tmp/hf_home/hub /tmp/hf_home/datasets /tmp/hf_home/models

CMD ["python3", "power_samp_math_approx.py", "--help"]
