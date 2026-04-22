# syntax=docker/dockerfile:1.7
# ── AI Football Assistant (RunPod GPU image) ────────────────────────────────
# Base: CUDA 12.1 runtime (matches torch 2.5 + cu121 wheels).
# Build:   docker build -t ghcr.io/<user>/ai-football-assistant:<tag> .
# Run:     docker run --gpus all -p 8000:8000 --env-file server/.env \
#              -v /workspace:/workspace ghcr.io/<user>/ai-football-assistant:<tag>
#
# The server expects /workspace to be a persistent volume. On RunPod that's
# the Network Volume; locally any bind-mounted directory works.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        libgl1 libglib2.0-0 \
        git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Install deps in two layers so requirements changes don't re-pull torch.
COPY server/requirements.txt /app/server/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 \
 && python -m pip install -r /app/server/requirements.txt

COPY server/ /app/server/

# Default workspace; RunPod mounts the Network Volume over this at runtime.
RUN mkdir -p /workspace
ENV WORKSPACE_DIR=/workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers"]
