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

# ── SAMURAI + SAM2 ──────────────────────────────────────────────────────────
# Copy SAMURAI inference code and SAM2 package source (checkpoints are
# downloaded at runtime to /workspace/weights/ by ensure_weights()).
COPY samurai/ /app/samurai/
# SAM2's __init__.py imports hydra at top level and demo.py imports loguru,
# so we MUST install those deps. Previously this ran with --no-deps which
# skipped hydra-core / iopath / loguru / omegaconf, and SAMURAI tracking
# crashed on `from hydra import initialize_config_module` before it could
# even argparse. We install the deps explicitly (avoids reinstalling torch),
# then the package itself with --no-deps to skip the CUDA extension build
# that would fail without nvcc in this runtime image.
RUN pip install \
        "hydra-core>=1.3.2" \
        "iopath>=0.1.10" \
        "omegaconf>=2.3.0" \
        "loguru>=0.7.0" \
        "tqdm>=4.66.1" \
 && pip install --no-deps --no-build-isolation -e /app/samurai/sam2

# Make sam2 importable even if the editable install didn't fully register.
ENV PYTHONPATH="/app/samurai/sam2:/app/samurai:${PYTHONPATH:-}"

# SAMURAI_SCRIPT must point at the SAM2 demo.py — that's the script whose
# argparse contract (--video_path / --txt_path / --video_output_path /
# --model_path) the server actually calls. The bundled run_samurai.py is a
# separate wrapper with a different CLI and is not used by the server.
# SAM2_MODEL_PATH is set at runtime by ensure_weights() once the checkpoint
# has been downloaded to the persistent Network Volume.
ENV SAMURAI_SCRIPT=/app/samurai/scripts/demo.py

# Default workspace; RunPod mounts the Network Volume over this at runtime.
RUN mkdir -p /workspace
ENV WORKSPACE_DIR=/workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers"]
