"""
Downloads YOLO + keypoint weights from HuggingFace Hub on startup. Weights
land in settings.weights_root and the env vars the pipeline already reads
(YOLO_MODEL_PATH, KEYPOINT_MODEL_PATH) are pointed at them.

Idempotent: if the files are already on disk (e.g. the RunPod Network Volume
was pre-warmed) the download is a no-op revision check.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from ..config import settings

log = logging.getLogger(__name__)


class WeightsError(RuntimeError):
    pass


def _download_one(repo_id: str, filename: str, dest_dir: Path, token: str | None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        log.info("weights: %s already present at %s (skipping download)", filename, target)
        return target

    log.info("weights: downloading %s/%s -> %s", repo_id, filename, target)
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest_dir),
            token=token,
        )
    except HfHubHTTPError as e:
        raise WeightsError(
            f"HF download failed for {repo_id}/{filename}: {e}. "
            f"Check HF_REPO_ID / HF_TOKEN / network."
        ) from e

    local_path = Path(local)
    if not local_path.exists() or local_path.stat().st_size == 0:
        raise WeightsError(f"Weights file empty after download: {local_path}")
    return local_path


_SAM2_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
)


def _download_url(url: str, dest_dir: Path, filename: str) -> Path:
    """Download a file from a plain URL (no auth). Skips if already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        log.info("weights: %s already present (skipping)", filename)
        return target
    log.info("weights: downloading %s -> %s", url, target)
    tmp = target.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(target)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise WeightsError(f"Failed to download {url}: {e}") from e
    return target


def ensure_weights() -> dict[str, Path]:
    """Pull all model weights and expose their paths via env vars."""
    yolo = _download_one(
        settings.HF_REPO_ID,
        settings.HF_YOLO_FILENAME,
        settings.weights_root,
        settings.HF_TOKEN,
    )
    kpt = _download_one(
        settings.HF_REPO_ID,
        settings.HF_KEYPOINT_FILENAME,
        settings.weights_root,
        settings.HF_TOKEN,
    )

    # SAM2 checkpoint — public Meta URL, no token required.
    sam2 = _download_url(_SAM2_URL, settings.weights_root, "sam2.1_hiera_base_plus.pt")

    # The pipeline reads these env vars to find weights.
    os.environ["YOLO_MODEL_PATH"] = str(yolo)
    os.environ["KEYPOINT_MODEL_PATH"] = str(kpt)
    os.environ["SAM2_MODEL_PATH"] = str(sam2)
    log.info("weights: YOLO_MODEL_PATH=%s", yolo)
    log.info("weights: KEYPOINT_MODEL_PATH=%s", kpt)
    log.info("weights: SAM2_MODEL_PATH=%s", sam2)
    return {"yolo": yolo, "keypoint": kpt, "sam2": sam2}
