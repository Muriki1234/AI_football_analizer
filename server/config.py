"""
Runtime configuration loaded from env vars. RunPod injects these via the Pod
environment settings; local dev reads from server/.env.

All paths default to /workspace (RunPod Network Volume) but fall back to a
dev-friendly layout under the repo when the volume isn't mounted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path("/workspace") if Path("/workspace").exists() else REPO_ROOT / "server"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / "server" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Auth ──────────────────────────────────────────────────────────────────
    API_KEY: str = Field(
        default="",
        description="Bearer token clients send in X-API-Key. Empty = auth disabled (dev only).",
    )

    # ── Network ───────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated list of allowed origins. Set to https://your.site in prod.",
    )

    # ── Storage layout ────────────────────────────────────────────────────────
    WORKSPACE_DIR: Path = Field(default_factory=lambda: DEFAULT_WORKSPACE)
    OUTPUT_DIR_NAME: str = "outputs"
    UPLOAD_DIR_NAME: str = "uploads"
    WEIGHTS_DIR_NAME: str = "weights"
    DB_FILENAME: str = "sessions.db"

    # ── Session lifecycle ─────────────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 48
    CLEANUP_INTERVAL_MINUTES: int = 30

    # ── Worker pool ───────────────────────────────────────────────────────────
    GPU_WORKERS: int = 1
    IO_WORKERS: int = 4
    TASK_TIMEOUT_SECONDS: int = 60 * 60  # 1h cap per pipeline task

    # ── Model weights (HuggingFace Hub) ───────────────────────────────────────
    HF_REPO_ID: str = Field(
        default="Muriki123/ai-football-assistant-weights",
        description="HF Hub repo that hosts soccana_best.pt and soccana_kpts_best.pt.",
    )
    HF_YOLO_FILENAME: str = "soccana_best.pt"
    HF_KEYPOINT_FILENAME: str = "soccana_kpts_best.pt"
    HF_TOKEN: str | None = None

    # ── Optional SAMURAI tracking ─────────────────────────────────────────────
    SAMURAI_SCRIPT: str | None = Field(
        default=None,
        description="Path to SAMURAI run script. If unset, /track returns 501.",
    )

    # ── DashScope (Qwen-VL for AI summary) ────────────────────────────────────
    DASHSCOPE_API_KEY: str | None = None
    DASHSCOPE_MODEL: str = "qwen-vl-max-latest"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Cloud Services ────────────────────────────────────────────────────────
    SUPABASE_URL: str | None = None
    SUPABASE_SERVICE_KEY: str | None = None
    
    R2_ACCOUNT_ID: str | None = None
    R2_BUCKET_NAME: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None
    R2_ENDPOINT_URL: str | None = None

    # ── Derived paths ─────────────────────────────────────────────────────────
    @property
    def output_root(self) -> Path:
        return self.WORKSPACE_DIR / self.OUTPUT_DIR_NAME

    @property
    def upload_root(self) -> Path:
        return self.WORKSPACE_DIR / self.UPLOAD_DIR_NAME

    @property
    def weights_root(self) -> Path:
        return self.WORKSPACE_DIR / self.WEIGHTS_DIR_NAME

    @property
    def db_path(self) -> Path:
        return self.WORKSPACE_DIR / self.DB_FILENAME

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()


def ensure_dirs() -> None:
    """Create all runtime directories on startup."""
    for p in (
        settings.WORKSPACE_DIR,
        settings.output_root,
        settings.upload_root,
        settings.weights_root,
    ):
        p.mkdir(parents=True, exist_ok=True)
