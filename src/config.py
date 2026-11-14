"""Configuration loader combining .env secrets and config/settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class EnvSettings(BaseSettings):
    """Environment variables loaded from `.env` (never committed to git)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    youtube_api_key: str = Field(default="", alias="YOUTUBE_API_KEY")
    youtube_channel_handle: str = Field(default="@NYANTECH", alias="YOUTUBE_CHANNEL_HANDLE")
    youtube_channel_id: str = Field(default="", alias="YOUTUBE_CHANNEL_ID")

    llm_provider: str = Field(default="none", alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")


class ChannelConfig(BaseModel):
    handle: str = "@NYANTECH"
    channel_id: str = ""
    timezone: str = "Asia/Jakarta"


class DateFilterConfig(BaseModel):
    start_date: str = "2026-01-01"
    end_date: str = "2026-12-31"
    use_local_timezone: bool = True


class ScrapingConfig(BaseModel):
    include_replies: bool = True
    max_comments_per_video: int | None = None
    transcript_languages: list[str] = Field(default_factory=lambda: ["id", "en"])
    request_delay_seconds: float = 0.15
    checkpoint_every_n_videos: int = 5


class SmartphoneFilterConfig(BaseModel):
    min_confidence_score: int = 2
    exclude_shorts: bool = False
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)


class PriceSegmentConfig(BaseModel):
    label: str
    min_idr: int
    max_idr: int | None = None


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    outputs_dir: str = "data/outputs"

    def resolve_raw_dir(self, root: Path = PROJECT_ROOT) -> Path:
        path = root / self.raw_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_processed_dir(self, root: Path = PROJECT_ROOT) -> Path:
        path = root / self.processed_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_outputs_dir(self, root: Path = PROJECT_ROOT) -> Path:
        path = root / self.outputs_dir
        path.mkdir(parents=True, exist_ok=True)
        return path


class AppConfig(BaseModel):
    channel: ChannelConfig = Field(default_factory=ChannelConfig)
    date_filter: DateFilterConfig = Field(default_factory=DateFilterConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    smartphone_filter: SmartphoneFilterConfig = Field(default_factory=SmartphoneFilterConfig)
    brands: dict[str, list[str]] = Field(default_factory=dict)
    price_segments: list[PriceSegmentConfig] = Field(default_factory=list)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: Path | None = None) -> tuple[AppConfig, EnvSettings]:
    """Load YAML pipeline config and `.env` credentials."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    env_settings = EnvSettings()

    target_path = config_path or DEFAULT_CONFIG_PATH
    raw_yaml: dict[str, Any] = {}
    if target_path.exists():
        with target_path.open("r", encoding="utf-8") as fh:
            raw_yaml = yaml.safe_load(fh) or {}

    app_config = AppConfig.model_validate(raw_yaml)

    # Allow .env channel overrides if provided
    if env_settings.youtube_channel_handle:
        app_config.channel.handle = env_settings.youtube_channel_handle
    if env_settings.youtube_channel_id:
        app_config.channel.channel_id = env_settings.youtube_channel_id

    return app_config, env_settings
