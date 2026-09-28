from __future__ import annotations
from pathlib import path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.ingest import BASE_DIR

DIR_PATH = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )