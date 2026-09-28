from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore


def get_engine() -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL."""
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
