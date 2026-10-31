"""Application settings loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_local_path: str = "./qdrant_data"
    collection_name: str = "videorag"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_provider: Literal["auto", "anthropic", "openai", "groq"] = "auto"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
   # groq_model: str = "llama-3.3-70b-versatile"
   # Change line 32 to:
    groq_model: str = "llama-3.1-8b-instant"

    # Ingestion / retrieval
    chunk_seconds: int = Field(default=60, ge=15, le=600)
    chunk_overlap_seconds: int = Field(default=10, ge=0, le=120)
    max_chunk_chars: int = Field(default=1500, ge=200, le=8000)
    max_playlist_videos: int = Field(default=50, ge=1, le=500)
    top_k: int = Field(default=6, ge=1, le=20)

    cors_origins: str = "*"

    @field_validator(
        "qdrant_url", "qdrant_api_key", "anthropic_api_key", "openai_api_key",
        "groq_api_key", "embedding_model", "collection_name", mode="before",
    )
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _lower_provider(cls, v):
        return v.strip().lower() if isinstance(v, str) and v.strip() else "auto"

    @property
    def cors_origin_list(self) -> list[str]:
        items = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return items or ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
