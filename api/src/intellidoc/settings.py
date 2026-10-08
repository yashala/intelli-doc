from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    database_url: str = "postgresql://intellidoc:intellidoc@localhost:5433/intellidoc"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("../artifacts")
    cors_origins: list[str] = ["http://localhost:5173"]
    max_upload_mb: int = 25

    # "offline" uses a deterministic extractive model; no network or keys needed.
    llm_mode: Literal["offline", "gemini"] = "offline"
    google_api_key: SecretStr | None = None
    answer_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-pro"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    default_chunker: str = "structure"
    default_mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"
    # Below this top cosine similarity the workflow treats retrieval as failed.
    min_similarity: float = 0.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
