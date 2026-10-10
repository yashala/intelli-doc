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
    llm_mode: Literal["offline", "gemini", "openai"] = "offline"
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    answer_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-5-mini"

    def api_key(self, provider: str) -> str:
        key = self.google_api_key if provider == "gemini" else self.openai_api_key
        if key is None:
            name = "GOOGLE_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            raise RuntimeError(f"{name} is required for provider {provider}")
        return key.get_secret_value()

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
