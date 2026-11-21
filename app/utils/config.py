"""
app/utils/config.py
-------------------
Single source of truth for all configuration values.

Design decisions:
- pydantic-settings BaseSettings auto-reads from environment variables and
  from a .env file, giving us 12-factor compliance for free.
- All values are typed and validated at startup — fail-fast beats silent
  misconfiguration at query time.
- A module-level `settings` singleton is imported everywhere else; there
  is no "global config dict" floating around.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "AgriRAG Assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_api_base_url: str = "http://localhost:8000"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "agri_knowledge"
    qdrant_api_key: str | None = None

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Language model ────────────────────────────────────────────────────────
    agriparam_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    agriparam_model_path: str | None = None  # Local path if model is cached elsewhere
    agriparam_device: str = "cpu"
    agriparam_max_new_tokens: int = 192
    agriparam_temperature: float = 0.3
    agriparam_top_p: float = 0.9

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_score_threshold: float = 0.35

    # ── PDF ingestion ─────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    data_raw_dir: Path = Path("./data/raw")
    data_processed_dir: Path = Path("./data/processed")

    # ── HuggingFace ───────────────────────────────────────────────────────────
    hf_home: str = "./data/.hf_cache"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def json_logs(self) -> bool:
        return self.is_production

    @field_validator("agriparam_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"agriparam_device must be one of {allowed}, got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# Module-level convenience alias so callers can do:
#   from app.utils.config import settings
settings: Settings = get_settings()
