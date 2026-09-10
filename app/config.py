"""
Application configuration via environment variables.
"""

import os
from dataclasses import dataclass


def _chunk_integer_from_env(name: str, default: int) -> int | None:
    """Defer malformed chunk values to startup validation instead of failing import."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return None


@dataclass
class Settings:
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "demo")  # "openai", "ollama", or "demo"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector Store
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./vectorstore")

    # Chunking
    CHUNK_SIZE: int | None = _chunk_integer_from_env("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int | None = _chunk_integer_from_env("CHUNK_OVERLAP", 100)

    # Upload
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        """Validate the active pipeline before constructing any backend clients."""
        if not isinstance(self.LLM_PROVIDER, str) or self.LLM_PROVIDER not in {"demo", "openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be demo, openai or ollama.")
        if not isinstance(self.LLM_MODEL, str):
            raise ValueError("LLM_MODEL must be a string.")
        if (
            not isinstance(self.CHUNK_SIZE, int)
            or isinstance(self.CHUNK_SIZE, bool)
            or not isinstance(self.CHUNK_OVERLAP, int)
            or isinstance(self.CHUNK_OVERLAP, bool)
            or self.CHUNK_SIZE <= 0
            or not 0 <= self.CHUNK_OVERLAP < self.CHUNK_SIZE
        ):
            raise ValueError("Chunk settings require integer size > 0 and 0 <= overlap < size.")
        if self.LLM_PROVIDER != "demo":
            for value in (self.LLM_MODEL, self.EMBEDDING_MODEL, self.VECTORSTORE_PATH):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Real mode requires model, embedding and storage settings.")
        if self.LLM_PROVIDER == "openai" and not isinstance(self.OPENAI_API_KEY, str):
            raise ValueError("OPENAI_API_KEY must be a string.")
        if self.LLM_PROVIDER == "ollama" and (
            not isinstance(self.OLLAMA_URL, str)
            or not self.OLLAMA_URL.startswith(("http://", "https://"))
        ):
            raise ValueError("OLLAMA_URL must be an HTTP or HTTPS URL.")


settings = Settings()
