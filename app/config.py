"""
Application configuration via environment variables.
"""

import os
from dataclasses import dataclass


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
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Upload
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "20"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
