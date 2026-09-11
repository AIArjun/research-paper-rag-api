"""
Application configuration via environment variables.

Malformed numeric values never crash module import. They are deferred to
validate(), which fails closed with the safe category invalid_configuration.
Secrets (the access token and provider key) are read here and never logged.
"""

import os
from dataclasses import dataclass


# Fixed request bounds. They are protocol limits rather than tuning knobs.
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 5
MAX_FILENAME_CHARS = 255
MULTIPART_ALLOWANCE_BYTES = 16 * 1024  # boundary, part headers and the filename
# A 2000-character question escaped as \uXXXX pairs (12 bytes per non-BMP character)
# is about 24 KiB, so 32 KiB keeps every documented-legal question under the cap.
MAX_JSON_BODY_BYTES = 32 * 1024
RETRY_AFTER_SECONDS = 5
MIN_ACCESS_TOKEN_CHARS = 32
MAX_ACCESS_TOKEN_CHARS = 512


def _integer_from_env(name: str, default: int) -> int | None:
    """Return the default when unset; None marks a malformed value for validate()."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _chunk_integer_from_env(name: str, default: int) -> int | None:
    """Defer malformed chunk values to startup validation instead of failing import."""
    return _integer_from_env(name, default)


def _is_positive_integer(value: object, maximum: int | None = None) -> bool:
    if type(value) is not int or value <= 0:
        return False
    return maximum is None or value <= maximum


def _is_non_negative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class ResourceLimits:
    """Validated byte/count ceilings used by middleware and the engine."""

    max_file_bytes: int
    max_upload_request_bytes: int
    max_json_body_bytes: int
    max_pdf_pages: int
    max_chunks_per_paper: int
    max_papers: int
    max_total_chunks: int
    max_question_chars: int
    max_top_k: int
    max_context_chars: int
    max_concurrent_queries: int

    def as_dict(self) -> dict:
        return {
            "max_file_bytes": self.max_file_bytes,
            "max_upload_request_bytes": self.max_upload_request_bytes,
            "max_json_body_bytes": self.max_json_body_bytes,
            "max_pdf_pages": self.max_pdf_pages,
            "max_chunks_per_paper": self.max_chunks_per_paper,
            "max_papers": self.max_papers,
            "max_total_chunks": self.max_total_chunks,
            "max_question_chars": self.max_question_chars,
            "max_top_k": self.max_top_k,
            "max_context_chars": self.max_context_chars,
            "max_concurrent_queries": self.max_concurrent_queries,
        }


@dataclass(frozen=True)
class BudgetAllowances:
    """Explicit positive ceilings for attempted model calls and charged tokens."""

    calls_per_day: int
    calls_total: int
    tokens_per_day: int
    tokens_total: int


@dataclass
class Settings:
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "demo")  # "openai", "ollama", or "demo"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Bounded provider use (real providers only). Retries are fixed at zero.
    LLM_TIMEOUT_SECONDS: int | None = _integer_from_env("LLM_TIMEOUT_SECONDS", 30)
    LLM_MAX_OUTPUT_TOKENS: int | None = _integer_from_env("LLM_MAX_OUTPUT_TOKENS", 400)

    # Model-call accounting. Real generation stays disabled until every value is set.
    MODEL_CALL_LEDGER_PATH: str = os.getenv("MODEL_CALL_LEDGER_PATH", "")
    MAX_MODEL_CALLS_PER_DAY: int | None = _integer_from_env("MAX_MODEL_CALLS_PER_DAY", 0)
    MAX_MODEL_CALLS_TOTAL: int | None = _integer_from_env("MAX_MODEL_CALLS_TOTAL", 0)
    MAX_MODEL_TOKENS_PER_DAY: int | None = _integer_from_env("MAX_MODEL_TOKENS_PER_DAY", 0)
    MAX_MODEL_TOKENS_TOTAL: int | None = _integer_from_env("MAX_MODEL_TOKENS_TOTAL", 0)

    # Access control for the shared demo. Empty means protected routes return 503.
    DEMO_ACCESS_TOKEN: str = os.getenv("DEMO_ACCESS_TOKEN", "")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "")

    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Vector Store
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "./vectorstore")

    # Chunking
    CHUNK_SIZE: int | None = _chunk_integer_from_env("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int | None = _chunk_integer_from_env("CHUNK_OVERLAP", 100)

    # Resource bounds (initial demo bounds, not proven production capacities).
    MAX_FILE_SIZE_MB: int | None = _integer_from_env("MAX_FILE_SIZE_MB", 10)  # MiB
    MAX_PDF_PAGES: int | None = _integer_from_env("MAX_PDF_PAGES", 60)
    MAX_CHUNKS_PER_PAPER: int | None = _integer_from_env("MAX_CHUNKS_PER_PAPER", 600)
    MAX_PAPERS: int | None = _integer_from_env("MAX_PAPERS", 20)
    MAX_TOTAL_CHUNKS: int | None = _integer_from_env("MAX_TOTAL_CHUNKS", 3000)
    MAX_CONTEXT_CHARS: int | None = _integer_from_env("MAX_CONTEXT_CHARS", 6000)
    MAX_CONCURRENT_QUERIES: int | None = _integer_from_env("MAX_CONCURRENT_QUERIES", 2)

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int | None = _integer_from_env("PORT", 8001)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def resource_limits(self) -> ResourceLimits:
        """Validated ceilings; raises ValueError for malformed or non-positive values."""
        values = {
            "MAX_FILE_SIZE_MB": (self.MAX_FILE_SIZE_MB, 64),
            "MAX_PDF_PAGES": (self.MAX_PDF_PAGES, 1000),
            "MAX_CHUNKS_PER_PAPER": (self.MAX_CHUNKS_PER_PAPER, 20000),
            "MAX_PAPERS": (self.MAX_PAPERS, 1000),
            "MAX_TOTAL_CHUNKS": (self.MAX_TOTAL_CHUNKS, 200000),
            "MAX_CONTEXT_CHARS": (self.MAX_CONTEXT_CHARS, 100000),
            "MAX_CONCURRENT_QUERIES": (self.MAX_CONCURRENT_QUERIES, 8),
        }
        for name, (value, maximum) in values.items():
            if not _is_positive_integer(value, maximum):
                raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
        max_file_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        return ResourceLimits(
            max_file_bytes=max_file_bytes,
            max_upload_request_bytes=max_file_bytes + MULTIPART_ALLOWANCE_BYTES,
            max_json_body_bytes=MAX_JSON_BODY_BYTES,
            max_pdf_pages=self.MAX_PDF_PAGES,
            max_chunks_per_paper=self.MAX_CHUNKS_PER_PAPER,
            max_papers=self.MAX_PAPERS,
            max_total_chunks=self.MAX_TOTAL_CHUNKS,
            max_question_chars=MAX_QUESTION_CHARS,
            max_top_k=MAX_TOP_K,
            max_context_chars=self.MAX_CONTEXT_CHARS,
            max_concurrent_queries=self.MAX_CONCURRENT_QUERIES,
        )

    def budget_allowances(self) -> BudgetAllowances | None:
        """Explicit allowances, or None when accounting is not configured.

        Raises ValueError for malformed values. Zero (the default) means the
        allowance is not configured, which keeps paid inference disabled.
        """
        values = (
            self.MAX_MODEL_CALLS_PER_DAY, self.MAX_MODEL_CALLS_TOTAL,
            self.MAX_MODEL_TOKENS_PER_DAY, self.MAX_MODEL_TOKENS_TOTAL,
        )
        if not all(_is_non_negative_integer(value) for value in values):
            raise ValueError("Model-call allowances must be non-negative integers.")
        if not isinstance(self.MODEL_CALL_LEDGER_PATH, str):
            raise ValueError("MODEL_CALL_LEDGER_PATH must be a string.")
        if not self.MODEL_CALL_LEDGER_PATH.strip() or not all(values):
            return None
        return BudgetAllowances(*values)

    def access_token_configured(self) -> bool:
        return access_token_is_valid(self.DEMO_ACCESS_TOKEN)

    def allowed_origins(self) -> list[str]:
        """Explicit origins only; raises ValueError for wildcards or malformed entries."""
        if not isinstance(self.ALLOWED_ORIGINS, str):
            raise ValueError("ALLOWED_ORIGINS must be a comma-separated string.")
        origins = []
        for entry in self.ALLOWED_ORIGINS.split(","):
            origin = entry.strip()
            if not origin:
                continue
            if (
                not origin.startswith(("http://", "https://"))
                or "*" in origin
                or "/" in origin.split("://", 1)[1]
                or any(char.isspace() for char in origin)
            ):
                raise ValueError("ALLOWED_ORIGINS entries must be explicit http(s) origins.")
            origins.append(origin)
        return origins

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
        self.resource_limits()
        self.allowed_origins()
        if not _is_positive_integer(self.LLM_TIMEOUT_SECONDS, 300):
            raise ValueError("LLM_TIMEOUT_SECONDS must be an integer between 1 and 300.")
        if not _is_positive_integer(self.LLM_MAX_OUTPUT_TOKENS, 4096):
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be an integer between 1 and 4096.")
        self.budget_allowances()
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


def access_token_is_valid(value: object) -> bool:
    """A usable shared credential: long enough, printable ASCII, no whitespace."""
    return (
        type(value) is str
        and MIN_ACCESS_TOKEN_CHARS <= len(value) <= MAX_ACCESS_TOKEN_CHARS
        and value.isascii()
        and value.isprintable()
        and not any(char.isspace() for char in value)
    )


settings = Settings()
