"""Small allowlisted diagnostics that never render exception messages or payloads."""

import logging


_CATEGORIES = frozenset({
    "invalid_configuration", "missing_api_key", "missing_dependency",
    "embedding_initialization_failed", "storage_initialization_failed",
    "model_initialization_failed", "generation_failed", "ingestion_failed",
    "query_failed", "budget_not_configured", "ledger_unavailable",
})
_MODULE_ROOTS = frozenset({
    "builtins", "langchain_huggingface", "langchain_chroma", "langchain_openai",
    "langchain_ollama", "langchain_core", "sentence_transformers", "transformers",
    "torch", "chromadb", "openai", "ollama", "pydantic", "pydantic_core",
    "huggingface_hub", "httpx", "httpcore", "requests", "numpy", "tokenizers",
    "safetensors", "onnxruntime", "sqlite3", "pdfminer", "pdfplumber", "pypdf",
})
_EXCEPTION_TYPES = frozenset({
    "Exception", "RuntimeError", "ValueError", "TypeError", "ImportError",
    "ModuleNotFoundError", "OSError", "FileNotFoundError", "PermissionError",
    "MemoryError", "TimeoutError", "ConnectionError", "ValidationError",
    "OutOfMemoryError", "APIError", "APIStatusError", "APIConnectionError",
    "APITimeoutError", "APIResponseValidationError", "AuthenticationError",
    "PermissionDeniedError", "RateLimitError", "BadRequestError", "NotFoundError",
    "ConflictError", "UnprocessableEntityError", "InternalServerError",
    "HTTPError", "HTTPStatusError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "TimeoutException", "RequestError", "NetworkError", "ResponseError",
    "HfHubHTTPError", "RepositoryNotFoundError", "GatedRepoError",
    "RevisionNotFoundError", "EntryNotFoundError", "LocalEntryNotFoundError",
    "OfflineModeIsEnabled", "ChromaError", "InvalidDimensionException",
    "InvalidArgumentError", "InternalError", "DatabaseError", "OperationalError",
    "IntegrityError", "ProgrammingError", "PSException", "PDFSyntaxError",
    "PdfReadError", "PdfStreamError",
})
_STATUS_CODES = frozenset({400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504})


def _module_root(value: object) -> str:
    if type(value) is str:
        root = value.split(".", 1)[0]
        if root in _MODULE_ROOTS:
            return root
    return "unknown"


def _read_attribute(value: object, name: str) -> object:
    """A diagnostic must not replace the original error if an attribute raises."""
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def safe_error_details(category: str, error: BaseException | None = None) -> dict:
    """Return fixed labels and a known HTTP code, without traversing request/body data."""
    details = {
        "category": category if type(category) is str and category in _CATEGORIES else "unknown_error",
        "exception_type": "none",
        "exception_module": "none",
        "dependency_module": "none",
        "status_code": None,
    }
    if error is None:
        return details

    exception_class = type(error)
    module = _module_root(_read_attribute(exception_class, "__module__"))
    name = _read_attribute(exception_class, "__name__")
    details["exception_module"] = module
    details["exception_type"] = (
        name if module != "unknown" and type(name) is str and name in _EXCEPTION_TYPES else "unknown"
    )
    if isinstance(error, ImportError):
        details["dependency_module"] = _module_root(_read_attribute(error, "name"))

    status = _read_attribute(error, "status_code")
    if type(status) is not int or status not in _STATUS_CODES:
        response = _read_attribute(error, "response")
        status = _read_attribute(response, "status_code") if response is not None else None
    if type(status) is int and status in _STATUS_CODES:
        details["status_code"] = status
    return details


def log_safe_error(
    logger: logging.Logger, category: str, error: BaseException | None = None
) -> None:
    """Log only the allowlisted fields; do not attach exc_info or traceback objects."""
    details = safe_error_details(category, error)
    logger.error(
        "RAG diagnostic category=%s exception_type=%s exception_module=%s dependency_module=%s status_code=%s",
        details["category"], details["exception_type"], details["exception_module"],
        details["dependency_module"], details["status_code"],
    )
