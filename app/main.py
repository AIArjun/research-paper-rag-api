"""
Research Paper RAG API
=======================
A research RAG (Retrieval-Augmented Generation) prototype that lets you
upload research papers (PDF) and ask questions with cited answers.

Stack: FastAPI + LangChain + ChromaDB + OpenAI/Ollama + Docker

Stage 2c protects one shared public-paper demo: bearer-token access, bounded
request and corpus sizes, single-admission ingestion, bounded query
concurrency, bounded provider use with a persistent call ledger, and
categorical errors and logs. One worker, one instance.

Author: Arjun Ponnaganti
LinkedIn: https://linkedin.com/in/arjun-ponnaganti
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.rag_engine import (
    RAGEngine, StorageMutationError, BackendUnavailableError, GenerationError,
    LimitExceededError, PDFExtractionError, BudgetExhaustedError, LedgerError,
)
from app.config import settings, MAX_QUESTION_CHARS, MAX_TOP_K, MAX_FILENAME_CHARS, RETRY_AFTER_SECONDS
from app.diagnostics import log_safe_error
from app.protection import (
    AccessTokenMiddleware, AdmissionMiddleware, RequestBodyLimitMiddleware, AdmissionSlot,
)

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-api")

_UPLOAD_READ_CHUNK = 256 * 1024


# ─── Lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    logger.info("Starting Research Paper RAG API...")
    logger.info("LLM Provider: %s", settings.LLM_PROVIDER)
    logger.info("Embedding Model: %s", settings.EMBEDDING_MODEL)
    logger.info("Access token configured: %s", settings.access_token_configured())
    yield
    logger.info("Shutting down RAG API.")


# ─── RAG Engine and admission ───
rag = RAGEngine()


def _configured_query_capacity() -> int:
    try:
        return settings.resource_limits().max_concurrent_queries
    except ValueError:
        return 1


# One mutation (upload or delete) at a time; a small fixed number of queries.
mutation_slot = AdmissionSlot(1, name="mutation")
query_slot = AdmissionSlot(_configured_query_capacity(), name="query")

# ─── FastAPI App ───
app = FastAPI(
    title="Research Paper RAG API",
    description=(
        "Protected shared demo of a RAG API. Upload research papers (PDF) and ask "
        "questions with page-level cited answers. Uses LangChain for orchestration, "
        "ChromaDB for vector storage, and OpenAI/Ollama for LLM inference. "
        "Every route except the landing page, docs, /health and /ready requires "
        "`Authorization: Bearer <DEMO_ACCESS_TOKEN>`."
    ),
    version="1.1.0",
    contact={
        "name": "Arjun Ponnaganti",
        "url": "https://linkedin.com/in/arjun-ponnaganti",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Documents the scheme in OpenAPI only; enforcement happens in the middleware.
bearer_scheme = HTTPBearer(auto_error=False, description="Shared demo access token")


def _body_limit_for_scope(scope: dict) -> int:
    limits = settings.resource_limits()
    if scope.get("path") == "/papers/upload":
        return limits.max_upload_request_bytes
    return limits.max_json_body_bytes


def _is_upload_request(scope: dict) -> bool:
    return scope.get("method") == "POST" and scope.get("path") == "/papers/upload"


# ─── Middleware (last added is outermost: CORS → access token → upload admission → body limit → app) ───
app.add_middleware(RequestBodyLimitMiddleware, limit_for_scope=_body_limit_for_scope)
# Uploads take the single mutation slot before any body byte is received; the
# slot resolver is a callable so a replaced module-level slot is honored.
app.add_middleware(AdmissionMiddleware, slot=lambda: mutation_slot, matches=_is_upload_request,
                   retry_after=RETRY_AFTER_SECONDS)
app.add_middleware(AccessTokenMiddleware, configured_token=lambda: settings.DEMO_ACCESS_TOKEN)


def _safe_allowed_origins() -> list[str]:
    try:
        return settings.allowed_origins()
    except ValueError:
        return []


app.add_middleware(
    CORSMiddleware,
    allow_origins=_safe_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ─── Error helpers ───
def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


def _error(status_code: int, message: str, category: str, request_id: str,
           headers: Optional[dict] = None, **extra) -> HTTPException:
    detail = {"message": message, "category": category, "request_id": request_id, **extra}
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _busy(request_id: str) -> HTTPException:
    return _error(
        429, "The demo is busy with another request. Retry shortly.", "busy", request_id,
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


def _limit_rejection(error: LimitExceededError, request_id: str) -> HTTPException:
    messages = {
        "file_too_large": (413, "The PDF exceeds the maximum file size."),
        "too_many_pages": (413, "The PDF exceeds the maximum page count."),
        "too_many_chunks": (413, "The PDF produces more chunks than a single paper may hold."),
        "paper_limit_reached": (409, "The demo corpus already holds the maximum number of papers. Delete one first."),
        "corpus_capacity_reached": (409, "The demo corpus has no room for this paper's chunks. Delete a paper first."),
        "question_too_long": (422, "The question exceeds the maximum length."),
    }
    status_code, message = messages.get(error.category, (400, "A configured limit was exceeded."))
    return _error(status_code, message, error.category, request_id, limit=error.limit)


def _budget_rejection(error: BudgetExhaustedError, request_id: str) -> HTTPException:
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else None
    return _error(
        429, "The model-call allowance is exhausted; no model request was made.",
        "budget_exhausted", request_id, headers=headers, scope=error.scope, kind=error.kind,
    )


_FRAMEWORK_ERRORS = {
    400: ("The request could not be parsed.", "invalid_request"),
    404: ("Not found.", "not_found"),
    405: ("Method not allowed.", "method_not_allowed"),
    413: ("The request body exceeds the configured limit.", "request_too_large"),
}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Give framework-raised errors (404, 405, multipart parse failures) the same categorical shape."""
    detail = exc.detail
    if not isinstance(detail, dict):
        message, category = _FRAMEWORK_ERRORS.get(exc.status_code, ("Request failed.", "http_error"))
        detail = {"message": message, "category": category}
    detail = {**detail}
    detail.setdefault("request_id", _request_id(request))
    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Report field locations and error types without echoing submitted values."""
    errors = []
    for error in exc.errors():
        loc = [part for part in error.get("loc", ()) if isinstance(part, (str, int))]
        errors.append({"loc": loc, "type": error.get("type", "invalid"), "msg": error.get("msg", "")})
    return JSONResponse(status_code=422, content={"detail": {
        "message": "Request validation failed.",
        "category": "invalid_request",
        "request_id": _request_id(request),
        "errors": errors,
    }})


# ─── Models ───
class ModelBudgetStatus(BaseModel):
    state: str = "not_applicable"
    configured: bool = False
    usage: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.1.0"
    papers_loaded: int = 0
    total_chunks: int = 0
    llm_provider: str = ""
    ready: bool = False
    configured_provider: str = ""
    configured_model: str = ""
    effective_retrieval: str = "unavailable"
    effective_generation: str = "unavailable"
    init_error: Optional[str] = None
    pending_cleanup_ids: list[str] = Field(default_factory=list)
    provider_connection_verified: bool = False
    access_configured: bool = False
    model_budget: ModelBudgetStatus = Field(default_factory=ModelBudgetStatus)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ReadinessResponse(BaseModel):
    ready: bool
    configured_provider: str
    configured_model: str
    effective_retrieval: str
    effective_generation: str
    init_error: Optional[str] = None
    pending_cleanup_ids: list[str] = Field(default_factory=list)
    provider_connection_verified: bool = False
    access_configured: bool = False
    model_budget: ModelBudgetStatus = Field(default_factory=ModelBudgetStatus)
    limits: Optional[dict] = None


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    pages: int
    chunks: int
    processing_time_ms: float
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_QUESTION_CHARS)
    paper_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Query a specific paper. If None, searches all papers.",
    )
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, description="Number of context chunks to retrieve")


class Citation(BaseModel):
    text: str
    page: Optional[int] = None
    paper: str
    relevance_score: float
    paper_id: Optional[str] = None
    chunk_id: Optional[str] = None


class ModelUsage(BaseModel):
    accounting: str  # "measured" (provider reported) or "reserved" (pre-call upper bound)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tokens_charged: int
    tokens_reserved: int
    context_chars: int
    reservation_bound: Optional[str] = None  # e.g. "tiktoken/o200k_base"


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[Citation]
    papers_searched: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    model_usage: Optional[ModelUsage] = None


class PaperInfo(BaseModel):
    paper_id: str
    filename: Optional[str] = None
    pages: Optional[int] = None
    chunks: int
    uploaded_at: Optional[str] = None
    status: str = "ready"


# ─── Readiness composition ───
async def _readiness_payload() -> dict:
    readiness = rag.get_readiness()
    access_configured = settings.access_token_configured()
    # The ledger summary is a local read-only SQLite query; keep it off the event loop.
    budget = await run_in_threadpool(rag.get_budget_status)
    try:
        limits = settings.resource_limits().as_dict()
    except ValueError:
        limits = None
    return {
        **readiness,
        "ready": bool(readiness["ready"] and access_configured and budget["state"] != "unavailable"),
        "access_configured": access_configured,
        "model_budget": budget,
        "limits": limits,
    }


def _assert_ready_for_work(request_id: str, operation: str) -> None:
    try:
        rag.assert_storage_ready()
        rag.assert_backend_ready()
    except StorageMutationError as error:
        raise _error(
            503, "Paper storage requires cleanup. Retry deletion of this ID first.",
            "storage_cleanup_required", request_id, paper_id=error.paper_id, cleanup_required=True,
        )
    except BackendUnavailableError as error:
        raise _error(
            503, f"The configured backend is not ready for {operation}.", error.category, request_id,
        )


async def _read_bounded_upload(file: UploadFile, limit: int, request_id: str) -> bytes:
    """Read the parsed file part in chunks and stop at the first byte over the ceiling."""
    parts = []
    total = 0
    try:
        while True:
            chunk = await file.read(_UPLOAD_READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise _error(413, "The PDF exceeds the maximum file size.", "file_too_large",
                             request_id, limit=limit)
            parts.append(chunk)
    except HTTPException:
        raise
    except Exception as error:
        log_safe_error(logger, "ingestion_failed", error)
        raise _error(400, "The uploaded file could not be read.", "unreadable_upload", request_id)
    return b"".join(parts)


# ─── Endpoints ───

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Paper RAG API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a1a; color: #e0e0e0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { max-width: 700px; padding: 48px; text-align: center; }
            h1 { font-size: 2.2rem; color: #fff; margin-bottom: 8px; }
            .accent { color: #a78bfa; }
            .subtitle { color: #888; font-size: 1rem; margin-bottom: 32px; }
            .features { display: flex; gap: 20px; justify-content: center; margin: 32px 0; flex-wrap: wrap; }
            .feature { background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2); border-radius: 12px; padding: 18px 24px; min-width: 180px; }
            .feature .icon { font-size: 1.6rem; margin-bottom: 6px; }
            .feature .label { font-size: 0.85rem; color: #aaa; }
            .links { display: flex; gap: 16px; justify-content: center; margin-top: 32px; }
            a.btn { display: inline-block; padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 0.95rem; transition: all 0.2s; }
            a.primary { background: #a78bfa; color: #0a0a1a; }
            a.primary:hover { background: #c4b5fd; }
            a.secondary { border: 1px solid #a78bfa; color: #a78bfa; }
            a.secondary:hover { background: rgba(167,139,250,0.1); }
            .footer { margin-top: 48px; font-size: 0.85rem; color: #555; }
            .footer a { color: #a78bfa; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>&#128218; Research Paper <span class="accent">RAG API</span></h1>
            <p class="subtitle">Upload papers. Ask questions. Get cited answers.</p>
            <div class="features">
                <div class="feature"><div class="icon">&#128196;</div>PDF Processing<div class="label">Chunk &amp; embed papers</div></div>
                <div class="feature"><div class="icon">&#128269;</div>Semantic Search<div class="label">ChromaDB vectors</div></div>
                <div class="feature"><div class="icon">&#129302;</div>LLM Answers<div class="label">OpenAI / Ollama</div></div>
                <div class="feature"><div class="icon">&#128206;</div>Citations<div class="label">Page-level sources</div></div>
            </div>
            <div class="links">
                <a href="/docs" class="btn primary">API Documentation</a>
                <a href="/redoc" class="btn secondary">ReDoc</a>
            </div>
            <p class="footer">Protected shared demo: API routes require a bearer access token.<br>Built by <a href="https://linkedin.com/in/arjun-ponnaganti">Arjun Ponnaganti</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Report local component readiness; this does not call the provider."""
    stats = rag.get_stats()
    payload = await _readiness_payload()
    payload.pop("limits", None)
    response = HealthResponse(
        status="healthy" if payload["ready"] else "unready",
        papers_loaded=stats["papers_loaded"],
        total_chunks=stats["total_chunks"],
        llm_provider=payload["effective_generation"],
        **payload,
    )
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=response.model_dump())


@app.get("/ready", response_model=ReadinessResponse, tags=["System"])
async def readiness_check():
    """Cheap local readiness, without indexing, embedding or inference calls."""
    response = ReadinessResponse(**await _readiness_payload())
    return JSONResponse(status_code=200 if response.ready else 503, content=response.model_dump())


@app.post(
    "/papers/upload", response_model=UploadResponse, tags=["Papers"],
    dependencies=[Security(bearer_scheme)],
)
async def upload_paper(
    request: Request,
    file: UploadFile = File(..., description="Research paper PDF to upload"),
):
    """
    Upload a research paper (PDF) to the knowledge base.

    The paper is:
    1. Extracted (text from each page)
    2. Chunked (split into overlapping segments)
    3. Embedded (converted to vectors)
    4. Stored (indexed in ChromaDB for retrieval)

    One upload is admitted at a time. The admission is taken by the middleware
    before the multipart body is received, so a busy demo answers 429 before
    any bytes are read; the same admission is handed to the worker thread here.
    """
    request_id = _request_id(request)
    try:
        limits = settings.resource_limits()
    except ValueError:
        raise _error(503, "Resource limits are misconfigured.", "invalid_configuration", request_id)
    _assert_ready_for_work(request_id, "uploads")

    # Validate file type without echoing the submitted name or type.
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf") or len(filename) > MAX_FILENAME_CHARS:
        raise _error(400, "Only PDF files with a .pdf name are accepted.", "invalid_file_type", request_id)
    if file.content_type and file.content_type != "application/pdf":
        raise _error(400, "The file part must be sent as application/pdf.", "invalid_content_type", request_id)

    admission = getattr(request.state, "admission", None)
    if admission is None:
        # Defensive only: the middleware admits before the body is parsed.
        admission = mutation_slot.admit()
        if admission is None:
            raise _busy(request_id)
    start = time.time()
    try:
        contents = await _read_bounded_upload(file, limits.max_file_bytes, request_id)
        if not contents:
            raise _error(400, "The uploaded file is empty.", "empty_file", request_id)
        try:
            result = await admission.run(rag.ingest_paper, contents, filename)
        except BackendUnavailableError as error:
            raise _error(503, "The configured backend is not ready for uploads.", error.category, request_id)
        except StorageMutationError as error:
            logger.error("[%s] Paper ingestion requires storage recovery", request_id)
            raise _error(
                503,
                (
                    "Paper storage failed. Retry deletion of this ID to confirm cleanup before retrying the upload."
                    if error.cleanup_required else
                    "Paper indexing failed and partial data was removed. Retry the upload when storage is available."
                ),
                "storage_failure", request_id,
                paper_id=error.paper_id, cleanup_required=error.cleanup_required,
            )
        except LimitExceededError as error:
            raise _limit_rejection(error, request_id)
        except PDFExtractionError:
            raise _error(400, "The file could not be parsed as a text PDF.", "invalid_pdf", request_id)
        except ValueError:
            raise _error(400, "PDF text or chunk configuration is invalid.", "invalid_pdf_content", request_id)
        except Exception as error:
            log_safe_error(logger, "ingestion_failed", error)
            raise _error(500, "Processing failed.", "ingestion_failed", request_id)
    finally:
        admission.release()
    elapsed = (time.time() - start) * 1000

    logger.info(
        "[%s] Paper uploaded | paper=%s | %d pages | %d chunks | %.0fms",
        request_id, result["paper_id"][:12], result["pages"], result["chunks"], elapsed,
    )

    return UploadResponse(
        paper_id=result["paper_id"],
        filename=result["filename"],
        pages=result["pages"],
        chunks=result["chunks"],
        processing_time_ms=round(elapsed, 2),
        message=f"Paper '{result['filename']}' is indexed. {result['chunks']} chunks available.",
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"], dependencies=[Security(bearer_scheme)])
async def query_papers(request: Request, body: QueryRequest):
    """
    Ask a question about uploaded research papers.

    Returns an LLM-generated answer with page-level citations
    from the most relevant paper chunks. Question text is never logged.
    """
    request_id = _request_id(request)
    logger.info(
        "[%s] Query received | question_chars=%d | top_k=%d | filtered=%s",
        request_id, len(body.question), body.top_k, body.paper_id is not None,
    )
    try:
        settings.resource_limits()
    except ValueError:
        raise _error(503, "Resource limits are misconfigured.", "invalid_configuration", request_id)
    _assert_ready_for_work(request_id, "queries")
    if rag.get_stats()["total_chunks"] == 0:
        raise _error(
            400, "No papers uploaded yet. Upload a paper first via POST /papers/upload.",
            "empty_corpus", request_id,
        )
    if mutation_slot.busy:
        raise _busy(request_id)
    admission = query_slot.admit()
    if admission is None:
        raise _busy(request_id)

    try:
        result = await admission.run(
            rag.query,
            question=body.question,
            paper_id=body.paper_id,
            top_k=body.top_k,
            request_id=request_id,
        )
    except BackendUnavailableError as error:
        raise _error(503, "The configured backend is not ready for queries.", error.category, request_id)
    except BudgetExhaustedError as error:
        logger.warning("[%s] Model call refused: %s %s allowance exhausted", request_id, error.scope, error.kind)
        raise _budget_rejection(error, request_id)
    except LedgerError as error:
        log_safe_error(logger, "ledger_unavailable", error.__cause__ or error)
        raise _error(503, "Model-call accounting is unavailable; no model request was made.",
                     "ledger_unavailable", request_id)
    except GenerationError:
        logger.error("[%s] Model generation failed", request_id)
        raise _error(502, "The model could not generate an answer. Please retry later.",
                     "generation_failed", request_id)
    except StorageMutationError as error:
        logger.error("[%s] Query blocked pending storage cleanup", request_id)
        raise _error(503, "Paper storage requires cleanup before querying. Retry deletion of this ID.",
                     "storage_cleanup_required", request_id, paper_id=error.paper_id)
    except LimitExceededError as error:
        raise _limit_rejection(error, request_id)
    except Exception as error:
        log_safe_error(logger, "query_failed", error)
        raise _error(500, "Query failed.", "query_failed", request_id)
    finally:
        admission.release()

    citations = [
        Citation(
            text=c["text"],
            page=c.get("page"),
            paper=c["paper"],
            relevance_score=round(c["score"], 4),
            paper_id=c.get("paper_id"),
            chunk_id=c.get("chunk_id"),
        )
        for c in result["citations"]
    ]

    logger.info(
        "[%s] Answer generated | %d citations | retrieval=%.0fms | generation=%.0fms | model=%s",
        request_id, len(citations), result["retrieval_time_ms"], result["generation_time_ms"],
        result["model_used"],
    )

    usage = result.get("model_usage")
    return QueryResponse(
        request_id=request_id,
        question=body.question,
        answer=result["answer"],
        citations=citations,
        papers_searched=result["papers_searched"],
        retrieval_time_ms=round(result["retrieval_time_ms"], 2),
        generation_time_ms=round(result["generation_time_ms"], 2),
        total_time_ms=round(result["retrieval_time_ms"] + result["generation_time_ms"], 2),
        model_used=result["model_used"],
        model_usage=ModelUsage(**usage) if isinstance(usage, dict) else None,
    )


@app.get("/papers", response_model=list[PaperInfo], tags=["Papers"], dependencies=[Security(bearer_scheme)])
async def list_papers():
    """List all uploaded papers in the knowledge base."""
    return rag.list_papers(include_pending=True)


@app.delete("/papers/{paper_id}", tags=["Papers"], dependencies=[Security(bearer_scheme)])
async def delete_paper(request: Request, paper_id: str):
    """Remove a paper from the knowledge base."""
    request_id = _request_id(request)
    if len(paper_id) > 128:
        raise _error(404, "Paper not found.", "paper_not_found", request_id)
    admission = mutation_slot.admit()
    if admission is None:
        raise _busy(request_id)
    try:
        success = await admission.run(rag.delete_paper, paper_id)
    except StorageMutationError as error:
        logger.error("[%s] Paper deletion requires storage recovery", request_id)
        raise _error(503, "Paper deletion did not complete. Retry deletion; the paper is not confirmed removed.",
                     "storage_failure", request_id, paper_id=error.paper_id)
    except Exception as error:
        log_safe_error(logger, "ingestion_failed", error)
        raise _error(500, "Deletion failed.", "deletion_failed", request_id)
    finally:
        admission.release()
    if not success:
        raise _error(404, "Paper not found.", "paper_not_found", request_id)
    return {"message": "Paper deleted.", "paper_id": paper_id, "request_id": request_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
