"""
Research Paper RAG API
=======================
A production-grade RAG (Retrieval-Augmented Generation) system that lets you
upload research papers (PDF) and ask questions with cited answers.

Stack: FastAPI + LangChain + ChromaDB + OpenAI/Ollama + Docker

Author: Arjun Ponnaganti
LinkedIn: https://linkedin.com/in/arjun-ponnaganti
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_engine import RAGEngine
from app.config import settings

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-api")


# ─── Lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    logger.info("Starting Research Paper RAG API...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    yield
    logger.info("Shutting down RAG API.")


# ─── RAG Engine ───
rag = RAGEngine()

# ─── FastAPI App ───
app = FastAPI(
    title="Research Paper RAG API",
    description=(
        "Production-oriented RAG API with modular architecture, containerization, "
        "CI/CD, and structured observability. Upload research papers (PDF) and ask "
        "questions with page-level cited answers. Uses LangChain for orchestration, "
        "ChromaDB for vector storage, and OpenAI/Ollama for LLM inference."
    ),
    version="1.0.0",
    contact={
        "name": "Arjun Ponnaganti",
        "url": "https://linkedin.com/in/arjun-ponnaganti",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Middleware ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───
class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    papers_loaded: int = 0
    total_chunks: int = 0
    llm_provider: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    pages: int
    chunks: int
    processing_time_ms: float
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    paper_id: Optional[str] = Field(
        default=None,
        description="Query a specific paper. If None, searches all papers.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class Citation(BaseModel):
    text: str
    page: Optional[int] = None
    paper: str
    relevance_score: float


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


class PaperInfo(BaseModel):
    paper_id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


# ─── Track uptime ───
_start_time = time.time()


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
            <p class="footer">Built by <a href="https://linkedin.com/in/arjun-ponnaganti">Arjun Ponnaganti</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and knowledge base status."""
    stats = rag.get_stats()
    return HealthResponse(
        papers_loaded=stats["papers_loaded"],
        total_chunks=stats["total_chunks"],
        llm_provider=settings.LLM_PROVIDER,
    )


@app.post("/papers/upload", response_model=UploadResponse, tags=["Papers"])
async def upload_paper(
    file: UploadFile = File(..., description="Research paper PDF to upload"),
):
    """
    Upload a research paper (PDF) to the knowledge base.

    The paper is:
    1. Extracted (text from each page)
    2. Chunked (split into overlapping segments)
    3. Embedded (converted to vectors)
    4. Stored (indexed in ChromaDB for retrieval)
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"Invalid content type: {file.content_type}")

    # Read file
    try:
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_MB}MB.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Process paper
    start = time.time()
    try:
        result = rag.ingest_paper(contents, file.filename)
        elapsed = (time.time() - start) * 1000
    except Exception as e:
        logger.error(f"Paper ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    logger.info(
        f"Paper uploaded: {file.filename} | {result['pages']} pages | "
        f"{result['chunks']} chunks | {elapsed:.0f}ms"
    )

    return UploadResponse(
        paper_id=result["paper_id"],
        filename=file.filename,
        pages=result["pages"],
        chunks=result["chunks"],
        processing_time_ms=round(elapsed, 2),
        message=f"Paper '{file.filename}' processed successfully. {result['chunks']} chunks indexed.",
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_papers(request: QueryRequest):
    """
    Ask a question about uploaded research papers.

    Returns an LLM-generated answer with page-level citations
    from the most relevant paper chunks.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Query: {request.question[:80]}...")

    if rag.get_stats()["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No papers uploaded yet. Upload a paper first via POST /papers/upload.",
        )

    try:
        result = rag.query(
            question=request.question,
            paper_id=request.paper_id,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    citations = [
        Citation(
            text=c["text"],
            page=c.get("page"),
            paper=c["paper"],
            relevance_score=round(c["score"], 4),
        )
        for c in result["citations"]
    ]

    logger.info(
        f"[{request_id}] Answer generated | {len(citations)} citations | "
        f"retrieval={result['retrieval_time_ms']:.0f}ms | "
        f"generation={result['generation_time_ms']:.0f}ms"
    )

    return QueryResponse(
        request_id=request_id,
        question=request.question,
        answer=result["answer"],
        citations=citations,
        papers_searched=result["papers_searched"],
        retrieval_time_ms=round(result["retrieval_time_ms"], 2),
        generation_time_ms=round(result["generation_time_ms"], 2),
        total_time_ms=round(result["retrieval_time_ms"] + result["generation_time_ms"], 2),
        model_used=result["model_used"],
    )


@app.get("/papers", response_model=list[PaperInfo], tags=["Papers"])
async def list_papers():
    """List all uploaded papers in the knowledge base."""
    return rag.list_papers()


@app.delete("/papers/{paper_id}", tags=["Papers"])
async def delete_paper(paper_id: str):
    """Remove a paper from the knowledge base."""
    success = rag.delete_paper(paper_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"message": f"Paper '{paper_id}' deleted successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
