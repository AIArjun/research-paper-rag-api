"""
RAG Engine
===========
Core retrieval-augmented generation engine.
Handles PDF extraction, chunking, embedding, vector storage, and LLM query.

Supports:
- OpenAI (GPT-4o-mini, GPT-4o) via API key
- Demo mode (no API key needed) for testing
- Ollama: configurable, but the protected demo fails closed
  (token_bound_unavailable) because the server-side Modelfile TEMPLATE/SYSTEM
  cannot be bounded from this client; see app/tokens.py

Bounds (Stage 2c): page, chunk, paper and corpus ceilings are checked before
embeddings or storage; every real-model call is reserved in a persistent
ledger before the provider is contacted; prompts are capped in size.

Locking: `_state_lock` guards the process-local registry and is only held
briefly, so readiness stays responsive. `_work_lock` serializes extraction,
indexing, deletion and retrieval, and is only held inside worker threads.
Order: work lock, then state lock. Provider calls hold neither.

Author: Arjun Ponnaganti
"""

import hashlib
import io
import logging
import time
from datetime import datetime, timezone
from threading import RLock
from typing import Optional

from app.config import settings
from app.diagnostics import log_safe_error
from app.ledger import BudgetExhaustedError, LedgerError, ModelCallLedger
from app.retrieval import CANDIDATE_LIMIT, evidence_window, hybrid_rank, lexical_rank
from app.tokens import TokenBound, TokenBoundError, token_bound_for

# Word boundary tolerance for pdfplumber as a fraction of the glyph size.
# Its fixed default (3 pt) is wider than the inter-word gap of common 10 pt
# body fonts (about 2.5 pt) in PDFs that position words without space glyphs,
# such as pdfTeX output, so whole lines were extracted as one run-together
# word and embedded as noise. 0.15 em sits between kerning gaps inside a word
# (near 0) and the narrowest justified inter-word gap (about 0.17 em).
WORD_GAP_RATIO = 0.15

logger = logging.getLogger("rag-api.engine")

__all__ = [
    "RAGEngine", "StorageMutationError", "BackendUnavailableError", "GenerationError",
    "LimitExceededError", "PDFExtractionError", "BudgetExhaustedError", "LedgerError",
]


class StorageMutationError(RuntimeError):
    """Storage work failed or needs cleanup before the corpus is usable."""

    def __init__(self, message: str, paper_id: Optional[str] = None, cleanup_required: bool = True):
        super().__init__(message)
        self.paper_id = paper_id
        self.cleanup_required = cleanup_required


class BackendUnavailableError(RuntimeError):
    """The requested pipeline has no usable initialized backend."""

    def __init__(self, category: str = "backend_unavailable"):
        super().__init__("The configured retrieval/generation backend is unavailable.")
        self.category = category


class GenerationError(RuntimeError):
    """A configured model failed to produce an answer; never a demo fallback."""


class LimitExceededError(RuntimeError):
    """A configured ceiling would be exceeded; nothing was stored or generated."""

    def __init__(self, category: str, limit: int, observed: Optional[int] = None):
        super().__init__(f"Limit exceeded: {category}")
        self.category = category
        self.limit = limit
        self.observed = observed


class PDFExtractionError(ValueError):
    """The upload could not be parsed as a text PDF; details stay out of responses."""


def _safe_attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _measured_usage(response: object) -> tuple[Optional[int], Optional[int]]:
    """Read provider-reported token counts when present; never inspect anything else."""
    usage = _safe_attribute(response, "usage_metadata")
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if type(input_tokens) is int and type(output_tokens) is int and input_tokens >= 0 and output_tokens >= 0:
        return input_tokens, output_tokens
    return None, None


def _response_text(response: object) -> str:
    """Extract answer text from a string, a chat message or text content blocks."""
    if isinstance(response, str):
        return response
    content = _safe_attribute(response, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        if parts:
            return "".join(parts)
    raise GenerationError("The configured model returned an unsupported response type.")


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for research papers.

    Pipeline:
        1. PDF → text extraction (per page)
        2. Text → chunks (recursive splitting with overlap)
        3. Chunks → embeddings (sentence-transformers or OpenAI)
        4. Embeddings → ChromaDB vector store
        5. Query → semantic retrieval → LLM generation with citations
    """

    def __init__(self):
        self.papers: dict[str, dict] = {}
        self.chunks_store: list[dict] = []
        self._page_texts: dict[str, dict[int, str]] = {}
        # Failed storage mutations remain recoverable by retrying deletion.
        # This guard is process-local; durable recovery is a separate milestone.
        self._pending_cleanup: dict[str, list[str]] = {}
        self._pending_metadata: dict[str, dict] = {}
        self._state_lock = RLock()
        self._work_lock = RLock()
        self._configured_provider = settings.LLM_PROVIDER
        self._configured_model = settings.LLM_MODEL
        self._init_error: Optional[str] = None
        self._vectorstore = None
        self._embeddings = None
        self._llm = None
        self._ledger: Optional[ModelCallLedger] = None
        self._token_bound: Optional[TokenBound] = None
        self._initialize()

    def _initialize(self):
        """Record explicit initialization failures without activating a demo fallback."""
        self._embeddings = None
        self._vectorstore = None
        self._llm = None
        self._ledger = None
        self._token_bound = None
        self._init_error = None
        try:
            settings.validate()
        except (ValueError, TypeError) as error:
            self._init_error = "invalid_configuration"
            log_safe_error(logger, self._init_error, error)
            return

        if self._configured_provider == "demo":
            logger.info("Running in demo mode — skipping embedding model and vector store.")
            return
        if self._configured_provider == "openai" and not settings.OPENAI_API_KEY.strip():
            self._init_error = "missing_api_key"
            log_safe_error(logger, self._init_error)
            return

        # Paid inference stays disabled until allowances and a ledger path are explicit.
        allowances = settings.budget_allowances()
        if allowances is None:
            self._init_error = "budget_not_configured"
            log_safe_error(logger, self._init_error)
            return
        try:
            self._ledger = ModelCallLedger(settings.MODEL_CALL_LEDGER_PATH, allowances)
        except LedgerError as error:
            self._init_error = "ledger_unavailable"
            log_safe_error(logger, self._init_error, error.__cause__ or error)
            return
        # Every reservation needs a model-supported upper bound; resolve it before
        # loading any heavy backend so an unsupported model fails closed cheaply.
        try:
            self._token_bound = token_bound_for(self._configured_provider, self._configured_model)
        except ImportError as error:
            self._init_error = "missing_dependency"
            self._ledger = None
            log_safe_error(logger, self._init_error, error)
            return
        except TokenBoundError as error:
            self._init_error = "token_bound_unavailable"
            self._ledger = None
            log_safe_error(logger, self._init_error, error.__cause__ or error)
            return

        failure_category = "embedding_initialization_failed"
        initialization_error = None
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            logger.info("Initializing embedding backend")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 32},
            )
            failure_category = "storage_initialization_failed"
            self._vectorstore = Chroma(
                collection_name="research_papers",
                embedding_function=self._embeddings,
                persist_directory=settings.VECTORSTORE_PATH,
            )
            logger.info("Vector store initialized (ChromaDB).")
            failure_category = "model_initialization_failed"
            self._init_llm()
        except ImportError as error:
            self._init_error = "missing_dependency"
            initialization_error = error
        except Exception as error:
            self._init_error = failure_category
            initialization_error = error
        if self._init_error is not None:
            self._embeddings = None
            self._vectorstore = None
            self._llm = None
            self._ledger = None
            self._token_bound = None
            log_safe_error(logger, self._init_error, initialization_error)

    def _init_llm(self):
        """Construct a bounded client; this does not verify remote credentials/connectivity.

        Every provider receives an explicit timeout and output cap and performs
        no automatic retries. A provider that cannot accept these bounds fails
        construction and is reported as model_initialization_failed. The Ollama
        branch is reached only once a model/template-specific token bound
        exists; today token_bound_for() fails closed before this point.
        """
        timeout = settings.LLM_TIMEOUT_SECONDS
        max_output_tokens = settings.LLM_MAX_OUTPUT_TOKENS
        if self._configured_provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self._configured_model,
                temperature=0.1,
                api_key=settings.OPENAI_API_KEY,
                timeout=timeout,
                max_retries=0,
                max_tokens=max_output_tokens,
            )
        elif self._configured_provider == "ollama":
            from langchain_ollama import OllamaLLM

            self._llm = OllamaLLM(
                model=self._configured_model,
                base_url=settings.OLLAMA_URL,
                validate_model_on_init=False,
                num_predict=max_output_tokens,
                client_kwargs={"timeout": timeout},
            )
        else:
            raise BackendUnavailableError("invalid_configuration")

    def get_readiness(self) -> dict:
        """Cheap local pipeline state; no model request, storage read or network probe."""
        with self._state_lock:
            real = isinstance(self._configured_provider, str) and self._configured_provider in {"openai", "ollama"}
            vector_ready = self._vectorstore is not None and self._embeddings is not None
            if self._init_error is not None:
                retrieval, generation = "unavailable", "unavailable"
            elif real:
                retrieval = "chroma" if vector_ready else "unavailable"
                generation = (
                    self._configured_provider
                    if self._llm is not None and self._ledger is not None and self._token_bound is not None
                    else "unavailable"
                )
            else:
                retrieval = "chroma" if vector_ready else "memory-keyword"
                generation = "demo"
            return {
                "configured_provider": (
                    self._configured_provider if isinstance(self._configured_provider, str) else "invalid"
                ),
                "configured_model": (
                    self._configured_model if isinstance(self._configured_model, str) else "invalid"
                ),
                "effective_retrieval": retrieval,
                "effective_generation": generation,
                "ready": (
                    retrieval != "unavailable" and generation != "unavailable"
                    and not self._pending_cleanup
                ),
                "init_error": self._init_error,
                "pending_cleanup_ids": list(self._pending_cleanup),
                # Local client construction is not evidence of a remote call.
                "provider_connection_verified": False,
            }

    def get_budget_status(self) -> dict:
        """Model-call accounting state: counts only, never prompts or paths.

        Unlike get_readiness() this reads the local ledger file (read-only), so
        the API calls it from a worker thread. A ledger that became unusable
        after startup is reported as "unavailable" and makes the API unready.
        """
        with self._state_lock:
            provider = self._configured_provider
            ledger = self._ledger
            init_error = self._init_error
            bound = self._token_bound
        status = {
            "state": "not_applicable", "configured": False, "usage": None,
            "token_bound": bound.name if bound is not None else None,
        }
        if not (isinstance(provider, str) and provider in {"openai", "ollama"}):
            return status
        if ledger is None:
            status["state"] = "not_configured" if init_error in (None, "budget_not_configured") else "unavailable"
            return status
        status["configured"] = True
        try:
            status["usage"] = ledger.summary()
            status["state"] = "ok"
        except LedgerError as error:
            log_safe_error(logger, "ledger_unavailable", error.__cause__ or error)
            status["state"] = "unavailable"
        return status

    def assert_backend_ready(self) -> None:
        """Fail closed for an invalid or incomplete requested real pipeline."""
        with self._state_lock:
            if self._init_error is not None:
                raise BackendUnavailableError(self._init_error)
            if self._configured_provider in {"openai", "ollama"}:
                if self._vectorstore is None or self._embeddings is None or self._llm is None:
                    raise BackendUnavailableError()
                if self._ledger is None:
                    raise BackendUnavailableError("budget_not_configured")
                if self._token_bound is None:
                    raise BackendUnavailableError("token_bound_unavailable")

    def _extract_pdf(self, pdf_bytes: bytes) -> list[dict]:
        """Extract text from PDF, page by page, after checking the page ceiling."""
        max_pages = settings.resource_limits().max_pdf_pages
        try:
            try:
                import pdfplumber

                pages = []
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    self._check_page_count(len(pdf.pages), max_pages)
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text(x_tolerance_ratio=WORD_GAP_RATIO) or ""
                        if text.strip():
                            pages.append({"page": i + 1, "text": text.strip()})
                return pages
            except ImportError:
                # Fallback to pypdf
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(pdf_bytes))
                self._check_page_count(len(reader.pages), max_pages)
                pages = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i + 1, "text": text.strip()})
                return pages
        except (LimitExceededError, MemoryError):
            raise
        except Exception as error:
            raise PDFExtractionError("The PDF could not be parsed.") from error

    @staticmethod
    def _check_page_count(page_count: int, max_pages: Optional[int]) -> None:
        if max_pages is not None and page_count > max_pages:
            raise LimitExceededError("too_many_pages", max_pages, page_count)

    def _chunk_text(
        self,
        pages: list[dict],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> list[dict]:
        """Split page texts into overlapping chunks with strictly advancing offsets."""
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_size <= 0
            or not 0 <= chunk_overlap < chunk_size
        ):
            raise ValueError("Chunk settings require integer size > 0 and 0 <= overlap < size.")

        chunks = []
        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page"]

            # Simple recursive splitting by sentences/paragraphs
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))

                # Try to break at sentence boundary
                if end < len(text):
                    last_period = text.rfind(".", start, end)
                    last_newline = text.rfind("\n", start, end)
                    break_at = max(last_period, last_newline)
                    # A short sentence boundary must not send the next offset
                    # backward, or leave it unchanged after applying overlap.
                    if break_at + 1 > start + chunk_overlap:
                        end = break_at + 1

                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "char_start": start,
                        "char_end": end,
                    })
                start = end - chunk_overlap if end < len(text) else len(text)

        return chunks

    # ─── Capacity accounting (state lock held by callers) ───

    def _occupied_paper_count(self) -> int:
        return len(set(self.papers) | set(self._pending_cleanup))

    def _occupied_chunk_count(self) -> int:
        ready = sum(paper["chunks"] for paper in self.papers.values())
        pending = sum(
            len(ids) for paper_id, ids in self._pending_cleanup.items() if paper_id not in self.papers
        )
        return ready + pending

    def ingest_paper(self, pdf_bytes: bytes, filename: str) -> dict:
        """
        Process and index a research paper.

        Identical bytes are idempotent within the loaded registry: retain the
        original filename, upload time and chunks without indexing again, even
        when the corpus is at capacity.

        Returns:
            dict with paper_id, canonical filename, pages count, chunks count
        """
        with self._work_lock:
            self.assert_backend_ready()
            return self._ingest_paper(pdf_bytes, filename)

    def _ingest_paper(self, pdf_bytes: bytes, filename: str) -> dict:
        limits = settings.resource_limits()
        paper_id = hashlib.sha256(pdf_bytes).hexdigest()
        with self._state_lock:
            if paper_id in self._pending_cleanup:
                raise StorageMutationError(
                    f"Paper '{paper_id}' needs storage cleanup; retry deletion before uploading.",
                    paper_id=paper_id,
                )
            if paper_id in self.papers:
                paper = self.papers[paper_id]
                return {
                    "paper_id": paper_id,
                    "filename": paper["filename"],
                    "pages": paper["pages"],
                    "chunks": paper["chunks"],
                }
            # Cheap ceilings first: no parsing for a corpus that cannot accept a paper.
            if len(pdf_bytes) > limits.max_file_bytes:
                raise LimitExceededError("file_too_large", limits.max_file_bytes, len(pdf_bytes))
            occupied_papers = self._occupied_paper_count()
            if occupied_papers >= limits.max_papers:
                raise LimitExceededError("paper_limit_reached", limits.max_papers, occupied_papers)
            occupied_chunks = self._occupied_chunk_count()
            if occupied_chunks >= limits.max_total_chunks:
                raise LimitExceededError("corpus_capacity_reached", limits.max_total_chunks, occupied_chunks)

        # Extract text (page ceiling is checked before any page is read)
        pages = self._extract_pdf(pdf_bytes)
        if not pages:
            raise ValueError("Could not extract text from PDF. Is it scanned/image-based?")

        # Chunk
        chunks = self._chunk_text(
            pages,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("Could not extract non-empty text chunks from PDF.")
        if len(chunks) > limits.max_chunks_per_paper:
            raise LimitExceededError("too_many_chunks", limits.max_chunks_per_paper, len(chunks))

        # Corpus capacity is checked and consumed under the same work lock that
        # serializes every mutation, so two uploads cannot both take the last room.
        with self._state_lock:
            occupied_papers = self._occupied_paper_count()
            if occupied_papers >= limits.max_papers:
                raise LimitExceededError("paper_limit_reached", limits.max_papers, occupied_papers)
            occupied_chunks = self._occupied_chunk_count()
            if occupied_chunks + len(chunks) > limits.max_total_chunks:
                raise LimitExceededError(
                    "corpus_capacity_reached", limits.max_total_chunks, occupied_chunks + len(chunks)
                )

        # Publish this record only after the storage operation succeeds.
        paper = {
            "paper_id": paper_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add to vector store
        if self._vectorstore is not None and self._embeddings is not None:
            texts = [c["text"] for c in chunks]
            metadatas = [
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}-{index}",
                    "filename": filename,
                    "page": c["page"],
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                }
                for index, c in enumerate(chunks)
            ]
            ids = [f"{paper_id}-{i}" for i in range(len(chunks))]
            try:
                self._vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            except Exception as error:
                # A backend may have written part of a batch before raising.
                try:
                    self._vectorstore.delete(ids=ids)
                except Exception:
                    with self._state_lock:
                        self._pending_cleanup[paper_id] = ids
                        self._pending_metadata[paper_id] = paper.copy()
                    raise StorageMutationError(
                        f"Indexing and rollback failed for paper '{paper_id}'. "
                        "Corpus queries are blocked until deletion is retried successfully.",
                        paper_id=paper_id,
                    ) from error
                raise StorageMutationError(
                    f"Indexing failed for paper '{paper_id}'; partial vectors were removed.",
                    paper_id=paper_id,
                    cleanup_required=False,
                ) from error
            logger.info("Added %d chunks to the vector store for paper %s", len(chunks), paper_id[:12])
        else:
            logger.info("Demo mode: stored %d chunks in memory for paper %s", len(chunks), paper_id[:12])

        # Publish the lexical corpus and original pages only after storage succeeds.
        # The existing paper/chunk ceilings bound this process-local mirror too.
        for i, chunk in enumerate(chunks):
            chunk.update(paper_id=paper_id, filename=filename, chunk_id=f"{paper_id}-{i}")
        with self._state_lock:
            self.chunks_store.extend(chunks)
            self._page_texts[paper_id] = {page["page"]: page["text"] for page in pages}
            self.papers[paper_id] = paper

        return {
            "paper_id": paper_id,
            "filename": paper["filename"],
            "pages": len(pages),
            "chunks": len(chunks),
        }

    def assert_storage_ready(self) -> None:
        """Reject access to a corpus whose last mutation needs explicit cleanup."""
        with self._state_lock:
            if self._pending_cleanup:
                raise StorageMutationError(
                    "Corpus queries are blocked by an incomplete storage mutation. "
                    "Retry deletion of the affected paper before querying.",
                    paper_id=next(iter(self._pending_cleanup)),
                )

    def query(
        self,
        question: str,
        paper_id: Optional[str] = None,
        top_k: int = 5,
        request_id: Optional[str] = None,
    ) -> dict:
        """
        Query the knowledge base with a question.

        Returns:
            dict with answer, citations, timing, model info and model usage
        """
        limits = settings.resource_limits()
        if not isinstance(question, str):
            raise ValueError("The question must be a string.")
        if len(question) > limits.max_question_chars:
            raise LimitExceededError("question_too_long", limits.max_question_chars, len(question))
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        top_k = min(top_k, limits.max_top_k)

        # ─── Retrieval ───
        retrieval_start = time.time()

        with self._work_lock:
            self.assert_storage_ready()
            self.assert_backend_ready()
            if self._vectorstore is not None:
                search_kwargs = {"k": CANDIDATE_LIMIT}
                if paper_id:
                    search_kwargs["filter"] = {"paper_id": paper_id}

                results = self._vectorstore.similarity_search_with_relevance_scores(
                    question, **search_kwargs
                )
                passages = [
                    {
                        "text": doc.page_content,
                        "page": doc.metadata.get("page"),
                        "paper": doc.metadata.get("filename", "unknown"),
                        "paper_id": doc.metadata.get("paper_id"),
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "score": score,
                    }
                    for doc, score in results
                ]
                # Storage and its lexical mirror are read under the same work lock.
                # Only indexed chunks in the requested scope may participate.
                chunks = [chunk for chunk in self.chunks_store
                          if not paper_id or chunk["paper_id"] == paper_id]
                if chunks:
                    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
                    dense = [(by_id[doc.metadata["chunk_id"]], score)
                             for doc, score in results if doc.metadata.get("chunk_id") in by_id]
                    ranked = hybrid_rank(dense, lexical_rank(question, chunks), top_k)
                    passages = []
                    for chunk, score in ranked:
                        header = f"[Source: {chunk['filename']}, Page {chunk['page']}]\n"
                        allowance = max(0, limits.max_context_chars // top_k - len(header) - 2)
                        page = self._page_texts.get(chunk["paper_id"], {}).get(chunk["page"], "")
                        passages.append({
                            "text": evidence_window(chunk, page, allowance),
                            "preview": chunk["text"][:300],
                            "page": chunk["page"], "paper": chunk["filename"],
                            "paper_id": chunk["paper_id"], "chunk_id": chunk["chunk_id"],
                            "score": score,
                        })
                else:
                    # Preserve compatibility with injected/test stores without a mirror.
                    passages = passages[:top_k]
            else:
                # Demo mode: simple keyword matching
                passages = self._demo_retrieve(question, paper_id, top_k)

        # Citation previews are presentation data, not the generation context.
        citations = [
            {**{key: value for key, value in passage.items() if key != "preview"},
             "text": passage.get("preview", passage["text"][:300])}
            for passage in passages
        ]

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Count unique papers searched
        papers_searched = len(set(c["paper"] for c in citations))

        # ─── Generation ───
        gen_start = time.time()
        model_usage = None

        if self._llm is not None and citations:
            answer, model_used, model_usage = self._generate_bounded(
                question, passages, limits.max_context_chars, request_id
            )
        elif not citations:
            answer = "The retrieved sources contain insufficient evidence to answer this question."
            model_used = "not-invoked"
        else:
            answer = self._demo_generate(question, citations)
            model_used = "demo-mode"

        gen_time = (time.time() - gen_start) * 1000

        return {
            "answer": answer,
            "citations": citations,
            "papers_searched": papers_searched,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": gen_time,
            "model_used": model_used,
            "model_usage": model_usage,
        }

    @staticmethod
    def _build_context(passages: list[dict], max_chars: int) -> str:
        """Join full passages in rank order until the context ceiling is reached."""
        blocks = []
        used = 0
        for passage in passages:
            block = f"[Source: {passage['paper']}, Page {passage.get('page', '?')}]\n{passage['text']}"
            separator = 2 if blocks else 0
            if used + separator + len(block) > max_chars:
                if not blocks:
                    blocks.append(block[:max_chars])
                break
            blocks.append(block)
            used += separator + len(block)
        return "\n\n".join(blocks)

    def _generate_bounded(
        self, question: str, passages: list[dict], max_context_chars: int, request_id: Optional[str]
    ) -> tuple[str, str, dict]:
        """Reserve an attempted call in the ledger, then invoke the bounded client once."""
        with self._state_lock:
            ledger = self._ledger
            llm = self._llm
            bound = self._token_bound
            provider = self._configured_provider
            model = self._configured_model
        if ledger is None:
            # Never call a model whose usage cannot be accounted for.
            raise BackendUnavailableError("budget_not_configured")
        if bound is None:
            raise BackendUnavailableError("token_bound_unavailable")
        context = self._build_context(passages, max_context_chars)
        prompt = (
            "You are a research assistant. Answer the question based ONLY on the "
            "provided context. Cite the source paper and page number for each claim. "
            "If the context doesn't contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        max_output_tokens = settings.LLM_MAX_OUTPUT_TOKENS
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise BackendUnavailableError("invalid_configuration")
        # An explicit model-supported upper bound (never a character heuristic).
        reserved_tokens = bound.reservation(prompt, max_output_tokens)
        # The reservation counts even if the provider fails, times out or hangs.
        call_id = ledger.reserve(str(provider), str(model), reserved_tokens, request_id)
        try:
            response = llm.invoke(prompt)
        except Exception as error:
            self._settle(ledger, call_id, "failed")
            log_safe_error(logger, "generation_failed", error)
            raise GenerationError("The configured model failed to generate an answer.") from error
        try:
            answer = _response_text(response)
        except GenerationError:
            self._settle(ledger, call_id, "failed")
            raise
        input_tokens, output_tokens = _measured_usage(response)
        self._settle(ledger, call_id, "succeeded", input_tokens, output_tokens)
        measured = input_tokens is not None
        if measured and input_tokens + output_tokens > reserved_tokens:
            # The bound was not an upper bound for this model; the ledger charges
            # the larger measured total so later calls see the overshoot.
            logger.warning(
                "[%s] Measured usage %d exceeded the reservation %d (bound=%s)",
                request_id, input_tokens + output_tokens, reserved_tokens, bound.name,
            )
        model_usage = {
            "accounting": "measured" if measured else "reserved",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_charged": (input_tokens + output_tokens) if measured and (input_tokens + output_tokens) > 0
            else reserved_tokens,
            "tokens_reserved": reserved_tokens,
            "context_chars": len(context),
            "reservation_bound": bound.name,
        }
        return answer, str(model), model_usage

    @staticmethod
    def _settle(ledger: ModelCallLedger, call_id: int, status: str,
                input_tokens: Optional[int] = None, output_tokens: Optional[int] = None) -> None:
        """A settlement failure leaves the conservative reservation in place."""
        try:
            ledger.settle(call_id, status, input_tokens, output_tokens)
        except LedgerError as error:
            log_safe_error(logger, "ledger_unavailable", error.__cause__ or error)

    def _demo_retrieve(
        self,
        question: str,
        paper_id: Optional[str],
        top_k: int,
    ) -> list[dict]:
        """Simple keyword-based retrieval for demo mode."""
        question_words = set(question.lower().split())
        scored = []
        with self._state_lock:
            snapshot = list(self.chunks_store)

        for chunk in snapshot:
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(question_words & chunk_words)
            if overlap > 0:
                score = overlap / max(len(question_words), 1)
                scored.append({
                    "text": chunk["text"],
                    "page": chunk.get("page"),
                    "paper": chunk.get("filename", "unknown"),
                    "paper_id": chunk.get("paper_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "score": min(score, 1.0),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _demo_generate(self, question: str, citations: list[dict]) -> str:
        """Generate a demo answer from citations without LLM."""
        if not citations:
            return (
                "I could not find relevant information in the uploaded papers "
                "to answer this question. Try uploading more papers or rephrasing your question."
            )

        top = citations[0]
        answer = (
            f"Based on the uploaded research papers, here is what I found:\n\n"
            f"From '{top['paper']}' (Page {top.get('page', '?')}):\n"
            f"{top['text'][:500]}\n\n"
        )
        if len(citations) > 1:
            answer += (
                f"Additional context from '{citations[1]['paper']}' "
                f"(Page {citations[1].get('page', '?')}) also discusses related content.\n\n"
            )
        answer += (
            "Note: This is a demo response using keyword matching. "
            "Configure an LLM (OpenAI or Ollama) for full RAG-powered answers with reasoning."
        )
        return answer

    def list_papers(self, include_pending: bool = False) -> list[dict]:
        """List ready papers; optionally expose recoverable pending mutations with status."""
        with self._state_lock:
            if not include_pending:
                return [
                    paper.copy() for paper_id, paper in self.papers.items()
                    if paper_id not in self._pending_cleanup
                ]
            rows = {
                paper_id: {**paper, "status": "ready"}
                for paper_id, paper in self.papers.items()
            }
            for paper_id, ids in self._pending_cleanup.items():
                metadata = self.papers.get(paper_id) or self._pending_metadata.get(paper_id)
                if metadata is None:
                    metadata = {
                        "paper_id": paper_id, "filename": None, "pages": None,
                        "chunks": len(ids), "uploaded_at": None,
                    }
                rows[paper_id] = {**metadata, "status": "pending_cleanup"}
            return list(rows.values())

    def delete_paper(self, paper_id: str) -> bool:
        """Remove a paper and its chunks from the knowledge base."""
        with self._work_lock:
            return self._delete_paper(paper_id)

    def _delete_paper(self, paper_id: str) -> bool:
        with self._state_lock:
            if paper_id not in self.papers and paper_id not in self._pending_cleanup:
                return False
            ids_to_delete = self._pending_cleanup.get(paper_id)
            if ids_to_delete is None and paper_id in self.papers:
                ids_to_delete = [
                    f"{paper_id}-{i}" for i in range(self.papers[paper_id]["chunks"])
                ]

        if self._vectorstore is not None:
            # Delete from ChromaDB
            try:
                self._vectorstore.delete(ids=ids_to_delete)
            except Exception as e:
                with self._state_lock:
                    self._pending_cleanup[paper_id] = ids_to_delete
                raise StorageMutationError(
                    f"Deletion failed for paper '{paper_id}'. "
                    "Corpus queries are blocked until deletion is retried successfully.",
                    paper_id=paper_id,
                ) from e
        elif paper_id in self._pending_cleanup:
            raise StorageMutationError(
                "The vector store is unavailable; storage cleanup cannot be confirmed.",
                paper_id=paper_id,
            )

        # Remove from memory
        with self._state_lock:
            self.chunks_store = [
                c for c in self.chunks_store if c.get("paper_id") != paper_id
            ]
            self.papers.pop(paper_id, None)
            self._page_texts.pop(paper_id, None)
            self._pending_cleanup.pop(paper_id, None)
            self._pending_metadata.pop(paper_id, None)
        logger.info("Deleted paper %s", paper_id[:12] if isinstance(paper_id, str) else "unknown")
        return True

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        papers = self.list_papers()
        total_chunks = sum(p["chunks"] for p in papers)
        return {
            "papers_loaded": len(papers),
            "total_chunks": total_chunks,
        }
