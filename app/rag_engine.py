"""
RAG Engine
===========
Core retrieval-augmented generation engine.
Handles PDF extraction, chunking, embedding, vector storage, and LLM query.

Supports:
- OpenAI (GPT-4o-mini, GPT-4o) via API key
- Ollama (llama3, mistral) for local/free inference
- Demo mode (no API key needed) for testing

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

logger = logging.getLogger("rag-api.engine")


class StorageMutationError(RuntimeError):
    """Storage work failed or needs cleanup before the corpus is usable."""

    def __init__(self, message: str, paper_id: Optional[str] = None, cleanup_required: bool = True):
        super().__init__(message)
        self.paper_id = paper_id
        self.cleanup_required = cleanup_required


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
        # Failed storage mutations remain recoverable by retrying deletion.
        # This guard is process-local; durable recovery is a separate milestone.
        self._pending_cleanup: dict[str, list[str]] = {}
        self._mutation_lock = RLock()
        self._vectorstore = None
        self._embeddings = None
        self._llm = None
        self._initialize()

    def _initialize(self):
        """Initialize embedding model and vector store."""
        # In demo mode, skip loading heavy ML models to stay under memory limits
        if settings.LLM_PROVIDER == "demo":
            logger.info("Running in demo mode — skipping embedding model and vector store.")
            self._embeddings = None
            self._vectorstore = None
            return

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            logger.info(f"Initializing embeddings: {settings.EMBEDDING_MODEL}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
            )
            self._vectorstore = Chroma(
                collection_name="research_papers",
                embedding_function=self._embeddings,
                persist_directory=settings.VECTORSTORE_PATH,
            )
            logger.info("Vector store initialized (ChromaDB).")
            self._init_llm()
        except ImportError as e:
            logger.warning(f"LangChain not fully installed: {e}. Running in demo mode.")
            self._embeddings = None
            self._vectorstore = None
        except Exception as e:
            logger.warning(f"Initialization error: {e}. Running in demo mode.")

    def _init_llm(self):
        """Initialize LLM based on provider setting."""
        try:
            if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=settings.LLM_MODEL,
                    temperature=0.1,
                    api_key=settings.OPENAI_API_KEY,
                )
                logger.info(f"LLM initialized: OpenAI {settings.LLM_MODEL}")
            elif settings.LLM_PROVIDER == "ollama":
                from langchain_community.llms import Ollama

                self._llm = Ollama(
                    model=settings.LLM_MODEL,
                    base_url=settings.OLLAMA_URL,
                )
                logger.info(f"LLM initialized: Ollama {settings.LLM_MODEL}")
            else:
                logger.info("No LLM configured — using demo mode for answers.")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}. Using demo mode.")

    def _extract_pdf(self, pdf_bytes: bytes) -> list[dict]:
        """Extract text from PDF, page by page."""
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i + 1, "text": text.strip()})
            return pages
        except ImportError:
            # Fallback to pypdf
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})
            return pages

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

    def ingest_paper(self, pdf_bytes: bytes, filename: str) -> dict:
        """
        Process and index a research paper.

        Identical bytes are idempotent within the loaded registry: retain the
        original filename, upload time and chunks without indexing again.

        Returns:
            dict with paper_id, canonical filename, pages count, chunks count
        """
        with self._mutation_lock:
            return self._ingest_paper(pdf_bytes, filename)

    def _ingest_paper(self, pdf_bytes: bytes, filename: str) -> dict:
        paper_id = hashlib.sha256(pdf_bytes).hexdigest()
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

        # Extract text
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

        # Publish this record only after the storage operation succeeds.
        paper = {
            "paper_id": paper_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add to vector store
        if self._vectorstore and self._embeddings:
            texts = [c["text"] for c in chunks]
            metadatas = [
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}-{index}",
                    "filename": filename,
                    "page": c["page"],
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
                    self._pending_cleanup[paper_id] = ids
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
            logger.info(f"Added {len(chunks)} chunks to vector store for '{filename}'")
        else:
            # Demo mode: store chunks in memory
            for i, c in enumerate(chunks):
                c["paper_id"] = paper_id
                c["filename"] = filename
                c["chunk_id"] = f"{paper_id}-{i}"
            self.chunks_store.extend(chunks)
            logger.info(f"Demo mode: stored {len(chunks)} chunks in memory")

        self.papers[paper_id] = paper
        return {
            "paper_id": paper_id,
            "filename": paper["filename"],
            "pages": len(pages),
            "chunks": len(chunks),
        }

    def assert_storage_ready(self) -> None:
        """Reject access to a corpus whose last mutation needs explicit cleanup."""
        with self._mutation_lock:
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
    ) -> dict:
        """
        Query the knowledge base with a question.

        Returns:
            dict with answer, citations, timing, and model info
        """
        # ─── Retrieval ───
        retrieval_start = time.time()

        with self._mutation_lock:
            self.assert_storage_ready()
            if self._vectorstore:
                search_kwargs = {"k": top_k}
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
            else:
                # Demo mode: simple keyword matching
                passages = self._demo_retrieve(question, paper_id, top_k)

        # Citation previews are presentation data, not the generation context.
        citations = [{**passage, "text": passage["text"][:300]} for passage in passages]

        retrieval_time = (time.time() - retrieval_start) * 1000

        # Count unique papers searched
        papers_searched = len(set(c["paper"] for c in citations))

        # ─── Generation ───
        gen_start = time.time()

        if self._llm and citations:
            context = "\n\n".join(
                f"[Source: {c['paper']}, Page {c.get('page', '?')}]\n{c['text']}"
                for c in passages
            )
            prompt = (
                "You are a research assistant. Answer the question based ONLY on the "
                "provided context. Cite the source paper and page number for each claim. "
                "If the context doesn't contain enough information, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            try:
                response = self._llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                model_used = settings.LLM_MODEL
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = self._demo_generate(question, citations)
                model_used = "demo-fallback"
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
        }

    def _demo_retrieve(
        self,
        question: str,
        paper_id: Optional[str],
        top_k: int,
    ) -> list[dict]:
        """Simple keyword-based retrieval for demo mode."""
        question_words = set(question.lower().split())
        scored = []

        for chunk in self.chunks_store:
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

    def list_papers(self) -> list[dict]:
        """List all papers in the knowledge base."""
        with self._mutation_lock:
            return [
                paper.copy() for paper_id, paper in self.papers.items()
                if paper_id not in self._pending_cleanup
            ]

    def delete_paper(self, paper_id: str) -> bool:
        """Remove a paper and its chunks from the knowledge base."""
        with self._mutation_lock:
            return self._delete_paper(paper_id)

    def _delete_paper(self, paper_id: str) -> bool:
        if paper_id not in self.papers and paper_id not in self._pending_cleanup:
            return False

        if self._vectorstore:
            # Delete from ChromaDB
            ids_to_delete = self._pending_cleanup.get(paper_id)
            if ids_to_delete is None:
                ids_to_delete = [
                    f"{paper_id}-{i}" for i in range(self.papers[paper_id]["chunks"])
                ]
            try:
                self._vectorstore.delete(ids=ids_to_delete)
            except Exception as e:
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
        self.chunks_store = [
            c for c in self.chunks_store if c.get("paper_id") != paper_id
        ]
        self.papers.pop(paper_id, None)
        self._pending_cleanup.pop(paper_id, None)
        logger.info(f"Deleted paper: {paper_id}")
        return True

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        papers = self.list_papers()
        total_chunks = sum(p["chunks"] for p in papers)
        return {
            "papers_loaded": len(papers),
            "total_chunks": total_chunks,
        }
