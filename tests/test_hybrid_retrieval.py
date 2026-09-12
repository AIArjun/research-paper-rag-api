"""Retrieval scope, evidence boundaries and index lifecycle, without paid calls."""

import copy

import pytest

from app.config import settings
from app.rag_engine import RAGEngine, StorageMutationError
from app.retrieval import evidence_window, hybrid_rank, lexical_rank
from tests.test_rag_foundation import FakeVectorStore


def test_lexical_ranking_recovers_exact_identifier_and_handles_empty_query():
    chunks = [
        {"chunk_id": "generic", "text": "The system retrieves and generates answers."},
        {"chunk_id": "specific", "text": "The document encoder uses BERT; the generator is BART-large."},
    ]
    assert lexical_rank("Which generator is BART-large?", chunks)[0][0]["chunk_id"] == "specific"
    assert lexical_rank("the and", chunks) == []
    assert lexical_rank("question", []) == []


def test_fusion_can_rescue_lexical_only_evidence_and_deduplicates_ids():
    dense = [({"chunk_id": f"d{i}"}, 1.0) for i in range(20)]
    exact = {"chunk_id": "exact"}
    hits = hybrid_rank(dense, [(exact, 10), (exact, 8)], 5)
    assert "exact" in [chunk["chunk_id"] for chunk, _ in hits]
    assert len({chunk["chunk_id"] for chunk, _ in hits}) == 5
    assert dict((c["chunk_id"], s) for c, s in hits)["exact"] == 1 / 61


@pytest.mark.parametrize("start,end,cap", [(0, 5, 12), (20, 25, 12), (45, 50, 12), (20, 25, 90)])
def test_window_is_exact_page_substring_and_keeps_seed(start, end, cap):
    page = "0123456789" * 5
    chunk = {"text": page[start:end], "char_start": start, "char_end": end}
    text = evidence_window(chunk, page, cap)
    assert text in page and chunk["text"] in text
    assert len(text) <= cap
    assert evidence_window(chunk, page, 0) == ""


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    engine = RAGEngine()
    engine._embeddings = object()
    engine._vectorstore = FakeVectorStore()
    return engine


def test_query_keeps_scope_page_preview_and_context_budget(engine, monkeypatch):
    pages = [
        {"page": 3, "text": "intro " * 100 + "The retriever is DPR. " + "BART-large is the generator. " * 80},
        {"page": 8, "text": "Unrelated page containing private material. " * 30},
    ]
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: pages)
    selected = engine.ingest_paper(b"one", "selected.pdf")
    other = engine.ingest_paper(b"two", "other.pdf")
    contexts = []
    engine._llm = object()

    def capture(question, passages, cap, request_id):
        contexts.append(RAGEngine._build_context(passages, cap))
        return "test model", "test", None

    monkeypatch.setattr(engine, "_generate_bounded", capture)
    result = engine.query("Which retriever and generator are used?", paper_id=selected["paper_id"])
    assert 0 < len(contexts[0]) <= settings.MAX_CONTEXT_CHARS
    assert "BART-large" in contexts[0] and "other.pdf" not in contexts[0]
    assert all(c["paper_id"] == selected["paper_id"] for c in result["citations"])
    for citation in result["citations"]:
        chunk = next(c for c in engine.chunks_store if c["chunk_id"] == citation["chunk_id"])
        assert citation["page"] == chunk["page"]
        assert citation["text"] == chunk["text"][:300]
        assert "preview" not in citation
    assert engine.delete_paper(selected["paper_id"])
    assert selected["paper_id"] not in engine._page_texts
    assert all(c["paper_id"] == other["paper_id"] for c in engine.chunks_store)


def test_failed_index_and_delete_preserve_mirror_consistency(engine, monkeypatch):
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 4, "text": "evidence " * 160}])
    paper = engine.ingest_paper(b"existing", "existing.pdf")
    before = copy.deepcopy((engine.chunks_store, engine._page_texts))
    engine._vectorstore.fail_add_after = 1
    with pytest.raises(StorageMutationError):
        engine.ingest_paper(b"failed", "failed.pdf")
    assert (engine.chunks_store, engine._page_texts) == before
    engine._vectorstore.fail_delete_after = 0
    with pytest.raises(StorageMutationError):
        engine.delete_paper(paper["paper_id"])
    assert (engine.chunks_store, engine._page_texts) == before
    with pytest.raises(StorageMutationError):
        engine.query("evidence")
    engine._vectorstore.fail_delete_after = None
    assert engine.delete_paper(paper["paper_id"])
    assert engine.chunks_store == [] and engine._page_texts == {}
