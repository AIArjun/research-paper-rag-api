"""Bounded foundation regressions; no model downloads, provider calls or real storage."""

import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest

from app.config import BudgetAllowances, settings
from app.ledger import ModelCallLedger
from app.rag_engine import RAGEngine, StorageMutationError
from tests.conftest import CharacterBound


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    return RAGEngine()


class FakeVectorStore:
    """Model partial backend writes and failures without relying on Chroma."""

    def __init__(self):
        self.rows = {}
        self.add_calls = 0
        self.search_calls = 0
        self.delete_attempts = []
        self.fail_add_after = None
        self.fail_delete_after = None

    def add_texts(self, texts, metadatas, ids):
        self.add_calls += 1
        for index, (text, metadata, chunk_id) in enumerate(zip(texts, metadatas, ids)):
            if self.fail_add_after is not None and index >= self.fail_add_after:
                raise OSError("Simulated interrupted indexing")
            self.rows[chunk_id] = (text, metadata.copy())
        return ids

    def delete(self, ids):
        self.delete_attempts.append(list(ids))
        for index, chunk_id in enumerate(ids):
            if self.fail_delete_after is not None and index >= self.fail_delete_after:
                raise OSError("Simulated interrupted deletion")
            self.rows.pop(chunk_id, None)

    def similarity_search_with_relevance_scores(self, question, k, filter=None):
        self.search_calls += 1
        results = []
        for text, metadata in self.rows.values():
            if filter is not None and metadata["paper_id"] != filter["paper_id"]:
                continue
            results.append((SimpleNamespace(page_content=text, metadata=metadata.copy()), 0.9))
        return results[:k]


@pytest.fixture
def vector_engine(engine, monkeypatch):
    store = FakeVectorStore()
    engine._vectorstore = store
    engine._embeddings = object()
    monkeypatch.setattr(
        engine, "_extract_pdf", lambda _: [{"page": 4, "text": "evidence " * 160}]
    )
    return engine, store


def bounded_chunks(engine, pages, size, overlap):
    """Stop a regressed loop before it hangs CI or allocates unbounded chunks."""
    previous_trace = sys.gettrace()
    code = engine._chunk_text.__func__.__code__
    line_events = 0

    def guard(frame, event, arg):
        nonlocal line_events
        if frame.f_code is code and event == "line":
            line_events += 1
            if line_events > 20_000:
                raise AssertionError("Chunk splitting exceeded its bounded execution allowance")
        return guard

    sys.settrace(guard)
    try:
        return engine._chunk_text(pages, chunk_size=size, chunk_overlap=overlap)
    finally:
        sys.settrace(previous_trace)


@pytest.mark.parametrize(
    "text,size,overlap",
    [
        ("A" * 150 + "." + "B" * 1000, 500, 100),
        ("A" * 50 + "." + "B" * 1000, 500, 100),
        ("Sentence. " * 100, 100, 20),
        ("A" * 1400, 500, 100),
        ("x", 1, 0),
        ("abcdefghijk", 5, 4),
        ("", 500, 100),
        ("  \n  ", 500, 100),
    ],
)
def test_chunking_terminates_advances_and_preserves_coverage(engine, text, size, overlap):
    chunks = bounded_chunks(engine, [{"page": 7, "text": text}], size, overlap)
    starts = [chunk["char_start"] for chunk in chunks]
    assert all(right > left for left, right in zip(starts, starts[1:]))
    covered = set()
    for chunk in chunks:
        assert chunk["page"] == 7
        assert 0 <= chunk["char_start"] < chunk["char_end"] <= len(text)
        assert chunk["char_end"] - chunk["char_start"] <= size
        assert chunk["text"] == text[chunk["char_start"]:chunk["char_end"]].strip()
        covered.update(range(chunk["char_start"], chunk["char_end"]))
    assert {index for index, char in enumerate(text) if not char.isspace()} <= covered


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11), (True, 0), (10, 1.5)])
def test_invalid_chunk_settings_rejected_even_for_empty_input(engine, size, overlap):
    with pytest.raises(ValueError, match="Chunk settings"):
        bounded_chunks(engine, [], size, overlap)


def test_blank_pages_do_not_renumber_physical_pages(engine):
    pages = [{"page": 1, "text": "  "}, {"page": 3, "text": "First evidence"}, {"page": 8, "text": "Second evidence"}]
    chunks = bounded_chunks(engine, pages, 500, 100)
    assert [chunk["page"] for chunk in chunks] == [3, 8]


def test_ingestion_honors_configured_chunk_size_and_overlap(engine, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 32)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 4)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 7, "text": "x" * 75}])
    result = engine.ingest_paper(b"configured chunks", "paper.pdf")
    assert result["chunks"] == 3
    assert [chunk["char_start"] for chunk in engine.chunks_store] == [0, 28, 56]
    assert all(len(chunk["text"]) <= 32 and chunk["page"] == 7 for chunk in engine.chunks_store)


def test_invalid_config_does_not_publish_a_paper(engine, monkeypatch):
    monkeypatch.setattr(settings, "CHUNK_SIZE", 10)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 10)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "text"}])
    with pytest.raises(ValueError, match="Chunk settings"):
        engine.ingest_paper(b"invalid configuration", "paper.pdf")
    assert engine.list_papers() == []
    assert engine.chunks_store == []


def test_full_document_identity_distinguishes_same_prefix(engine, monkeypatch):
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "Document text"}])
    first_bytes, second_bytes = b"A" * 1024 + b"first", b"A" * 1024 + b"second"
    first = engine.ingest_paper(first_bytes, "same-name.pdf")
    second = engine.ingest_paper(second_bytes, "same-name.pdf")
    assert first["paper_id"] == hashlib.sha256(first_bytes).hexdigest()
    assert second["paper_id"] == hashlib.sha256(second_bytes).hexdigest()
    assert first["paper_id"] != second["paper_id"]
    assert engine.get_stats() == {"papers_loaded": 2, "total_chunks": 2}


@pytest.mark.parametrize("backend", ["demo", "vector"])
def test_identical_upload_is_idempotent_and_retains_canonical_metadata(vector_engine, monkeypatch, backend):
    engine, store = vector_engine
    if backend == "demo":
        engine._vectorstore = None
        engine._embeddings = None
    original = engine.ingest_paper(b"identical bytes", "original.pdf")
    metadata = copy.deepcopy(engine.papers)
    chunks = copy.deepcopy(engine.chunks_store)

    def unexpected_extraction(_):
        pytest.fail("An identical duplicate must not be extracted or indexed again")

    monkeypatch.setattr(engine, "_extract_pdf", unexpected_extraction)
    duplicate = engine.ingest_paper(b"identical bytes", "renamed.pdf")
    assert duplicate == original
    assert duplicate["filename"] == "original.pdf"
    assert engine.papers == metadata
    assert engine.chunks_store == chunks
    assert store.add_calls == (1 if backend == "vector" else 0)


@pytest.mark.parametrize("writes_before_error", [0, 1])
def test_index_failure_rolls_back_partial_rows_without_registering(vector_engine, writes_before_error):
    engine, store = vector_engine
    existing = engine.ingest_paper(b"existing", "existing.pdf")
    original_rows = copy.deepcopy(store.rows)
    original_registry = copy.deepcopy(engine.papers)
    store.fail_add_after = writes_before_error
    with pytest.raises(StorageMutationError, match="partial vectors were removed") as failure:
        engine.ingest_paper(b"failed", "failed.pdf")
    assert failure.value.cleanup_required is False
    assert store.rows == original_rows
    assert engine.papers == original_registry
    assert engine._pending_cleanup == {}
    assert engine.query("evidence", paper_id=existing["paper_id"])["citations"]


def test_failed_rollback_blocks_queries_until_cleanup_retry(vector_engine):
    engine, store = vector_engine
    existing = engine.ingest_paper(b"existing", "existing.pdf")
    original_rows = copy.deepcopy(store.rows)
    store.fail_add_after = 1
    store.fail_delete_after = 0
    failed_bytes = b"failed rollback"
    failed_id = hashlib.sha256(failed_bytes).hexdigest()
    with pytest.raises(StorageMutationError, match="rollback failed") as failed_upload:
        engine.ingest_paper(failed_bytes, "failed.pdf")
    assert failed_upload.value.cleanup_required is True
    assert failed_upload.value.paper_id == failed_id
    assert failed_id not in engine.papers
    assert [paper["paper_id"] for paper in engine.list_papers()] == [existing["paper_id"]]
    assert engine.get_stats()["papers_loaded"] == 1
    assert failed_id in engine._pending_cleanup
    with pytest.raises(StorageMutationError, match="blocked") as blocked_query:
        engine.query("evidence")
    assert blocked_query.value.paper_id == failed_id
    with pytest.raises(StorageMutationError) as blocked_storage:
        engine.assert_storage_ready()
    assert blocked_storage.value.paper_id == failed_id
    assert store.search_calls == 0
    with pytest.raises(StorageMutationError, match="needs storage cleanup"):
        engine.ingest_paper(failed_bytes, "retry.pdf")
    store.fail_delete_after = None
    assert engine.delete_paper(failed_id) is True
    assert store.rows == original_rows
    assert engine._pending_cleanup == {}
    assert engine.query("evidence")["citations"]
    store.fail_add_after = None
    assert engine.ingest_paper(failed_bytes, "retry.pdf")["paper_id"] == failed_id


def test_partial_delete_failure_preserves_recovery_and_can_be_retried(vector_engine):
    engine, store = vector_engine
    selected = engine.ingest_paper(b"selected", "selected.pdf")
    other = engine.ingest_paper(b"other", "other.pdf")
    selected_id = selected["paper_id"]
    metadata = engine.papers[selected_id].copy()
    store.fail_delete_after = 1
    with pytest.raises(StorageMutationError, match="Deletion failed") as failed_delete:
        engine.delete_paper(selected_id)
    assert failed_delete.value.paper_id == selected_id
    assert engine.papers[selected_id] == metadata
    assert [paper["paper_id"] for paper in engine.list_papers()] == [other["paper_id"]]
    assert engine.get_stats()["papers_loaded"] == 1
    with pytest.raises(StorageMutationError, match="blocked"):
        engine.query("evidence", paper_id=other["paper_id"])
    store.fail_delete_after = None
    assert engine.delete_paper(selected_id) is True
    assert selected_id not in engine.papers
    assert not any(chunk_id.startswith(selected_id) for chunk_id in store.rows)
    assert engine.query("evidence", paper_id=other["paper_id"])["citations"]
    assert engine.delete_paper(selected_id) is False


def test_cleanup_cannot_be_acknowledged_without_the_backend(vector_engine):
    engine, store = vector_engine
    selected = engine.ingest_paper(b"selected", "selected.pdf")
    store.fail_delete_after = 0
    with pytest.raises(StorageMutationError):
        engine.delete_paper(selected["paper_id"])
    engine._vectorstore = None
    with pytest.raises(StorageMutationError, match="cleanup cannot be confirmed"):
        engine.delete_paper(selected["paper_id"])
    assert selected["paper_id"] in engine._pending_cleanup
    assert selected["paper_id"] in engine.papers


@pytest.mark.parametrize("backend", ["demo", "vector"])
def test_generation_receives_full_evidence_but_returns_short_previews(vector_engine, monkeypatch, backend, tmp_path):
    engine, store = vector_engine
    # Any injected model must be accounted for; an unaccounted model is never invoked.
    engine._ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), BudgetAllowances(5, 5, 100000, 100000))
    engine._token_bound = CharacterBound()
    if backend == "demo":
        engine._vectorstore = None
        engine._embeddings = None
    marker = "DECISIVE: the answer is forty-two"
    text = "evidence " + "x" * 340 + " " + marker
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 9, "text": text}])
    uploaded = engine.ingest_paper(b"late evidence", "source.pdf")
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="forty-two" if marker in prompt else "missing evidence")

    engine._llm = FakeModel()
    result = engine.query("evidence", paper_id=uploaded["paper_id"], top_k=1)
    assert result["answer"] == "forty-two"
    assert len(prompts) == 1 and text in prompts[0]
    citation = result["citations"][0]
    assert citation["text"] == text[:300]
    assert marker not in citation["text"]
    assert citation["paper"] == "source.pdf" and citation["page"] == 9
    assert citation["paper_id"] == uploaded["paper_id"]
    assert citation["chunk_id"] == f"{uploaded['paper_id']}-0"
    assert "full_text" not in citation
