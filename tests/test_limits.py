"""Every ceiling rejects before expensive work; duplicates stay idempotent at capacity."""

import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.config import BudgetAllowances, settings
from app.ledger import ModelCallLedger
from app.protection import AdmissionSlot
from app.rag_engine import LimitExceededError, RAGEngine
from tests.conftest import AUTH_HEADERS, minimal_pdf, multi_page_pdf
from tests.test_rag_foundation import FakeVectorStore


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    return RAGEngine()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(api, "rag", RAGEngine())
    monkeypatch.setattr(api, "mutation_slot", AdmissionSlot(1, "mutation"))
    monkeypatch.setattr(api, "query_slot", AdmissionSlot(2, "query"))
    return TestClient(api.app, headers=AUTH_HEADERS)


# ─── Engine-level ceilings ───

def test_page_ceiling_rejects_before_any_page_text_is_extracted(engine, monkeypatch):
    pdfplumber = pytest.importorskip("pdfplumber")
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 3)

    class Page:
        def extract_text(self):
            pytest.fail("Text extraction must not run past the page ceiling")

    class FakePDF:
        pages = [Page() for _ in range(4)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(pdfplumber, "open", lambda *_: FakePDF())
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"four pages", "four.pdf")
    assert (error.value.category, error.value.limit, error.value.observed) == ("too_many_pages", 3, 4)
    assert engine.list_papers() == [] and engine.chunks_store == []


def test_real_pdf_over_the_page_ceiling_is_rejected_and_within_it_is_accepted(engine, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 2)
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(multi_page_pdf(3), "three.pdf")
    assert error.value.category == "too_many_pages"
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 3)
    assert engine.ingest_paper(multi_page_pdf(3), "three.pdf")["pages"] == 3


def test_chunk_ceiling_rejects_before_embeddings_or_storage(engine, monkeypatch):
    store = FakeVectorStore()
    engine._vectorstore = store
    engine._embeddings = object()
    monkeypatch.setattr(settings, "MAX_CHUNKS_PER_PAPER", 5)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "evidence " * 2000}])
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"many chunks", "many.pdf")
    assert error.value.category == "too_many_chunks" and error.value.limit == 5
    assert store.add_calls == 0 and engine.list_papers() == []


def test_paper_ceiling_rejects_before_extraction_and_keeps_duplicates_idempotent(engine, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAPERS", 1)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "Evidence"}])
    first = engine.ingest_paper(b"first", "first.pdf")
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: pytest.fail("No extraction when the corpus is full"))
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"second", "second.pdf")
    assert error.value.category == "paper_limit_reached" and error.value.limit == 1
    assert engine.ingest_paper(b"first", "renamed.pdf") == first
    assert engine.get_stats() == {"papers_loaded": 1, "total_chunks": 1}


def test_pending_cleanup_rows_occupy_capacity(engine, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAPERS", 2)
    monkeypatch.setattr(settings, "MAX_TOTAL_CHUNKS", 3)
    engine._pending_cleanup["ghost"] = ["ghost-0", "ghost-1", "ghost-2"]
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: pytest.fail("No extraction without room"))
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"new", "new.pdf")
    assert error.value.category == "corpus_capacity_reached"
    engine._pending_cleanup["other"] = ["other-0"]
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"new", "new.pdf")
    assert error.value.category == "paper_limit_reached"


def test_corpus_chunk_capacity_is_checked_before_storage_and_frees_after_deletion(engine, monkeypatch):
    store = FakeVectorStore()
    engine._vectorstore = store
    engine._embeddings = object()
    monkeypatch.setattr(settings, "MAX_TOTAL_CHUNKS", 3)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "evidence " * 100}])
    first = engine.ingest_paper(b"first", "first.pdf")
    assert first["chunks"] == 2
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"second", "second.pdf")
    assert error.value.category == "corpus_capacity_reached" and error.value.limit == 3
    assert store.add_calls == 1 and len(store.rows) == 2
    assert engine.ingest_paper(b"first", "again.pdf") == first
    assert engine.delete_paper(first["paper_id"]) is True
    assert engine.ingest_paper(b"second", "second.pdf")["chunks"] == 2


def test_file_size_ceiling_is_checked_before_parsing(engine, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: pytest.fail("No parsing of an oversized file"))
    with pytest.raises(LimitExceededError) as error:
        engine.ingest_paper(b"x" * (1024 * 1024 + 1), "big.pdf")
    assert error.value.category == "file_too_large" and error.value.limit == 1024 * 1024


def test_malformed_pdf_is_a_categorical_value_error(engine):
    with pytest.raises(ValueError) as error:
        engine.ingest_paper(b"%PDF-1.4 this is not really a pdf document", "bad.pdf")
    assert "secret" not in str(error.value)
    assert engine.list_papers() == []


def test_question_length_and_top_k_are_bounded_in_the_engine(engine, monkeypatch):
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "evidence " * 800}])
    assert engine.ingest_paper(b"paper", "paper.pdf")["chunks"] > 5
    with pytest.raises(LimitExceededError) as error:
        engine.query("evidence " * 300)
    assert error.value.category == "question_too_long" and error.value.limit == 2000
    assert len(engine.query("evidence", top_k=50)["citations"]) == 5
    with pytest.raises(ValueError):
        engine.query("evidence", top_k=0)
    with pytest.raises(ValueError):
        engine.query(["not", "a", "string"])


def test_generation_context_is_capped_in_rank_order():
    passages = [{"paper": "paper.pdf", "page": index, "text": f"P{index}-" + "x" * 96} for index in range(5)]
    block = len("[Source: paper.pdf, Page 0]\n") + 100
    context = RAGEngine._build_context(passages, block * 2 + 2)
    assert context.count("[Source:") == 2 and "P0-" in context and "P1-" in context and "P2-" not in context
    assert len(context) <= block * 2 + 2
    truncated = RAGEngine._build_context(passages, 40)
    assert len(truncated) == 40 and truncated.startswith("[Source: paper.pdf, Page 0]")
    assert RAGEngine._build_context([], 1000) == ""


def test_prompt_context_honors_the_configured_ceiling(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 2, "text": "evidence " * 800}])
    engine.ingest_paper(b"paper", "paper.pdf")
    engine._ledger = ModelCallLedger(str(tmp_path / "ledger.sqlite3"), BudgetAllowances(5, 5, 100000, 100000))
    prompts = []

    class FakeModel:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="bounded answer")

    engine._llm = FakeModel()
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 700)
    result = engine.query("evidence", top_k=5)
    context = prompts[0].split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert len(context) <= 700
    assert result["model_usage"]["context_chars"] == len(context)
    assert len(result["citations"]) == 5


# ─── API mapping ───

@pytest.mark.parametrize("category,status", [
    ("file_too_large", 413), ("too_many_pages", 413), ("too_many_chunks", 413),
    ("paper_limit_reached", 409), ("corpus_capacity_reached", 409),
])
def test_upload_limit_rejections_are_categorical(client, monkeypatch, category, status):
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: (_ for _ in ()).throw(LimitExceededError(category, 7, 9)))
    response = client.post("/papers/upload", files={"file": ("a.pdf", io.BytesIO(minimal_pdf()), "application/pdf")})
    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["category"] == category and detail["limit"] == 7 and detail["request_id"]
    assert client.get("/papers").json() == []
    assert not api.mutation_slot.busy


def test_real_upload_over_the_page_ceiling_is_413_and_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 2)
    response = client.post("/papers/upload", files={"file": ("long.pdf", io.BytesIO(multi_page_pdf(3)), "application/pdf")})
    assert response.status_code == 413
    assert response.json()["detail"]["category"] == "too_many_pages"
    assert client.get("/papers").json() == []
    assert client.get("/health").json()["total_chunks"] == 0


def test_upload_file_part_over_the_ceiling_is_413_while_reading(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: pytest.fail("An oversized file must not be ingested"))
    oversized = b"%PDF-1.4\n" + b"x" * (1024 * 1024 + 1)
    response = client.post("/papers/upload", files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")})
    assert response.status_code == 413
    assert response.json()["detail"]["category"] in {"file_too_large", "request_too_large"}
    empty = client.post("/papers/upload", files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")})
    assert empty.status_code == 400 and empty.json()["detail"]["category"] == "empty_file"


def test_query_request_bounds_are_enforced_by_validation(client, monkeypatch):
    monkeypatch.setattr(api.rag, "query", lambda **_: pytest.fail("Out-of-bound requests must not reach the engine"))
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    assert client.post("/query", json={"question": "What is attention?", "top_k": 6}).status_code == 422
    assert client.post("/query", json={"question": "q" * 2001}).status_code == 422
    assert client.post("/query", json={"question": "What?", "paper_id": "p" * 129}).status_code == 422
