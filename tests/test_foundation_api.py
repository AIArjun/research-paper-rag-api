"""API contracts for foundation failures and idempotent uploads."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, rag
from app.rag_engine import StorageMutationError


@pytest.fixture
def client():
    return TestClient(app)


def test_duplicate_upload_returns_canonical_filename(client, monkeypatch):
    monkeypatch.setattr(rag, "ingest_paper", lambda *_: {
        "paper_id": "same-id", "filename": "original.pdf", "pages": 1, "chunks": 3,
    })
    response = client.post("/papers/upload", files={"file": ("renamed.pdf", b"%PDF-stub", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["filename"] == "original.pdf"
    assert "original.pdf" in response.json()["message"]


@pytest.mark.parametrize("operation", ["upload", "query", "delete"])
def test_storage_failure_is_503_without_backend_details(client, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise StorageMutationError("backend-secret-internal-location", paper_id="recover-this-id")

    if operation == "upload":
        monkeypatch.setattr(rag, "ingest_paper", fail)
        response = client.post("/papers/upload", files={"file": ("a.pdf", b"%PDF-stub", "application/pdf")})
    elif operation == "query":
        monkeypatch.setattr(rag, "get_stats", lambda: {"total_chunks": 1, "papers_loaded": 1})
        monkeypatch.setattr(rag, "query", fail)
        response = client.post("/query", json={"question": "What is the result?"})
    else:
        monkeypatch.setattr(rag, "delete_paper", fail)
        response = client.delete("/papers/test-id")
    assert response.status_code == 503
    assert "backend-secret" not in response.text
    assert response.json()["detail"]["paper_id"] == "recover-this-id"


def test_pending_cleanup_is_explicit_even_when_no_papers_are_loaded(client, monkeypatch):
    monkeypatch.setattr(rag, "_pending_cleanup", {"failed-upload": ["failed-upload-0"]})
    monkeypatch.setattr(rag, "papers", {})
    response = client.post("/query", json={"question": "What is the result?"})
    assert response.status_code == 503
    assert response.json()["detail"]["paper_id"] == "failed-upload"


def test_successful_rollback_directs_client_to_retry_upload(client, monkeypatch):
    def fail(*args, **kwargs):
        raise StorageMutationError("internal", paper_id="cleaned-up", cleanup_required=False)

    monkeypatch.setattr(rag, "ingest_paper", fail)
    response = client.post("/papers/upload", files={"file": ("a.pdf", b"%PDF-stub", "application/pdf")})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["cleanup_required"] is False
    assert "Retry the upload" in detail["message"]
    assert "deletion" not in detail["message"]


def test_query_preserves_stable_citation_identifiers(client, monkeypatch):
    monkeypatch.setattr(rag, "get_stats", lambda: {"total_chunks": 1, "papers_loaded": 1})
    monkeypatch.setattr(rag, "query", lambda **_: {
        "answer": "supported", "citations": [{"text": "passage", "paper": "a.pdf", "page": 3,
            "score": 0.8, "paper_id": "full-id", "chunk_id": "full-id-0"}],
        "papers_searched": 1, "retrieval_time_ms": 0, "generation_time_ms": 0, "model_used": "fake",
    })
    response = client.post("/query", json={"question": "What is the result?"})
    assert response.status_code == 200
    assert response.json()["citations"][0]["paper_id"] == "full-id"
    assert response.json()["citations"][0]["chunk_id"] == "full-id-0"
