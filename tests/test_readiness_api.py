"""Readiness and failure response contracts; no provider calls or downloads."""

from fastapi.testclient import TestClient
import pytest

import app.main as api
from app.config import settings
from app.rag_engine import RAGEngine, GenerationError
from tests.conftest import AUTH_HEADERS


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(api, "rag", RAGEngine())
    return TestClient(api.app, headers=AUTH_HEADERS)


def test_demo_health_reports_effective_modes(client):
    health = client.get("/health")
    ready = client.get("/ready")
    assert health.status_code == ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["configured_provider"] == "demo"
    assert ready.json()["effective_retrieval"] == "memory-keyword"
    assert ready.json()["effective_generation"] == "demo"
    assert ready.json()["provider_connection_verified"] is False
    assert health.json()["llm_provider"] == "demo"


def test_missing_real_key_is_unready_before_empty_corpus_or_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(api, "rag", RAGEngine())
    calls=[]
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *a: calls.append("ingest"))
    monkeypatch.setattr(api.rag, "query", lambda **k: calls.append("query"))
    for path in ["/ready", "/health"]:
        response=client.get(path)
        assert response.status_code == 503
        data=response.json()
        assert data["ready"] is False
        assert data["configured_provider"] == "openai"
        assert data["effective_generation"] == "unavailable"
        assert data["init_error"]
    query=client.post("/query",json={"question":"What is in the paper?"})
    upload=client.post("/papers/upload",files={"file":("a.pdf",b"%PDF-stub","application/pdf")})
    assert query.status_code == upload.status_code == 503
    assert "answer" not in query.json()
    assert calls == []


def test_generation_failure_has_no_successful_fallback_or_internal_error(client, monkeypatch):
    def fail(**kwargs):
        raise GenerationError("upstream-api-secret-and-internal-url")

    monkeypatch.setattr(api.rag,"get_stats",lambda:{"papers_loaded":1,"total_chunks":1})
    monkeypatch.setattr(api.rag,"query",fail)
    response=client.post("/query",json={"question":"What is in the paper?"})
    assert response.status_code == 502
    assert "secret" not in response.text
    assert "demo" not in response.text
    assert response.json()["detail"]["request_id"]
    assert client.get("/ready").status_code == 200


def test_pending_cleanup_is_visible_and_affects_health_without_losing_recovery(client, monkeypatch):
    paper={"paper_id":"pending-paper","filename":"paper.pdf","pages":1,"chunks":2,
           "uploaded_at":"2026-09-10T00:00:00+00:00"}
    api.rag.papers[paper["paper_id"]]=paper
    api.rag._pending_cleanup[paper["paper_id"]]=["pending-paper-0","pending-paper-1"]
    response=client.get("/ready")
    assert response.status_code == 503
    assert response.json()["pending_cleanup_ids"] == ["pending-paper"]
    assert client.get("/health").status_code == 503
    listed=client.get("/papers")
    assert listed.status_code == 200
    assert listed.json()[0]["status"] == "pending_cleanup"
    assert listed.json()[0]["filename"] == "paper.pdf"
    assert api.rag.get_stats()["papers_loaded"] == 0

    class CleanupStore:
        def delete(self,ids):
            assert ids == ["pending-paper-0","pending-paper-1"]

    api.rag._vectorstore=CleanupStore()
    assert client.delete("/papers/pending-paper").status_code == 200
    assert client.get("/ready").status_code == 200


def test_pending_record_without_metadata_is_explicitly_unknown(client):
    api.rag._pending_cleanup["unknown-paper"]=["unknown-paper-0"]
    response=client.get("/papers")
    assert response.status_code == 200
    paper=response.json()[0]
    assert paper["paper_id"] == "unknown-paper"
    assert paper["status"] == "pending_cleanup"
    assert paper["filename"] is None
    assert paper["pages"] is None
    assert paper["uploaded_at"] is None


@pytest.mark.parametrize("field,value", [("LLM_PROVIDER", []), ("LLM_MODEL", None)])
def test_invalid_typed_configuration_still_returns_readiness_json(client, monkeypatch, field, value):
    monkeypatch.setattr(settings, field, value)
    monkeypatch.setattr(api, "rag", RAGEngine())
    for path in ["/ready", "/health"]:
        response = client.get(path)
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["init_error"] == "invalid_configuration"
        assert data["effective_generation"] == "unavailable"
        assert isinstance(data["configured_provider"], str)
        assert isinstance(data["configured_model"], str)
