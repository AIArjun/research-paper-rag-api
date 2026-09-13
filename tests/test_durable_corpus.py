"""Restart, crash boundaries and fail-closed storage; no real provider calls."""

import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.config import settings
from app.rag_engine import BackendUnavailableError, RAGEngine, StorageMutationError
from tests.test_rag_foundation import FakeVectorStore
from tests.test_readiness_engine import fake_modules, real_configuration


class Crash(BaseException):
    """Abrupt process termination bypasses ordinary exception rollback."""


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PAPER_STORE_PATH", str(tmp_path / "papers.sqlite3"))
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    engines = []

    def start():
        engine = RAGEngine()
        engines.append(engine)
        monkeypatch.setattr(engine, "_extract_pdf", lambda _: [
            {"page": 2, "text": "alpha evidence " * 70},
            {"page": 4, "text": "beta evidence " * 65},
        ])
        return engine

    yield start
    for engine in engines:
        engine.close()


@pytest.fixture
def indexed(corpus, fake_modules, real_configuration, monkeypatch):
    store = FakeVectorStore()
    store._collection = SimpleNamespace(count=lambda: len(store.rows))
    store.get = lambda **_: {
        "ids": list(store.rows),
        "documents": [row[0] for row in store.rows.values()],
        "metadatas": [row[1] for row in store.rows.values()],
    }
    monkeypatch.setattr(fake_modules["langchain_chroma"], "Chroma", lambda **_: store)
    return corpus, store


def test_restart_preserves_bytes_pages_chunks_metadata_and_duplicate_identity(corpus, monkeypatch):
    engine = corpus()
    paper = engine.ingest_paper(b"original PDF bytes", "original.pdf")
    before = (engine.list_papers(), list(engine.chunks_store), dict(engine._page_texts))
    engine.close()
    restored = corpus()
    assert restored.get_readiness()["corpus_storage"] == "persistent"
    assert (restored.list_papers(), restored.chunks_store, restored._page_texts) == before
    assert restored._paper_store.read_pdf(paper["paper_id"]) == b"original PDF bytes"
    monkeypatch.setattr(restored, "_extract_pdf", lambda _: pytest.fail("duplicate reparsed"))
    monkeypatch.setattr(settings, "MAX_PAPERS", 1)
    assert restored.ingest_paper(b"original PDF bytes", "renamed.pdf") == paper
    assert restored.query("alpha", paper_id=paper["paper_id"])["citations"][0]["page"] == 2


@pytest.mark.parametrize("boundary", ["after_stage", "after_vectors", "after_ready"])
def test_interrupted_upload_is_pending_or_committed_never_half_visible(indexed, monkeypatch, boundary):
    start, store = indexed
    engine = start()
    operation = engine._paper_store.stage if boundary == "after_stage" else (
        store.add_texts if boundary == "after_vectors" else engine._paper_store.mark)

    def crash(*args, **kwargs):
        operation(*args, **kwargs)
        raise Crash()

    target, method = (engine._paper_store, "stage") if boundary == "after_stage" else (
        (store, "add_texts") if boundary == "after_vectors" else (engine._paper_store, "mark"))
    with monkeypatch.context() as patch:
        patch.setattr(target, method, crash)
        with pytest.raises(Crash):
            engine.ingest_paper(b"interrupted", "interrupted.pdf")
    engine.close()
    restored = start()
    paper_id = hashlib.sha256(b"interrupted").hexdigest()
    if boundary == "after_ready":
        assert restored.get_readiness()["ready"]
        assert restored.list_papers()[0]["paper_id"] == paper_id
        assert restored.ingest_paper(b"interrupted", "retry.pdf")["filename"] == "interrupted.pdf"
    else:
        assert not restored.get_readiness()["ready"]
        assert restored.list_papers() == []
        assert restored.list_papers(include_pending=True)[0]["status"] == "pending_cleanup"
        with pytest.raises(StorageMutationError):
            restored.query("evidence")
        assert restored.delete_paper(paper_id)
        assert store.rows == {}
        assert restored.get_readiness()["ready"]


@pytest.mark.parametrize("boundary", ["before_vectors", "partial_vectors", "after_vectors", "after_record"])
def test_interrupted_delete_never_resurrects_and_preserves_other_papers(indexed, monkeypatch, boundary):
    start, store = indexed
    engine = start()
    selected = engine.ingest_paper(b"selected", "selected.pdf")
    other = engine.ingest_paper(b"other", "other.pdf")
    original_delete, original_remove = store.delete, engine._paper_store.remove

    def crash_delete(ids):
        if boundary == "partial_vectors":
            original_delete(ids[:1])
        elif boundary == "after_vectors":
            original_delete(ids)
        raise Crash()

    def crash_remove(paper_id):
        original_remove(paper_id)
        raise Crash()

    with monkeypatch.context() as patch:
        if boundary == "after_record":
            patch.setattr(engine._paper_store, "remove", crash_remove)
        else:
            patch.setattr(store, "delete", crash_delete)
        with pytest.raises(Crash):
            engine.delete_paper(selected["paper_id"])
    engine.close()
    restored = start()
    assert [p["paper_id"] for p in restored.list_papers()] == [other["paper_id"]]
    if boundary != "after_record":
        assert restored.get_readiness()["pending_cleanup_ids"] == [selected["paper_id"]]
        assert restored.delete_paper(selected["paper_id"])
    assert restored._paper_store.read_pdf(selected["paper_id"]) is None
    assert restored._paper_store.read_pdf(other["paper_id"]) == b"other"
    restored.close()
    again = start()
    assert again.get_readiness()["ready"]
    assert again.get_stats()["papers_loaded"] == 1
    assert not any(key.startswith(selected["paper_id"]) for key in store.rows)
    assert again._ledger.summary()["calls_total"] == 0


def test_failed_rollback_survives_restart_and_can_be_deleted(indexed):
    start, store = indexed
    engine = start()
    store.fail_add_after = 1
    store.fail_delete_after = 0
    with pytest.raises(StorageMutationError):
        engine.ingest_paper(b"failed", "failed.pdf")
    engine.close()
    restored = start()
    assert not restored.get_readiness()["ready"]
    store.fail_delete_after = None
    paper_id = restored.list_papers(include_pending=True)[0]["paper_id"]
    assert restored.delete_paper(paper_id)
    assert store.rows == {}


@pytest.mark.parametrize("mutation", ["missing", "wrong_text", "wrong_metadata"])
def test_rebuilds_missing_or_inconsistent_ready_vectors_without_reextracting(indexed, mutation):
    start, store = indexed
    engine = start()
    paper = engine.ingest_paper(b"ready", "ready.pdf")
    expected = {key: (row[0], dict(row[1])) for key, row in store.rows.items()}
    engine.close()
    key = next(iter(store.rows))
    if mutation == "missing":
        store.rows.clear()
    elif mutation == "wrong_text":
        store.rows[key] = ("wrong", store.rows[key][1])
    else:
        store.rows[key][1]["page"] = 999
    restored = start()
    assert restored.get_readiness()["ready"]
    assert store.rows == expected
    assert restored.list_papers()[0]["paper_id"] == paper["paper_id"]
    assert restored._ledger.summary()["calls_total"] == 0


def test_unknown_vectors_fail_closed_without_destroying_them(indexed):
    start, store = indexed
    engine = start()
    engine.close()
    store.add_texts(["unrelated"], [{"paper_id": "other"}], ["other-0"])
    restored = start()
    assert restored.get_readiness()["init_error"] == "corpus_recovery_failed"
    assert "other-0" in store.rows
    with pytest.raises(BackendUnavailableError):
        restored.query("unrelated")


def test_failed_repair_stays_unavailable_until_restart_succeeds(indexed):
    start, store = indexed
    engine = start()
    paper = engine.ingest_paper(b"repair", "repair.pdf")
    engine.close()
    store.rows.clear()
    store.fail_add_after = 1
    failed = start()
    assert failed.get_readiness()["init_error"] == "corpus_recovery_failed"
    with pytest.raises(BackendUnavailableError):
        failed.delete_paper(paper["paper_id"])
    failed.close()
    store.fail_add_after = None
    recovered = start()
    assert recovered.get_readiness()["ready"]
    assert len(store.rows) == paper["chunks"]


def test_ready_commit_failure_rolls_back_vectors_and_canonical_data(indexed, monkeypatch):
    start, store = indexed
    engine = start()

    def fail(*_, **__):
        raise sqlite3.OperationalError("failed ready commit")

    monkeypatch.setattr(engine._paper_store, "mark", fail)
    with pytest.raises(StorageMutationError) as error:
        engine.ingest_paper(b"commit failed", "failed.pdf")
    assert error.value.cleanup_required is False
    assert not store.rows and not engine.list_papers()
    engine.close()
    assert start().get_stats()["papers_loaded"] == 0


def test_api_reports_storage_mode_protects_library_and_closes_owner(corpus, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as api
    from tests.conftest import AUTH_HEADERS

    engine = corpus()
    paper = engine.ingest_paper(b"api bytes", "api.pdf")
    monkeypatch.setattr(api, "rag", engine)
    with TestClient(api.app) as client:
        assert client.get("/papers").status_code == 401
        health = client.get("/health")
        assert health.json()["corpus_storage"] == "persistent"
        assert settings.PAPER_STORE_PATH not in health.text
        assert client.get("/papers", headers=AUTH_HEADERS).json()[0]["paper_id"] == paper["paper_id"]
    assert corpus().get_readiness()["ready"]


@pytest.mark.parametrize("operation", ["stage", "mark", "remove"])
def test_database_write_failure_does_not_report_success(corpus, monkeypatch, operation):
    engine = corpus()
    paper = engine.ingest_paper(b"existing", "existing.pdf")

    def fail(*_, **__):
        raise sqlite3.OperationalError("disk full private detail")

    with monkeypatch.context() as patch:
        patch.setattr(engine._paper_store, operation, fail)
        with pytest.raises(StorageMutationError):
            if operation == "stage":
                engine.ingest_paper(b"new", "new.pdf")
            else:
                engine.delete_paper(paper["paper_id"])
    assert not engine.get_readiness()["ready"]
    engine.close()
    restored = corpus()
    if operation == "remove":
        assert restored.list_papers() == []
        assert restored.delete_paper(paper["paper_id"])
    else:
        assert restored.get_readiness()["ready"]
        assert restored.list_papers()[0]["paper_id"] == paper["paper_id"]


def test_second_process_owner_is_refused_and_close_releases_lock(corpus):
    first = corpus()
    second = corpus()
    assert second.get_readiness()["init_error"] == "corpus_unavailable"
    assert first.get_readiness()["ready"]
    first.close()
    assert corpus().get_readiness()["ready"]


@pytest.mark.parametrize("change", ["chunking", "corrupt_pdf", "capacity", "broken_database"])
def test_invalid_saved_state_fails_closed(corpus, monkeypatch, change):
    engine = corpus()
    engine.ingest_paper(b"one", "one.pdf")
    engine.ingest_paper(b"two", "two.pdf")
    engine.close()
    if change == "chunking":
        monkeypatch.setattr(settings, "CHUNK_SIZE", 300)
    elif change == "capacity":
        monkeypatch.setattr(settings, "MAX_PAPERS", 1)
    elif change == "broken_database":
        Path(settings.PAPER_STORE_PATH).write_bytes(b"corrupted")
    else:
        with sqlite3.connect(settings.PAPER_STORE_PATH) as conn:
            conn.execute("UPDATE papers SET pdf=?", (b"changed bytes",))
    restored = corpus()
    assert not restored.get_readiness()["ready"]
    assert restored.get_readiness()["corpus_storage"] == "unavailable"
    assert restored.list_papers() == []


def test_real_process_death_releases_lock_and_retains_upload_journal(tmp_path):
    environment = {**os.environ, "LLM_PROVIDER": "demo", "OPENAI_API_KEY": "",
                   "PAPER_STORE_PATH": str(tmp_path / "crash.sqlite3")}
    crash = subprocess.run([sys.executable, "-c", """
import os
from app.rag_engine import RAGEngine
engine = RAGEngine()
assert engine.get_readiness()['ready']
engine._extract_pdf = lambda _: [{'page': 1, 'text': 'saved evidence'}]
original = engine._paper_store.stage
def stage(*args):
    original(*args)
    os._exit(73)
engine._paper_store.stage = stage
engine.ingest_paper(b'crash fixture', 'crash.pdf')
"""], env=environment, capture_output=True, timeout=15)
    assert crash.returncode == 73, crash.stderr.decode()
    restore = subprocess.run([sys.executable, "-c", """
from app.rag_engine import RAGEngine
engine = RAGEngine()
assert not engine.get_readiness()['ready']
paper = engine.list_papers(include_pending=True)[0]
assert paper['status'] == 'pending_cleanup'
assert engine.delete_paper(paper['paper_id'])
assert engine.get_readiness()['ready']
engine.close()
engine = RAGEngine()
assert engine.list_papers() == [] and engine.get_readiness()['ready']
engine.close()
print('crash recovery passed')
"""], env=environment, capture_output=True, timeout=15)
    assert restore.returncode == 0, restore.stderr.decode()
    assert b"crash recovery passed" in restore.stdout
