"""Admission slots, cancellation safety and responsiveness under slow fake work."""

import asyncio
import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.config import settings
from app.protection import Admission, AdmissionSlot
from app.rag_engine import RAGEngine
from tests.conftest import AUTH_HEADERS, TEST_ACCESS_TOKEN, minimal_pdf
from tests.test_protection_api import _scope

AUTH_ASGI = [(b"authorization", ("Bearer " + TEST_ACCESS_TOKEN).encode("ascii"))]


def _multipart_upload(pdf: bytes, boundary: bytes = b"rag-test-boundary") -> bytes:
    return (
        b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"paper.pdf\"\r\n"
        b"Content-Type: application/pdf\r\n\r\n" + pdf + b"\r\n--" + boundary + b"--\r\n"
    )


def _upload_headers(body: bytes, authenticated: bool = True):
    headers = [(b"content-type", b"multipart/form-data; boundary=rag-test-boundary"),
               (b"content-length", str(len(body)).encode("ascii"))]
    return (AUTH_ASGI if authenticated else []) + headers


def _fresh_demo(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(api, "rag", RAGEngine())
    monkeypatch.setattr(api, "mutation_slot", AdmissionSlot(1, "mutation"))
    monkeypatch.setattr(api, "query_slot", AdmissionSlot(2, "query"))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(api, "rag", RAGEngine())
    monkeypatch.setattr(api, "mutation_slot", AdmissionSlot(1, "mutation"))
    monkeypatch.setattr(api, "query_slot", AdmissionSlot(2, "query"))
    return TestClient(api.app, headers=AUTH_HEADERS)


def test_slot_capacity_is_enforced_and_release_is_idempotent():
    slot = AdmissionSlot(2)
    first, second = slot.admit(), slot.admit()
    assert first is not None and second is not None
    assert slot.admit() is None and slot.busy and slot.in_use == 2
    first.release()
    first.release()
    assert slot.in_use == 1 and not slot.busy
    second.release()
    assert slot.in_use == 0
    with pytest.raises(ValueError):
        AdmissionSlot(0)


def test_slot_stays_held_until_the_worker_thread_finishes_even_when_cancelled():
    slot = AdmissionSlot(1)
    release_work = threading.Event()
    work_started = threading.Event()
    finished = []

    def slow_work():
        work_started.set()
        release_work.wait(timeout=10)
        finished.append(True)
        return "done"

    async def scenario():
        admission = slot.admit()
        task = asyncio.create_task(admission.run(slow_work))
        await asyncio.get_running_loop().run_in_executor(None, work_started.wait, 5)
        assert admission.state == Admission.RUNNING
        task.cancel()
        await asyncio.sleep(0.2)
        # The thread is still running: the slot must still be held and a new admission refused.
        assert slot.busy and slot.admit() is None
        release_work.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return admission

    admission = asyncio.run(scenario())
    assert finished == [True]
    # The cancelled await returns before the worker's finally block; the thread releases.
    deadline = time.monotonic() + 5
    while admission.state != Admission.RELEASED and time.monotonic() < deadline:
        time.sleep(0.01)
    assert admission.state == Admission.RELEASED
    assert not slot.busy and slot.in_use == 0


def test_admission_abandoned_before_dispatch_releases_once_and_never_runs_work():
    slot = AdmissionSlot(1)
    admission = slot.admit()
    assert slot.busy
    # Simulate the awaiting side being cancelled before the thread picked up the work.
    admission._state = Admission.TRANSFERRED
    admission.release()
    assert not slot.busy and admission.state == Admission.RELEASED
    ran = []
    assert admission._guarded(lambda: ran.append(True)) is None
    assert ran == [] and slot.in_use == 0
    with pytest.raises(RuntimeError):
        asyncio.run(admission.run(lambda: None))


def test_run_releases_after_normal_completion_and_after_worker_exceptions():
    slot = AdmissionSlot(1)

    async def scenario():
        admission = slot.admit()
        assert await admission.run(lambda value: value * 2, 21) == 42
        assert not slot.busy
        failing = slot.admit()
        with pytest.raises(ValueError):
            await failing.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        return slot.in_use

    assert asyncio.run(scenario()) == 0


def test_health_and_readiness_stay_responsive_during_slow_ingestion(client, monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def slow_extract(_):
        started.set()
        release.wait(timeout=15)
        return [{"page": 1, "text": "Slow evidence about attention mechanisms."}]

    monkeypatch.setattr(api.rag, "_extract_pdf", slow_extract)
    pdf = minimal_pdf()
    outcomes = {}

    def upload():
        outcomes["upload"] = client.post(
            "/papers/upload", files={"file": ("slow.pdf", io.BytesIO(pdf), "application/pdf")}
        )

    worker = threading.Thread(target=upload, daemon=True)
    worker.start()
    assert started.wait(timeout=10)
    try:
        probe_client = TestClient(api.app)
        for path in ("/ready", "/health"):
            begun = time.monotonic()
            response = probe_client.get(path)
            assert response.status_code == 200, response.text
            assert time.monotonic() - begun < 2.0
        # A second mutation and a query are refused immediately instead of queueing.
        busy_upload = client.post(
            "/papers/upload", files={"file": ("other.pdf", io.BytesIO(pdf + b"\n%other"), "application/pdf")}
        )
        assert busy_upload.status_code == 429
        assert busy_upload.headers["retry-after"] == "5"
        assert busy_upload.json()["detail"]["category"] == "busy"
        busy_delete = client.delete("/papers/anything")
        assert busy_delete.status_code == 429
        monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
        busy_query = client.post("/query", json={"question": "What is attention?"})
        assert busy_query.status_code == 429
        assert client.get("/papers").status_code == 200
    finally:
        release.set()
        worker.join(timeout=15)
    assert outcomes["upload"].status_code == 200, outcomes["upload"].text
    assert not api.mutation_slot.busy
    monkeypatch.setattr(api.rag, "_extract_pdf", lambda _: [{"page": 1, "text": "quick"}])
    retry = client.post(
        "/papers/upload", files={"file": ("other.pdf", io.BytesIO(pdf + b"\n%other"), "application/pdf")}
    )
    assert retry.status_code == 200


def test_query_concurrency_is_bounded_and_slots_return(client, monkeypatch):
    release = threading.Event()
    started = threading.Barrier(3, timeout=10)

    def slow_query(**kwargs):
        started.wait()
        release.wait(timeout=15)
        return {
            "answer": "slow", "citations": [], "papers_searched": 0,
            "retrieval_time_ms": 0.0, "generation_time_ms": 0.0, "model_used": "demo-mode",
            "model_usage": None,
        }

    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    monkeypatch.setattr(api.rag, "query", slow_query)
    results = []

    def ask():
        results.append(client.post("/query", json={"question": "What is attention?"}).status_code)

    workers = [threading.Thread(target=ask, daemon=True) for _ in range(2)]
    for worker in workers:
        worker.start()
    started.wait()
    try:
        assert api.query_slot.busy
        refused = client.post("/query", json={"question": "What is attention?"})
        assert refused.status_code == 429
        assert refused.json()["detail"]["category"] == "busy"
        assert client.get("/ready").status_code == 200
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=15)
    assert results == [200, 200]
    assert api.query_slot.in_use == 0
    monkeypatch.setattr(api.rag, "query", lambda **kwargs: {
        "answer": "fast", "citations": [], "papers_searched": 0,
        "retrieval_time_ms": 0.0, "generation_time_ms": 0.0, "model_used": "demo-mode", "model_usage": None,
    })
    assert client.post("/query", json={"question": "What is attention?"}).status_code == 200


def test_second_simultaneous_upload_is_refused_before_any_body_is_received(monkeypatch):
    """Review probe: two authorized uploads with held body receivers; only one may proceed."""
    _fresh_demo(monkeypatch)
    seen_in_worker = []

    def extract(_):
        # The admission taken before parsing is the one held here, inside the worker thread.
        seen_in_worker.append((api.mutation_slot.in_use, api.mutation_slot.admit()))
        return [{"page": 1, "text": "Evidence about attention mechanisms."}]

    monkeypatch.setattr(api.rag, "_extract_pdf", extract)
    body = _multipart_upload(minimal_pdf())
    headers = _upload_headers(body)

    async def scenario():
        first_receive_called = asyncio.Event()
        release_first = asyncio.Event()

        async def first_receive():
            first_receive_called.set()
            await release_first.wait()
            return {"type": "http.request", "body": body, "more_body": False}

        first_messages = []

        async def first_send(message):
            first_messages.append(message)

        first = asyncio.create_task(api.app(_scope("POST", "/papers/upload", headers), first_receive, first_send))
        await asyncio.wait_for(first_receive_called.wait(), 10)
        assert api.mutation_slot.in_use == 1

        second_messages = []

        async def second_receive():
            pytest.fail("A refused upload must never read its body")

        async def second_send(message):
            second_messages.append(message)

        await api.app(_scope("POST", "/papers/upload", headers), second_receive, second_send)
        assert second_messages[0]["status"] == 429
        assert dict(second_messages[0]["headers"])[b"retry-after"] == b"5"
        assert json.loads(second_messages[1]["body"])["detail"]["category"] == "busy"
        assert api.mutation_slot.in_use == 1

        release_first.set()
        await asyncio.wait_for(first, 30)
        return first_messages

    messages = asyncio.run(scenario())
    assert messages[0]["status"] == 200, messages
    assert seen_in_worker == [(1, None)]
    assert api.mutation_slot.in_use == 0


def test_parse_failure_oversize_and_disconnect_release_the_upload_admission(monkeypatch):
    _fresh_demo(monkeypatch)
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: pytest.fail("Nothing to ingest on these paths"))

    async def call(headers, receive):
        messages = []

        async def send(message):
            messages.append(message)

        await api.app(_scope("POST", "/papers/upload", headers), receive, send)
        return messages[0]["status"]

    async def scenario():
        garbage = b"this is not multipart"

        async def garbage_receive():
            return {"type": "http.request", "body": garbage, "more_body": False}

        assert await call(_upload_headers(garbage), garbage_receive) == 400
        assert api.mutation_slot.in_use == 0

        async def disconnect():
            return {"type": "http.disconnect"}

        assert await call(_upload_headers(b"x" * 100), disconnect) == 400
        assert api.mutation_slot.in_use == 0

        monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
        oversize = _multipart_upload(b"%PDF" + b"x" * (1024 * 1024 + 4096))

        async def oversize_receive():
            return {"type": "http.request", "body": oversize, "more_body": False}

        assert await call(_upload_headers(oversize), oversize_receive) == 413
        assert api.mutation_slot.in_use == 0

    asyncio.run(scenario())


def test_unauthorized_upload_takes_no_admission(monkeypatch):
    _fresh_demo(monkeypatch)
    admitted = []
    real_admit = api.mutation_slot.admit
    monkeypatch.setattr(api.mutation_slot, "admit", lambda: admitted.append(True) or real_admit())

    async def scenario():
        messages = []

        async def receive():
            pytest.fail("No body is read without a token")

        async def send(message):
            messages.append(message)

        await api.app(_scope("POST", "/papers/upload", _upload_headers(b"x", authenticated=False)), receive, send)
        return messages[0]["status"]

    assert asyncio.run(scenario()) == 401
    assert admitted == [] and api.mutation_slot.in_use == 0


def test_rejected_and_failed_uploads_release_the_mutation_slot(client, monkeypatch):
    pdf = minimal_pdf()
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: (_ for _ in ()).throw(RuntimeError("private detail")))
    failed = client.post("/papers/upload", files={"file": ("a.pdf", io.BytesIO(pdf), "application/pdf")})
    assert failed.status_code == 500 and "private detail" not in failed.text
    assert not api.mutation_slot.busy
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    rejected = client.post(
        "/papers/upload", files={"file": ("big.pdf", io.BytesIO(b"%PDF" + b"x" * (1024 * 1024 + 10)), "application/pdf")}
    )
    assert rejected.status_code == 413
    assert not api.mutation_slot.busy
