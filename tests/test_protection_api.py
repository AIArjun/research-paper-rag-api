"""Access control, request-size and sanitization contracts; no provider calls."""

import asyncio
import http.client
import io
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import app.main as api
from app.config import settings
from app.protection import AdmissionSlot, presented_token_matches
from app.rag_engine import RAGEngine
from tests.conftest import AUTH_HEADERS, TEST_ACCESS_TOKEN, minimal_pdf

PROTECTED = [("GET", "/papers"), ("POST", "/papers/upload"), ("POST", "/query"), ("DELETE", "/papers/some-id")]
AUTH_ASGI = [(b"authorization", ("Bearer " + TEST_ACCESS_TOKEN).encode("ascii"))]
MULTIPART_ASGI = [(b"content-type", b"multipart/form-data; boundary=rag-test-boundary")]
MULTIPART_PREFIX = (
    b"--rag-test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.pdf\"\r\n"
    b"Content-Type: application/pdf\r\n\r\n%PDF-1.4\n"
)


def _multipart_stream(total_bytes: int, chunk_size: int = 65536):
    """A syntactically valid multipart file part that never terminates before total_bytes."""
    chunks = [MULTIPART_PREFIX + b"x" * (chunk_size - len(MULTIPART_PREFIX))]
    while sum(len(chunk) for chunk in chunks) < total_bytes:
        chunks.append(b"x" * chunk_size)
    return chunks


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(api, "rag", RAGEngine())
    monkeypatch.setattr(api, "mutation_slot", AdmissionSlot(1, "mutation"))
    monkeypatch.setattr(api, "query_slot", AdmissionSlot(2, "query"))
    return TestClient(api.app)


def _forbid_backend_work(monkeypatch):
    for name in ("ingest_paper", "query", "delete_paper"):
        monkeypatch.setattr(api.rag, name, lambda *a, **k: pytest.fail("No backend work before authorization"))


def _protected_request(client, method, path, headers=None):
    if path == "/papers/upload":
        return client.post(path, headers=headers,
                           files={"file": ("a.pdf", io.BytesIO(minimal_pdf()), "application/pdf")})
    if path == "/query":
        return client.post(path, headers=headers, json={"question": "What is attention?"})
    return client.request(method, path, headers=headers)


def _scope(method, path, headers):
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": headers, "client": ("testclient", 50000),
        "server": ("testserver", 80), "state": {},
    }


def _asgi_request(method, path, headers, chunks):
    """Drive the full app with a streaming body; returns status, JSON body and bytes consumed."""
    consumed = []

    async def run():
        pending = list(chunks)

        async def receive():
            if pending:
                chunk = pending.pop(0)
                consumed.append(len(chunk))
                return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
            return {"type": "http.request", "body": b"", "more_body": False}

        messages = []

        async def send(message):
            messages.append(message)

        await api.app(_scope(method, path, headers), receive, send)
        return messages

    messages = asyncio.run(run())
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return messages[0]["status"], (json.loads(body) if body else None), consumed


# ─── Access token ───

@pytest.mark.parametrize("method,path", PROTECTED)
def test_missing_token_is_401_with_a_challenge_before_any_work(client, monkeypatch, method, path):
    _forbid_backend_work(monkeypatch)
    response = _protected_request(client, method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="research-paper-rag-demo"'
    assert response.json()["detail"]["category"] == "unauthorized"
    assert len(response.headers["x-request-id"]) == 12
    assert response.json()["detail"]["request_id"] == response.headers["x-request-id"]
    assert TEST_ACCESS_TOKEN not in response.text


@pytest.mark.parametrize("value", [
    "Bearer " + TEST_ACCESS_TOKEN[:-1] + "X",
    "Bearer " + TEST_ACCESS_TOKEN + "x",
    "Bearer " + TEST_ACCESS_TOKEN[:-1],
    "Basic " + TEST_ACCESS_TOKEN,
    "Bearer",
    "Bearer ",
    TEST_ACCESS_TOKEN,
    "Bearer wrong-token-that-is-long-enough-0123456789abcdef",
])
def test_wrong_or_malformed_credentials_are_401_invalid_token(client, monkeypatch, value):
    _forbid_backend_work(monkeypatch)
    response = client.get("/papers", headers={"Authorization": value})
    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]
    assert TEST_ACCESS_TOKEN not in response.text


def test_correct_token_is_accepted_with_a_case_insensitive_scheme(client):
    for scheme in ("Bearer", "bearer", "BEARER"):
        response = client.get("/papers", headers={"Authorization": f"{scheme} {TEST_ACCESS_TOKEN}"})
        assert response.status_code == 200 and response.json() == []


def test_token_comparison_rejects_prefixes_and_unconfigured_tokens():
    presented = b"Bearer " + TEST_ACCESS_TOKEN.encode()
    assert presented_token_matches(presented, TEST_ACCESS_TOKEN)
    assert presented_token_matches(b"  bearer   " + TEST_ACCESS_TOKEN.encode() + b" ", TEST_ACCESS_TOKEN)
    assert not presented_token_matches(b"Bearer " + TEST_ACCESS_TOKEN[:-1].encode(), TEST_ACCESS_TOKEN)
    assert not presented_token_matches(None, TEST_ACCESS_TOKEN)
    assert not presented_token_matches(b"", TEST_ACCESS_TOKEN)
    assert not presented_token_matches(b"Bearer short", "short")
    assert not presented_token_matches(presented, "")


@pytest.mark.parametrize("configured", [
    "", "too-short", "x" * 31, "has whitespace " + "a" * 40, "tab\t" + "a" * 40,
    None, 12345, "ünïcodé-" * 8, "x" * 513,
])
def test_absent_or_invalid_configured_token_is_categorical_503(client, monkeypatch, configured):
    monkeypatch.setattr(settings, "DEMO_ACCESS_TOKEN", configured)
    _forbid_backend_work(monkeypatch)
    attempts = [{}, AUTH_HEADERS]
    if isinstance(configured, str) and configured and configured.isascii():
        attempts.append({"Authorization": f"Bearer {configured}"})
    for headers in attempts:
        response = client.get("/papers", headers=headers)
        assert response.status_code == 503
        assert response.json()["detail"]["category"] == "access_not_configured"
        if isinstance(configured, str) and configured:
            assert configured not in response.text
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["access_configured"] is False and ready.json()["ready"] is False
    assert ready.json()["init_error"] is None
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["status"] == "unready"
    assert client.get("/").status_code == 200


def test_unconfigured_token_wins_over_backend_state_in_real_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "DEMO_ACCESS_TOKEN", "")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(api, "rag", RAGEngine())
    response = client.post("/query", json={"question": "What is attention?"})
    assert response.status_code == 503
    assert response.json()["detail"]["category"] == "access_not_configured"
    assert client.get("/ready").json()["init_error"] == "missing_api_key"


def test_public_endpoints_never_expose_the_token(client):
    for path in ("/", "/docs", "/redoc", "/openapi.json", "/health", "/ready"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert TEST_ACCESS_TOKEN not in response.text
        assert TEST_ACCESS_TOKEN[:20] not in response.text
    openapi = client.get("/openapi.json").json()
    assert "HTTPBearer" in openapi["components"]["securitySchemes"]
    assert client.get("/ready").json()["access_configured"] is True
    assert client.get("/health").json()["access_configured"] is True


def test_unauthorized_upload_never_reads_the_body():
    async def run():
        async def receive():
            pytest.fail("The body must not be read before authorization")

        messages = []

        async def send(message):
            messages.append(message)

        headers = MULTIPART_ASGI + [(b"content-length", b"4096")]
        await api.app(_scope("POST", "/papers/upload", headers), receive, send)
        return messages

    messages = asyncio.run(run())
    assert messages[0]["status"] == 401


# ─── Request body limits ───

def test_upload_over_the_cap_without_content_length_is_413_before_ingestion(monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: pytest.fail("An oversized body must never be ingested"))
    limit = settings.resource_limits().max_upload_request_bytes
    status, payload, consumed = _asgi_request(
        "POST", "/papers/upload", AUTH_ASGI + MULTIPART_ASGI, _multipart_stream(limit + 4 * 65536)
    )
    assert status == 413
    assert payload["detail"]["category"] == "request_too_large"
    assert payload["detail"]["limit_bytes"] == limit
    assert limit < sum(consumed) <= limit + 65536


def test_misleading_content_length_does_not_bypass_the_cap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: pytest.fail("An oversized body must never be ingested"))
    limit = settings.resource_limits().max_upload_request_bytes
    headers = AUTH_ASGI + MULTIPART_ASGI + [(b"content-length", b"100")]
    status, payload, consumed = _asgi_request("POST", "/papers/upload", headers, _multipart_stream(20 * 65536))
    assert status == 413 and payload["detail"]["category"] == "request_too_large"
    assert sum(consumed) <= limit + 65536


def test_declared_content_length_over_the_cap_is_rejected_without_reading():
    limit = settings.resource_limits().max_upload_request_bytes

    async def run():
        async def receive():
            pytest.fail("A body declared over the cap must not be read")

        messages = []

        async def send(message):
            messages.append(message)

        headers = AUTH_ASGI + MULTIPART_ASGI + [(b"content-length", str(limit + 1).encode())]
        await api.app(_scope("POST", "/papers/upload", headers), receive, send)
        return messages

    messages = asyncio.run(run())
    assert messages[0]["status"] == 413


def test_json_routes_have_a_small_body_cap(monkeypatch):
    monkeypatch.setattr(api.rag, "query", lambda **_: pytest.fail("An oversized body must never reach the engine"))
    limit = settings.resource_limits().max_json_body_bytes
    oversized = b'{"question": "' + b"a" * (limit + 10) + b'"}'
    headers = AUTH_ASGI + [(b"content-type", b"application/json")]
    status, payload, _ = _asgi_request("POST", "/query", headers + [(b"content-length", str(len(oversized)).encode())], [oversized])
    assert status == 413 and payload["detail"]["category"] == "request_too_large"
    status, payload, _ = _asgi_request("POST", "/query", headers, [oversized[i:i + 4096] for i in range(0, len(oversized), 4096)])
    assert status == 413 and payload["detail"]["category"] == "request_too_large"
    within = json.dumps({"question": "What is attention?"}).encode()
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 0, "total_chunks": 0})
    status, payload, _ = _asgi_request("POST", "/query", headers, [within])
    assert status == 400 and payload["detail"]["category"] == "empty_corpus"


@pytest.mark.parametrize("declared", [b"abc", b"-1", b"1e3"])
def test_invalid_content_length_is_400_before_parsing(declared):
    headers = AUTH_ASGI + MULTIPART_ASGI + [(b"content-length", declared)]
    status, payload, consumed = _asgi_request("POST", "/papers/upload", headers, [b"x" * 10])
    assert status == 400 and payload["detail"]["category"] == "invalid_request"
    assert consumed == []


@pytest.mark.parametrize("http_implementation", ["h11", "httptools"])
def test_real_server_enforces_the_cap_on_a_chunked_upload(monkeypatch, http_implementation):
    pytest.importorskip(http_implementation)
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: pytest.fail("An oversized body must never be ingested"))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    config = uvicorn.Config(api.app, log_level="warning", lifespan="off", access_log=False,
                            loop="asyncio", http=http_implementation)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    try:
        def chunks():
            yield from _multipart_stream(20 * 65536)  # 1.25 MiB, above 1 MiB plus the allowance

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request("POST", "/papers/upload", body=chunks(), encode_chunked=True, headers={
            "Authorization": "Bearer " + TEST_ACCESS_TOKEN,
            "Content-Type": "multipart/form-data; boundary=rag-test-boundary",
        })
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 413
        assert payload["detail"]["category"] == "request_too_large"
        connection.close()

        unauthorized = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        unauthorized.request("GET", "/papers")
        denied = unauthorized.getresponse()
        denied.read()
        assert denied.status == 401 and denied.getheader("www-authenticate").startswith("Bearer")
        unauthorized.close()
    finally:
        server.should_exit = True
        thread.join(timeout=20)


# ─── CORS ───

def test_cors_defaults_to_no_allowed_origins(client):
    preflight = client.options("/query", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
    })
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers
    simple = client.get("/health", headers={"Origin": "https://evil.example"})
    assert simple.status_code == 200
    assert "access-control-allow-origin" not in simple.headers


def test_an_allowed_origin_does_not_replace_the_token(monkeypatch):
    monkeypatch.setattr(api, "rag", RAGEngine())
    wrapped = CORSMiddleware(
        api.app, allow_origins=["https://allowed.example"], allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"],
    )
    client = TestClient(wrapped)
    preflight = client.options("/papers", headers={
        "Origin": "https://allowed.example", "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://allowed.example"
    denied = client.get("/papers", headers={"Origin": "https://allowed.example"})
    assert denied.status_code == 401
    allowed = client.get("/papers", headers={"Origin": "https://allowed.example", **AUTH_HEADERS})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"


@pytest.mark.parametrize("raw,expected", [
    ("", []),
    ("https://a.example, http://localhost:3000 ,", ["https://a.example", "http://localhost:3000"]),
])
def test_allowed_origins_parse_explicit_entries(monkeypatch, raw, expected):
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", raw)
    assert settings.allowed_origins() == expected


@pytest.mark.parametrize("raw", ["*", "https://*.example", "a.example", "https://a.example/path", "https://a.example b"])
def test_wildcard_or_malformed_origins_fail_closed(monkeypatch, raw):
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", raw)
    with pytest.raises(ValueError):
        settings.allowed_origins()
    assert RAGEngine().get_readiness()["init_error"] == "invalid_configuration"


# ─── Configuration robustness ───

@pytest.mark.parametrize("variable", [
    "MAX_FILE_SIZE_MB", "MAX_PDF_PAGES", "MAX_CHUNKS_PER_PAPER", "MAX_PAPERS", "MAX_TOTAL_CHUNKS",
    "MAX_CONTEXT_CHARS", "MAX_CONCURRENT_QUERIES", "LLM_TIMEOUT_SECONDS", "LLM_MAX_OUTPUT_TOKENS",
    "MAX_MODEL_CALLS_PER_DAY", "MAX_MODEL_TOKENS_TOTAL",
])
def test_malformed_numeric_environment_fails_closed_without_crashing_import(variable):
    environment = {**os.environ, "LLM_PROVIDER": "demo", variable: "ten"}
    check = subprocess.run(
        [sys.executable, "-c", (
            "from fastapi.testclient import TestClient; import app.main as api; "
            "from tests.conftest import AUTH_HEADERS; "
            "client = TestClient(api.app); ready = client.get('/ready'); "
            "assert ready.status_code == 503, ready.text; "
            "assert ready.json()['init_error'] == 'invalid_configuration', ready.text; "
            "query = client.post('/query', headers=AUTH_HEADERS, json={'question': 'What is attention?'}); "
            "assert query.status_code == 503, query.text; "
            "assert query.json()['detail']['category'] == 'invalid_configuration', query.text; "
            "upload = client.post('/papers/upload', headers=AUTH_HEADERS, files={'file': ('a.pdf', b'%PDF', 'application/pdf')}); "
            "assert upload.status_code == 503, upload.text; "
            "assert upload.json()['detail']['category'] == 'invalid_configuration', upload.text; "
            "print('fail-closed')"
        )],
        cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == "fail-closed"


@pytest.mark.parametrize("variable,value", [("MAX_FILE_SIZE_MB", 0), ("MAX_FILE_SIZE_MB", 65), ("MAX_PAPERS", -1),
                                            ("MAX_CONCURRENT_QUERIES", 9), ("MAX_TOTAL_CHUNKS", None)])
def test_out_of_range_limits_are_invalid_configuration(client, monkeypatch, variable, value):
    monkeypatch.setattr(settings, variable, value)
    with pytest.raises(ValueError):
        settings.resource_limits()
    assert client.get("/ready").json()["limits"] is None
    for response in (
        client.post("/query", headers=AUTH_HEADERS, json={"question": "What is attention?"}),
        client.post("/papers/upload", headers=AUTH_HEADERS,
                    files={"file": ("a.pdf", io.BytesIO(minimal_pdf()), "application/pdf")}),
    ):
        assert response.status_code == 503 and response.json()["detail"]["category"] == "invalid_configuration"


def test_readiness_reports_the_effective_limits(client):
    limits = client.get("/ready").json()["limits"]
    assert limits["max_file_bytes"] == 10 * 1024 * 1024
    assert limits["max_upload_request_bytes"] == 10 * 1024 * 1024 + 16 * 1024
    assert limits["max_json_body_bytes"] == 32 * 1024
    assert limits["max_pdf_pages"] == 60 and limits["max_chunks_per_paper"] == 600
    assert limits["max_papers"] == 20 and limits["max_total_chunks"] == 3000
    assert limits["max_question_chars"] == 2000 and limits["max_top_k"] == 5
    assert limits["max_context_chars"] == 6000 and limits["max_concurrent_queries"] == 2
    assert "limits" not in client.get("/health").json()


# ─── Sanitized errors and logs ───

def test_processing_failures_are_categorical_and_logs_hold_no_details(client, monkeypatch, caplog):
    secret = "private-provider-detail-and-filesystem-path"
    monkeypatch.setattr(api.rag, "ingest_paper", lambda *_: (_ for _ in ()).throw(RuntimeError(secret)))
    with caplog.at_level(logging.INFO):
        response = client.post("/papers/upload", headers=AUTH_HEADERS, files={
            "file": ("private-filename-secret.pdf", io.BytesIO(minimal_pdf()), "application/pdf"),
        })
    assert response.status_code == 500
    assert response.json()["detail"] == {
        "message": "Processing failed.", "category": "ingestion_failed",
        "request_id": response.headers["x-request-id"],
    }
    assert secret not in response.text and secret not in caplog.text
    assert "private-filename-secret" not in caplog.text
    assert "category=ingestion_failed" in caplog.text and "exception_type=RuntimeError" in caplog.text
    assert "Traceback" not in caplog.text


def test_successful_uploads_do_not_log_the_filename(client, monkeypatch, caplog):
    monkeypatch.setattr(api.rag, "_extract_pdf", lambda _: [{"page": 1, "text": "Evidence about attention."}])
    with caplog.at_level(logging.INFO):
        response = client.post("/papers/upload", headers=AUTH_HEADERS, files={
            "file": ("client-chosen-name-secret.pdf", io.BytesIO(minimal_pdf()), "application/pdf"),
        })
    assert response.status_code == 200
    assert "client-chosen-name-secret" not in caplog.text
    assert "Paper uploaded" in caplog.text and response.json()["paper_id"][:12] in caplog.text


def test_query_failures_are_categorical_and_question_text_is_never_logged(client, monkeypatch, caplog):
    question = "SENSITIVE question about a confidential merger"
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    monkeypatch.setattr(api.rag, "query", lambda **_: (_ for _ in ()).throw(RuntimeError("private query detail")))
    with caplog.at_level(logging.INFO):
        response = client.post("/query", headers=AUTH_HEADERS, json={"question": question})
    assert response.status_code == 500
    assert response.json()["detail"]["category"] == "query_failed"
    assert "private query detail" not in response.text and "private query detail" not in caplog.text
    assert "SENSITIVE" not in caplog.text and "merger" not in caplog.text
    assert "question_chars=" in caplog.text and "category=query_failed" in caplog.text
    monkeypatch.setattr(api.rag, "query", lambda **_: {
        "answer": "ok", "citations": [], "papers_searched": 0, "retrieval_time_ms": 0.0,
        "generation_time_ms": 0.0, "model_used": "demo-mode", "model_usage": None,
    })
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert client.post("/query", headers=AUTH_HEADERS, json={"question": question}).status_code == 200
    assert "SENSITIVE" not in caplog.text and "merger" not in caplog.text


def test_validation_errors_do_not_echo_submitted_content(client):
    long_question = "CONFIDENTIAL " * 200
    response = client.post("/query", headers=AUTH_HEADERS, json={"question": long_question, "top_k": 9})
    assert response.status_code == 422
    assert "CONFIDENTIAL" not in response.text
    detail = response.json()["detail"]
    assert detail["category"] == "invalid_request" and detail["request_id"]
    locations = [tuple(error["loc"]) for error in detail["errors"]]
    assert ("body", "question") in locations and ("body", "top_k") in locations
    assert all(set(error) == {"loc", "type", "msg"} for error in detail["errors"])
    malformed = client.post("/query", headers={**AUTH_HEADERS, "Content-Type": "application/json"},
                            content=b'{"question": "SECRET-BODY')
    assert malformed.status_code == 422 and "SECRET-BODY" not in malformed.text
    wrong = client.post("/papers/upload", headers=AUTH_HEADERS,
                        files={"file": ("secret-name.docx", b"data", "application/x-secret")})
    assert wrong.status_code == 400 and wrong.json()["detail"]["category"] == "invalid_file_type"
    assert "secret-name" not in wrong.text
    wrong_type = client.post("/papers/upload", headers=AUTH_HEADERS,
                             files={"file": ("paper.pdf", b"data", "application/x-secret-type")})
    assert wrong_type.status_code == 400 and wrong_type.json()["detail"]["category"] == "invalid_content_type"
    assert "x-secret-type" not in wrong_type.text


def test_framework_errors_share_the_categorical_shape(client):
    unknown = client.get("/nope", headers=AUTH_HEADERS)
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == {"message": "Not found.", "category": "not_found",
                                        "request_id": unknown.headers["x-request-id"]}
    wrong_method = client.post("/papers", headers=AUTH_HEADERS)
    assert wrong_method.status_code == 405
    assert wrong_method.json()["detail"]["category"] == "method_not_allowed"
    assert wrong_method.json()["detail"]["request_id"]
    unparseable = client.post("/papers/upload", headers={**AUTH_HEADERS, "Content-Type": "multipart/form-data"},
                              content=b"junk-body-SECRET")
    assert unparseable.status_code == 400
    assert unparseable.json()["detail"]["category"] == "invalid_request"
    assert "SECRET" not in unparseable.text and "junk" not in unparseable.text
    # Application errors keep their own categories, headers and request ids.
    busy = client.post("/query", headers=AUTH_HEADERS, json={"question": "What is attention?"})
    assert busy.status_code == 400 and busy.json()["detail"]["category"] == "empty_corpus"


def test_a_maximal_non_ascii_question_fits_under_the_json_cap(client, monkeypatch):
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    seen = {}

    def record(**kwargs):
        seen["question"] = kwargs["question"]
        return {"answer": "ok", "citations": [], "papers_searched": 0, "retrieval_time_ms": 0.0,
                "generation_time_ms": 0.0, "model_used": "demo-mode", "model_usage": None}

    monkeypatch.setattr(api.rag, "query", record)
    question = "\U0001F600" * 2000
    body = json.dumps({"question": question}).encode("ascii")  # \uXXXX escapes: 12 bytes per character
    assert 24000 < len(body) <= settings.resource_limits().max_json_body_bytes
    response = client.post("/query", headers={**AUTH_HEADERS, "Content-Type": "application/json"}, content=body)
    assert response.status_code == 200, response.text
    assert seen["question"] == question
    too_long = json.dumps({"question": question + "x"}).encode("ascii")
    assert client.post("/query", headers={**AUTH_HEADERS, "Content-Type": "application/json"}, content=too_long).status_code == 422


def test_delete_and_not_found_responses_are_categorical(client, monkeypatch):
    missing = client.delete("/papers/does-not-exist", headers=AUTH_HEADERS)
    assert missing.status_code == 404 and missing.json()["detail"]["category"] == "paper_not_found"
    too_long = client.delete("/papers/" + "x" * 200, headers=AUTH_HEADERS)
    assert too_long.status_code == 404
    monkeypatch.setattr(api.rag, "_extract_pdf", lambda _: [{"page": 1, "text": "Evidence about attention."}])
    uploaded = client.post("/papers/upload", headers=AUTH_HEADERS,
                           files={"file": ("a.pdf", io.BytesIO(minimal_pdf()), "application/pdf")})
    deleted = client.delete(f"/papers/{uploaded.json()['paper_id']}", headers=AUTH_HEADERS)
    assert deleted.status_code == 200
    assert deleted.json() == {"message": "Paper deleted.", "paper_id": uploaded.json()["paper_id"],
                              "request_id": deleted.headers["x-request-id"]}
