"""Control behavior of scripts/verify_live_demo.py against a fake server.

No credential and no network: a local HTTP fake plays the protected API and
records every request, so the tests can prove that the default run makes no
generation call, that live calls are opt-in, bounded and never retried, that
the token is read from the environment only and never reaches the evidence
files, and that a changed ledger identity is reported as not persisted.
"""

import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.conftest import TEST_ACCESS_TOKEN, multi_page_pdf

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_live_demo.py"
_spec = importlib.util.spec_from_file_location("verify_live_demo", MODULE_PATH)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


class FakeDemo:
    """Records requests; counts only unfiltered queries as paid calls."""

    def __init__(self):
        self.requests = []
        self.calls_total = 0
        self.ledger_created_at = "2026-09-11T10:00:00+00:00"
        self.fail_live_after = None  # index of the live call that answers 502
        self.live_unaccounted = False  # answer 200 without a model call or ledger charge
        self.lock = threading.Lock()

    def ready(self):
        return {
            "ready": True, "configured_provider": "openai", "configured_model": "gpt-4o-mini",
            "effective_retrieval": "chroma", "effective_generation": "openai", "access_configured": True,
            "model_budget": {"state": "ok", "configured": True, "token_bound": "tiktoken/o200k_base",
                             "usage": {"calls_today": self.calls_total, "calls_total": self.calls_total,
                                       "tokens_charged_today": 2200 * self.calls_total,
                                       "tokens_charged_total": 2200 * self.calls_total,
                                       "tokens_measured_total": 2200 * self.calls_total,
                                       "calls_unsettled": 0, "ledger_created_at": self.ledger_created_at}},
        }


def _handler(state: FakeDemo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status, payload, headers=None):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Request-ID", "fake-request")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            return self.headers.get("Authorization") == "Bearer " + TEST_ACCESS_TOKEN

        def _record(self, kind, **extra):
            with state.lock:
                state.requests.append({"method": self.command, "path": self.path, "kind": kind,
                                       "authorized": self._authorized(), **extra})

        def do_GET(self):
            if self.path == "/ready":
                self._record("ready")
                return self._send(200, state.ready())
            if self.path == "/papers":
                self._record("list")
                if not self._authorized():
                    return self._send(401, {"detail": {"category": "unauthorized"}},
                                      {"WWW-Authenticate": "Bearer"})
                return self._send(200, [])
            self._send(404, {"detail": {"category": "not_found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not self._authorized():
                self._record("unauthorized_post")
                return self._send(401, {"detail": {"category": "unauthorized"}}, {"WWW-Authenticate": "Bearer"})
            if self.path == "/papers/upload":
                self._record("upload", body_bytes=len(raw))
                return self._send(200, {"paper_id": hashlib.sha256(raw).hexdigest()[:16], "filename": "x.pdf",
                                        "pages": 2, "chunks": 3, "processing_time_ms": 1.0, "message": "ok"})
            if self.path == "/query":
                body = json.loads(raw)
                if body.get("paper_id", "").startswith("no-such-paper-"):
                    self._record("abstention")
                    return self._send(200, {"request_id": "r", "question": body["question"], "answer": "n/a",
                                            "citations": [], "papers_searched": 0, "retrieval_time_ms": 1.0,
                                            "generation_time_ms": 0.0, "total_time_ms": 1.0,
                                            "model_used": "not-invoked", "model_usage": None})
                with state.lock:
                    index = sum(1 for r in state.requests if r["kind"] == "live")
                    state.requests.append({"method": "POST", "path": "/query", "kind": "live",
                                           "authorized": True, "question": body["question"]})
                    if not state.live_unaccounted:
                        state.calls_total += 1  # a failed attempt stays charged, like the real ledger
                if state.live_unaccounted:
                    return self._send(200, {"request_id": "r", "question": body["question"], "answer": "Demo.",
                                            "citations": [{"text": "N = 6", "page": 3, "paper": "a.pdf",
                                                           "relevance_score": 0.5}],
                                            "papers_searched": 1, "retrieval_time_ms": 1.0,
                                            "generation_time_ms": 0.0, "total_time_ms": 1.0,
                                            "model_used": "demo-mode", "model_usage": None})
                if state.fail_live_after is not None and index >= state.fail_live_after:
                    return self._send(502, {"detail": {"category": "generation_failed"}})
                return self._send(200, {"request_id": "r", "question": body["question"], "answer": "Six layers.",
                                        "citations": [{"text": "N = 6", "page": 3, "paper": "a.pdf",
                                                       "relevance_score": 0.5}],
                                        "papers_searched": 1, "retrieval_time_ms": 1.0, "generation_time_ms": 2.0,
                                        "total_time_ms": 3.0, "model_used": "gpt-4o-mini",
                                        "model_usage": {"accounting": "measured", "input_tokens": 1900,
                                                        "output_tokens": 300, "tokens_charged": 2200,
                                                        "tokens_reserved": 2400, "context_chars": 5000,
                                                        "reservation_bound": "tiktoken/o200k_base"}})
            self._send(404, {"detail": {"category": "not_found"}})

    return Handler


@pytest.fixture
def fake_demo():
    state = FakeDemo()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.base_url = "http://127.0.0.1:%d" % server.server_address[1]
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fixtures(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    manifest = []
    for name, pages in (("one.pdf", 2), ("two.pdf", 3)):
        content = multi_page_pdf(pages)
        (fixture_dir / name).write_bytes(content)
        manifest.append({"file": name, "sha256": hashlib.sha256(content).hexdigest(), "pages": 2, "chunks": 3})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"dir": str(fixture_dir), "manifest": str(manifest_path), "count": len(manifest)}


def _run(fake_demo, tmp_path, fixtures, *extra, environ=None):
    output = tmp_path / "evidence"
    argv = ["--base-url", fake_demo.base_url, "--output", str(output), "--manifest", fixtures["manifest"],
            "--fixture-dir", fixtures["dir"], *extra]
    code = verify.main(argv, {"DEMO_ACCESS_TOKEN": TEST_ACCESS_TOKEN} if environ is None else environ)
    files = sorted(output.glob("*")) if output.exists() else []
    evidence = json.loads(next(f for f in files if f.suffix == ".json").read_text()) if files else None
    return code, evidence, files


def _kinds(fake_demo):
    return [r["kind"] for r in fake_demo.requests]


def test_default_run_makes_no_generation_call_and_deletes_nothing(fake_demo, tmp_path, fixtures):
    code, evidence, files = _run(fake_demo, tmp_path, fixtures)
    assert code == 0
    kinds = _kinds(fake_demo)
    assert "live" not in kinds
    assert kinds.count("upload") == fixtures["count"]
    assert kinds.count("abstention") == 1
    assert not any(r["method"] == "DELETE" for r in fake_demo.requests)
    assert evidence["checks"]["safe_phase_passed"] is True
    assert evidence["checks"]["no_model_call_in_safe_phase"] is True
    assert evidence["checks"]["abstention_not_invoked"] is True
    assert all(upload["matches_expected"] for upload in evidence["uploads"])
    # The unauthenticated and wrong-token probes were sent without the real credential.
    assert [r["authorized"] for r in fake_demo.requests if r["kind"] == "list"][:2] == [False, False]
    assert {f.suffix for f in files} == {".json", ".md"}


def test_token_is_never_written_to_the_evidence_files(fake_demo, tmp_path, fixtures):
    code, _, files = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "1")
    assert code == 0
    for path in files:
        assert TEST_ACCESS_TOKEN not in path.read_text()
        assert "Authorization" not in path.read_text()


def test_live_generation_is_opt_in_and_bounded(fake_demo, tmp_path, fixtures):
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "2")
    assert code == 0
    assert _kinds(fake_demo).count("live") == 2
    assert evidence["checks"]["live_calls_attempted"] == 2
    assert [r["ledger_delta"]["calls_total"] for r in evidence["live"]] == [1, 1]
    assert evidence["live"][0]["response"]["body"]["model_used"] == "gpt-4o-mini"
    assert evidence["live"][0]["response"]["body"]["model_usage"]["accounting"] == "measured"


def test_more_than_the_hard_cap_is_refused_by_the_parser(fake_demo, tmp_path, fixtures):
    with pytest.raises(SystemExit):
        _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", str(verify.MAX_LIVE_CALLS + 1))
    assert fake_demo.requests == []


def test_a_failed_live_call_is_not_retried_stops_the_live_phase_and_fails_the_run(fake_demo, tmp_path, fixtures):
    fake_demo.fail_live_after = 0
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "3")
    assert code == 1
    assert _kinds(fake_demo).count("live") == 1  # one attempt, no retry, no further question
    assert evidence["checks"]["safe_phase_passed"] is True
    assert evidence["checks"]["live_phase_passed"] is False
    assert evidence["live_stopped_early"] == {"after_calls": 1, "status": 502}
    assert evidence["live"][0]["accounted"] is False
    assert "status_not_200" in evidence["live"][0]["failure_reasons"]
    assert evidence["live"][0]["ledger_delta"]["calls_total"] == 1  # the failed attempt stayed charged


def test_a_live_answer_without_model_backing_or_accounting_fails_the_run(fake_demo, tmp_path, fixtures):
    fake_demo.live_unaccounted = True
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "2")
    assert code == 1
    assert evidence["checks"]["live_phase_passed"] is False
    assert all(result["accounted"] is False for result in evidence["live"])
    reasons = set(evidence["live"][0]["failure_reasons"])
    assert {"model_used_not_configured_model", "usage_not_measured", "ledger_calls_delta_not_one"} <= reasons


def test_an_accounted_live_run_passes(fake_demo, tmp_path, fixtures):
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "1")
    assert code == 0
    assert evidence["checks"]["live_phase_passed"] is True
    assert evidence["live"][0]["accounted"] is True and evidence["live"][0]["failure_reasons"] == []


def test_a_declared_page_or_chunk_mismatch_fails_preflight_and_blocks_paid_calls(fake_demo, tmp_path, fixtures):
    manifest = json.loads(Path(fixtures["manifest"]).read_text())
    manifest[1]["chunks"] = 999
    Path(fixtures["manifest"]).write_text(json.dumps(manifest))
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures, "--live", "--max-live-calls", "3")
    assert code == 1
    assert evidence["checks"]["uploads_200"] is True
    assert evidence["checks"]["uploads_match_expected"] is False
    assert evidence["checks"]["safe_phase_passed"] is False
    assert "live" not in _kinds(fake_demo)
    assert "live" not in evidence


def test_missing_token_is_refused_before_any_request(fake_demo, tmp_path, fixtures):
    code, evidence, files = _run(fake_demo, tmp_path, fixtures, environ={})
    assert code == 2 and evidence is None and files == []
    assert fake_demo.requests == []


def test_no_command_line_token_option_exists():
    parser = verify.build_parser()
    assert not any("token" in option for option in parser._option_string_actions)


def test_fixture_digest_mismatch_uploads_nothing(fake_demo, tmp_path, fixtures):
    manifest = json.loads(Path(fixtures["manifest"]).read_text())
    manifest[0]["sha256"] = "0" * 64
    Path(fixtures["manifest"]).write_text(json.dumps(manifest))
    code, evidence, _ = _run(fake_demo, tmp_path, fixtures)
    assert code == 1
    assert "upload" not in _kinds(fake_demo)
    assert "digest mismatch" in evidence["aborted"]


def test_ledger_comparison_detects_a_replaced_ledger(fake_demo, tmp_path, fixtures):
    code, first, files = _run(fake_demo, tmp_path, fixtures, "--skip-uploads")
    assert code == 0
    previous = next(f for f in files if f.suffix == ".json")
    same_dir = tmp_path / "second"
    code = verify.main(["--base-url", fake_demo.base_url, "--output", str(same_dir), "--skip-uploads",
                        "--compare-ledger", str(previous)], {"DEMO_ACCESS_TOKEN": TEST_ACCESS_TOKEN})
    assert code == 0
    second = json.loads(next(same_dir.glob("*.json")).read_text())
    assert second["checks"]["ledger_persisted"] is True

    fake_demo.ledger_created_at = "2026-09-11T12:00:00+00:00"  # a fresh ledger after a bad deploy
    kinds_before = list(_kinds(fake_demo))
    code = verify.main(["--base-url", fake_demo.base_url, "--output", str(tmp_path / "third"),
                        "--compare-ledger", str(previous), "--live", "--max-live-calls", "3"],
                       {"DEMO_ACCESS_TOKEN": TEST_ACCESS_TOKEN})
    assert code == 1
    third = json.loads(next((tmp_path / "third").glob("*.json")).read_text())
    assert third["checks"]["ledger_persisted"] is False
    assert third["checks"]["safe_phase_passed"] is False
    # Discovered on the first snapshot: only readiness reads followed, no upload and no paid call.
    assert set(_kinds(fake_demo)[len(kinds_before):]) == {"ready"}
    assert "live" not in third and third.get("uploads") is None
