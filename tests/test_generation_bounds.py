"""Bounded provider construction and accounted generation; fake backends only."""

import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as api
from app.config import settings
from app.ledger import LedgerError
from app.protection import AdmissionSlot
from app.rag_engine import BackendUnavailableError, BudgetExhaustedError, GenerationError, RAGEngine
from app.tokens import FRAMING_TOKENS
from tests.conftest import AUTH_HEADERS
from tests.test_readiness_engine import fake_modules, real_configuration  # noqa: F401 (fixtures)


def _evidence_engine(count=1):
    engine = RAGEngine()
    assert engine.get_readiness()["ready"] is True
    engine._vectorstore.results = [
        (SimpleNamespace(page_content="Evidence " * 50, metadata={
            "filename": "paper.pdf", "page": index + 1, "paper_id": "paper", "chunk_id": f"paper-{index}",
        }), 0.9)
        for index in range(count)
    ]
    return engine


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "mutation_slot", AdmissionSlot(1, "mutation"))
    monkeypatch.setattr(api, "query_slot", AdmissionSlot(2, "query"))
    return TestClient(api.app, headers=AUTH_HEADERS)


# ─── Bounded clients ───

def test_the_supported_provider_is_constructed_with_explicit_bounds(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    engine = RAGEngine()
    for key, value in {"timeout": 30, "max_retries": 0, "max_tokens": 400, "model": "fake-test-model"}.items():
        assert engine._llm.configuration[key] == value
    assert engine.get_readiness()["ready"] is True
    assert engine.get_budget_status()["state"] == "ok"
    assert engine.get_budget_status()["token_bound"] == "tiktoken/fake_base"


def test_ollama_fails_closed_as_an_unsupported_token_bound_configuration(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(fake_modules["langchain_ollama"], "OllamaLLM",
                        lambda **_: pytest.fail("No Ollama client is constructed without a token bound"))
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "token_bound_unavailable"
    assert engine.get_budget_status() == {
        "state": "unavailable", "configured": False, "usage": None, "token_bound": None,
    }
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("A question")
    assert error.value.category == "token_bound_unavailable"


def test_reservations_come_from_the_model_bound_not_a_character_heuristic(
    monkeypatch, real_configuration, fake_modules
):
    monkeypatch.setattr(settings, "LLM_MAX_OUTPUT_TOKENS", 123)
    engine = _evidence_engine()
    prompts = []

    def capture(prompt):
        prompts.append(prompt)
        return "an answer"

    engine._llm = SimpleNamespace(invoke=capture)
    usage = engine.query("What is the evidence? 🙂 日本語")["model_usage"]
    prompt = prompts[0]
    assert usage["tokens_reserved"] == len(prompt) + FRAMING_TOKENS + 123  # fake encoding: one token per character
    assert usage["reservation_bound"] == "tiktoken/fake_base"
    assert usage["tokens_reserved"] > len(prompt) // 3 + 1 + 123
    assert engine._ledger.summary()["tokens_charged_total"] == usage["tokens_reserved"]


def test_an_openai_model_without_a_tiktoken_encoding_fails_closed_before_loading_backends(
    monkeypatch, real_configuration, fake_modules
):
    monkeypatch.setattr(settings, "LLM_MODEL", "unmapped-model")
    monkeypatch.setattr(fake_modules["langchain_huggingface"], "HuggingFaceEmbeddings",
                        lambda **_: pytest.fail("No backend is loaded without a token bound"))
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state["init_error"] == "token_bound_unavailable" and state["ready"] is False
    assert engine._llm is None and engine._ledger is None and engine._token_bound is None
    assert engine.get_budget_status()["token_bound"] is None
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("A question")
    assert error.value.category == "token_bound_unavailable"


def test_missing_tiktoken_is_a_missing_dependency(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "missing_dependency"
    assert engine._ledger is None


def test_a_measured_overshoot_is_charged_and_logged_without_content(
    monkeypatch, real_configuration, fake_modules, caplog
):
    monkeypatch.setattr(settings, "MAX_MODEL_TOKENS_PER_DAY", 90500)
    monkeypatch.setattr(settings, "MAX_MODEL_TOKENS_TOTAL", 90500)
    engine = _evidence_engine()
    engine._llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
        content="answer", usage_metadata={"input_tokens": 90000, "output_tokens": 10, "total_tokens": 90010},
    ))
    usage = engine.query("What is the evidence?", request_id="overshoot")["model_usage"]
    assert usage["tokens_charged"] == 90010 > usage["tokens_reserved"]
    assert "exceeded the reservation" in caplog.text and "Evidence" not in caplog.text
    assert engine._ledger.summary()["tokens_charged_total"] == 90010
    with pytest.raises(BudgetExhaustedError):
        engine.query("Another question?")


def test_configured_bounds_flow_into_the_client(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 12)
    monkeypatch.setattr(settings, "LLM_MAX_OUTPUT_TOKENS", 55)
    engine = RAGEngine()
    assert engine._llm.configuration["timeout"] == 12 and engine._llm.configuration["max_tokens"] == 55


def test_a_provider_that_cannot_accept_the_bounds_fails_closed(monkeypatch, real_configuration, fake_modules):
    class Unbounded:
        def __init__(self, model, temperature, api_key):
            pass

    monkeypatch.setattr(fake_modules["langchain_openai"], "ChatOpenAI", Unbounded)
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "model_initialization_failed"
    assert engine._llm is None and engine._ledger is None
    with pytest.raises(BackendUnavailableError):
        engine.query("A question")


@pytest.mark.parametrize("field,value", [
    ("LLM_TIMEOUT_SECONDS", 0), ("LLM_TIMEOUT_SECONDS", None), ("LLM_TIMEOUT_SECONDS", 301),
    ("LLM_MAX_OUTPUT_TOKENS", 0), ("LLM_MAX_OUTPUT_TOKENS", None), ("LLM_MAX_OUTPUT_TOKENS", 5000),
    ("MAX_MODEL_CALLS_TOTAL", None), ("MAX_MODEL_TOKENS_PER_DAY", -1), ("MODEL_CALL_LEDGER_PATH", None),
])
def test_invalid_provider_or_budget_settings_are_invalid_configuration(
    monkeypatch, real_configuration, fake_modules, field, value
):
    monkeypatch.setattr(settings, field, value)
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "invalid_configuration"
    assert engine._llm is None and engine._ledger is None


# ─── Disabled-by-default accounting ───

@pytest.mark.parametrize("field,value", [
    ("MODEL_CALL_LEDGER_PATH", ""), ("MODEL_CALL_LEDGER_PATH", "   "), ("MAX_MODEL_CALLS_PER_DAY", 0),
    ("MAX_MODEL_CALLS_TOTAL", 0), ("MAX_MODEL_TOKENS_PER_DAY", 0), ("MAX_MODEL_TOKENS_TOTAL", 0),
])
def test_real_generation_stays_disabled_until_every_allowance_is_explicit(
    monkeypatch, real_configuration, fake_modules, field, value
):
    monkeypatch.setattr(settings, field, value)
    monkeypatch.setattr(fake_modules["langchain_huggingface"], "HuggingFaceEmbeddings",
                        lambda **_: pytest.fail("No backend is loaded without an explicit budget"))
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state["init_error"] == "budget_not_configured" and state["ready"] is False
    assert state["effective_generation"] == "unavailable"
    assert engine._llm is None and engine._ledger is None
    assert engine.get_budget_status() == {
        "state": "not_configured", "configured": False, "usage": None, "token_bound": None,
    }
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("A question")
    assert error.value.category == "budget_not_configured"


def test_demo_mode_ignores_budget_settings(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setattr(settings, "MODEL_CALL_LEDGER_PATH", "")
    engine = RAGEngine()
    assert engine.get_readiness()["ready"] is True
    assert engine.get_budget_status() == {
        "state": "not_applicable", "configured": False, "usage": None, "token_bound": None,
    }


def test_an_unusable_ledger_fails_closed_before_loading_backends(monkeypatch, real_configuration, fake_modules, tmp_path):
    (tmp_path / "ledger.sqlite3").write_bytes(b"not a database" * 100)
    monkeypatch.setattr(fake_modules["langchain_huggingface"], "HuggingFaceEmbeddings",
                        lambda **_: pytest.fail("No backend is loaded with a corrupt ledger"))
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "ledger_unavailable"
    assert engine.get_budget_status()["state"] == "unavailable"
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("A question")
    assert error.value.category == "ledger_unavailable"
    assert (tmp_path / "ledger.sqlite3").read_bytes() == b"not a database" * 100


def test_an_injected_model_without_a_ledger_is_never_invoked(monkeypatch, real_configuration, fake_modules):
    engine = _evidence_engine()
    engine._ledger = None
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("What is the evidence?")
    assert error.value.category == "budget_not_configured"
    assert engine._llm.calls == 0


# ─── Accounting around the call ───

def test_calls_are_reserved_before_the_provider_and_measured_when_reported(real_configuration, fake_modules):
    engine = _evidence_engine()
    order = []
    original_reserve = engine._ledger.reserve

    def recording_reserve(*args, **kwargs):
        order.append("reserve")
        return original_reserve(*args, **kwargs)

    engine._ledger.reserve = recording_reserve

    class Measured:
        def invoke(self, prompt):
            order.append("invoke")
            assert "Evidence" in prompt
            return SimpleNamespace(content="measured answer", usage_metadata={
                "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
            })

    engine._llm = Measured()
    result = engine.query("What is the evidence?", request_id="req-measured")
    assert order == ["reserve", "invoke"]
    assert result["answer"] == "measured answer" and result["model_used"] == "fake-test-model"
    usage = result["model_usage"]
    assert usage["accounting"] == "measured"
    assert (usage["input_tokens"], usage["output_tokens"], usage["tokens_charged"]) == (120, 30, 150)
    assert usage["tokens_reserved"] > 400 and usage["context_chars"] > 0
    summary = engine._ledger.summary()
    assert summary["calls_total"] == 1 and summary["tokens_charged_total"] == 150
    assert summary["tokens_measured_total"] == 150 and summary["calls_unsettled"] == 0


@pytest.mark.parametrize("response", ["a plain string answer", SimpleNamespace(content="content only"),
                                      SimpleNamespace(content=[{"type": "text", "text": "block answer"}])])
def test_unreported_usage_is_charged_at_the_reservation(real_configuration, fake_modules, response):
    engine = _evidence_engine()
    engine._llm = SimpleNamespace(invoke=lambda prompt: response)
    result = engine.query("What is the evidence?")
    usage = result["model_usage"]
    assert usage["accounting"] == "reserved"
    assert usage["input_tokens"] is None and usage["output_tokens"] is None
    assert usage["tokens_charged"] == usage["tokens_reserved"]
    assert engine._ledger.summary()["tokens_charged_total"] == usage["tokens_reserved"]
    assert result["answer"] in {"a plain string answer", "content only", "block answer"}


@pytest.mark.parametrize("failure", [TimeoutError("provider timed out"), OSError("connection reset"),
                                     RuntimeError("private provider detail")])
def test_failed_and_timed_out_calls_stay_counted(real_configuration, fake_modules, failure, caplog):
    engine = _evidence_engine()

    def fail(prompt):
        raise failure

    engine._llm = SimpleNamespace(invoke=fail)
    with pytest.raises(GenerationError):
        engine.query("What is the evidence?")
    summary = engine._ledger.summary()
    assert summary["calls_total"] == 1 and summary["calls_unsettled"] == 0
    assert summary["tokens_charged_total"] > 400
    assert "private provider detail" not in caplog.text and "category=generation_failed" in caplog.text
    engine._llm = SimpleNamespace(invoke=lambda prompt: object())
    with pytest.raises(GenerationError):
        engine.query("What is the evidence?")
    assert engine._ledger.summary()["calls_total"] == 2


def test_abstention_consumes_no_allowance(real_configuration, fake_modules):
    engine = _evidence_engine(count=0)
    result = engine.query("A question nothing covers")
    assert result["model_used"] == "not-invoked" and result["model_usage"] is None
    assert engine._llm.calls == 0
    assert engine._ledger.summary()["calls_total"] == 0


def test_an_exhausted_allowance_makes_zero_further_provider_calls(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_TOTAL", 2)
    engine = _evidence_engine()
    assert engine.query("First?")["answer"] == "A fake test answer."
    assert engine.query("Second?")["answer"] == "A fake test answer."
    for _ in range(3):
        with pytest.raises(BudgetExhaustedError) as refused:
            engine.query("Third?")
        assert refused.value.scope == "total" and refused.value.kind == "calls"
    assert engine._llm.calls == 2
    assert engine._ledger.summary()["calls_total"] == 2


def test_a_truncated_ledger_fails_closed_on_restart_instead_of_replenishing(
    monkeypatch, real_configuration, fake_modules, tmp_path
):
    """Review probe at engine level: no fresh budget after the ledger file is truncated."""
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_TOTAL", 1)
    first = _evidence_engine()
    first.query("Only call?")
    assert first._llm.calls == 1
    (tmp_path / "ledger.sqlite3").write_bytes(b"")
    restarted = RAGEngine()
    assert restarted.get_readiness()["init_error"] == "ledger_unavailable"
    assert restarted.get_budget_status()["state"] == "unavailable"
    with pytest.raises(BackendUnavailableError) as error:
        restarted.query("Second call?")
    assert error.value.category == "ledger_unavailable"
    assert restarted._llm is None
    assert (tmp_path / "ledger.sqlite3").stat().st_size == 0


def test_exhaustion_survives_an_engine_restart_on_the_same_ledger(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_TOTAL", 1)
    first = _evidence_engine()
    first.query("First?")
    restarted = _evidence_engine()
    with pytest.raises(BudgetExhaustedError):
        restarted.query("Second?")
    assert restarted._llm.calls == 0
    assert restarted.get_budget_status()["usage"]["calls_total"] == 1


def test_a_ledger_write_failure_blocks_the_call(real_configuration, fake_modules):
    engine = _evidence_engine()

    def broken(*args, **kwargs):
        raise LedgerError("The ledger could not record the call.")

    engine._ledger.reserve = broken
    with pytest.raises(LedgerError):
        engine.query("What is the evidence?")
    assert engine._llm.calls == 0


def test_concurrent_queries_cannot_exceed_the_call_allowance(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_TOTAL", 3)
    engine = _evidence_engine()
    gate = threading.Barrier(6, timeout=20)
    outcomes = []
    lock = threading.Lock()

    def ask(index):
        gate.wait()
        try:
            engine.query(f"Question {index}?")
            outcome = "answered"
        except BudgetExhaustedError:
            outcome = "refused"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=ask, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert outcomes.count("answered") == 3 and outcomes.count("refused") == 3
    assert engine._llm.calls == 3


# ─── API surface ───

def test_query_api_maps_budget_and_ledger_refusals(client, monkeypatch):
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    monkeypatch.setattr(api.rag, "query", lambda **_: (_ for _ in ()).throw(BudgetExhaustedError("daily", "calls", 120)))
    response = client.post("/query", json={"question": "What is attention?"})
    assert response.status_code == 429 and response.headers["retry-after"] == "120"
    detail = response.json()["detail"]
    assert detail["category"] == "budget_exhausted" and detail["scope"] == "daily" and detail["kind"] == "calls"
    monkeypatch.setattr(api.rag, "query", lambda **_: (_ for _ in ()).throw(BudgetExhaustedError("total", "tokens")))
    response = client.post("/query", json={"question": "What is attention?"})
    assert response.status_code == 429 and "retry-after" not in response.headers
    monkeypatch.setattr(api.rag, "query", lambda **_: (_ for _ in ()).throw(LedgerError("private ledger path")))
    response = client.post("/query", json={"question": "What is attention?"})
    assert response.status_code == 503 and response.json()["detail"]["category"] == "ledger_unavailable"
    assert "private ledger path" not in response.text
    assert api.query_slot.in_use == 0


def test_query_response_exposes_safe_usage_metadata(client, monkeypatch):
    monkeypatch.setattr(api.rag, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    monkeypatch.setattr(api.rag, "query", lambda **_: {
        "answer": "supported", "citations": [], "papers_searched": 0, "retrieval_time_ms": 1.0,
        "generation_time_ms": 2.0, "model_used": "fake-test-model",
        "model_usage": {"accounting": "measured", "input_tokens": 10, "output_tokens": 5,
                        "tokens_charged": 15, "tokens_reserved": 700, "context_chars": 120},
    })
    response = client.post("/query", json={"question": "What is attention?"})
    assert response.status_code == 200
    assert response.json()["model_usage"] == {
        "accounting": "measured", "input_tokens": 10, "output_tokens": 5,
        "tokens_charged": 15, "tokens_reserved": 700, "context_chars": 120, "reservation_bound": None,
    }


def test_a_ledger_that_vanishes_at_runtime_makes_the_api_unready_without_recreating_it(
    client, monkeypatch, real_configuration, fake_modules, tmp_path
):
    engine = _evidence_engine()
    monkeypatch.setattr(engine, "get_stats", lambda: {"papers_loaded": 1, "total_chunks": 1})
    monkeypatch.setattr(api, "rag", engine)
    assert client.get("/ready").status_code == 200
    ledger_file = tmp_path / "ledger.sqlite3"
    ledger_file.unlink()
    ready = client.get("/ready")
    assert ready.status_code == 503 and ready.json()["ready"] is False
    assert ready.json()["model_budget"]["state"] == "unavailable"
    assert ready.json()["init_error"] is None
    assert client.get("/health").status_code == 503
    query = client.post("/query", json={"question": "What is the evidence?"})
    assert query.status_code == 503 and query.json()["detail"]["category"] == "ledger_unavailable"
    assert engine._llm.calls == 0
    assert not ledger_file.exists(), "a readiness probe or a call must never create a fresh ledger"


def test_readiness_reports_budget_state_without_secrets_or_paths(client, monkeypatch, real_configuration, fake_modules, tmp_path):
    monkeypatch.setattr(api, "rag", RAGEngine())
    ready = client.get("/ready")
    assert ready.status_code == 200, ready.text
    budget = ready.json()["model_budget"]
    assert budget["state"] == "ok" and budget["configured"] is True
    assert budget["usage"]["calls_total"] == 0 and budget["usage"]["total_call_allowance"] == 10
    assert budget["usage"]["ledger_created_at"]
    assert str(tmp_path) not in ready.text and "not-a-real-key" not in ready.text
    assert ready.json()["effective_generation"] == "openai"
    assert ready.json()["provider_connection_verified"] is False
