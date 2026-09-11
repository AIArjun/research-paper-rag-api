"""Readiness and failure behavior using local fake modules only; no provider calls."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.config import settings
from app.rag_engine import (
    BackendUnavailableError,
    GenerationError,
    RAGEngine,
    StorageMutationError,
)


@pytest.fixture
def fake_modules(monkeypatch):
    modules = {}
    for name in (
        "langchain_huggingface",
        "langchain_chroma",
        "langchain_ollama",
        "langchain_openai",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.configuration = kwargs

    class FakeStore:
        def __init__(self, **kwargs):
            self.results = []

        def similarity_search_with_relevance_scores(self, question, **kwargs):
            return self.results

    class FakeModel:
        def __init__(self, **kwargs):
            self.calls = 0
            self.configuration = kwargs

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content="A fake test answer.")

    modules["langchain_huggingface"].HuggingFaceEmbeddings = FakeEmbeddings
    modules["langchain_chroma"].Chroma = FakeStore
    modules["langchain_ollama"].OllamaLLM = FakeModel
    modules["langchain_openai"].ChatOpenAI = FakeModel

    # A deterministic tokenizer: one token per character; "unmapped-model" is unknown.
    class FakeEncoding:
        name = "fake_base"

        def encode(self, text, disallowed_special=()):
            return [1] * len(text)

    def encoding_for_model(model):
        if model == "unmapped-model":
            raise KeyError(model)
        return FakeEncoding()

    tiktoken = ModuleType("tiktoken")
    tiktoken.encoding_for_model = encoding_for_model
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken)
    modules["tiktoken"] = tiktoken
    return modules


@pytest.fixture
def real_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_MODEL", "fake-test-model")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(settings, "CHUNK_SIZE", 500)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 100)
    # Real generation is disabled until accounting is explicit; tests use a temporary ledger.
    monkeypatch.setattr(settings, "MODEL_CALL_LEDGER_PATH", str(tmp_path / "ledger.sqlite3"))
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_PER_DAY", 10)
    monkeypatch.setattr(settings, "MAX_MODEL_CALLS_TOTAL", 10)
    monkeypatch.setattr(settings, "MAX_MODEL_TOKENS_PER_DAY", 100000)
    monkeypatch.setattr(settings, "MAX_MODEL_TOKENS_TOTAL", 100000)


@pytest.mark.parametrize(
    "field,value",
    [
        ("LLM_PROVIDER", "unknown"), ("LLM_PROVIDER", []), ("LLM_PROVIDER", None),
        ("LLM_MODEL", None), ("CHUNK_SIZE", 0), ("CHUNK_OVERLAP", 500),
    ],
)
def test_invalid_startup_settings_are_explicitly_unready(monkeypatch, field, value):
    monkeypatch.setattr(settings, field, value)
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state["ready"] is False
    assert state["init_error"] == "invalid_configuration"
    assert state["effective_retrieval"] == "unavailable"
    assert state["effective_generation"] == "unavailable"
    assert isinstance(state["configured_provider"], str)
    assert isinstance(state["configured_model"], str)
    with pytest.raises(BackendUnavailableError) as error:
        engine.query("A question")
    assert error.value.category == "invalid_configuration"


@pytest.mark.parametrize("variable", ["CHUNK_SIZE", "CHUNK_OVERLAP"])
def test_malformed_chunk_environment_reports_readiness_instead_of_import_failure(variable):
    environment = {
        **os.environ,
        "LLM_PROVIDER": "demo",
        "OPENAI_API_KEY": "",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "100",
        variable: "not-an-integer",
    }
    check = subprocess.run(
        [sys.executable, "-c", (
            "from app.rag_engine import RAGEngine; "
            "state = RAGEngine().get_readiness(); "
            "assert state['ready'] is False; "
            "assert state['init_error'] == 'invalid_configuration'; "
            "print('invalid_configuration')"
        )],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == "invalid_configuration"


def test_missing_key_fails_before_importing_real_dependencies(monkeypatch, real_configuration):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "   ")
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == "missing_api_key"
    assert engine._vectorstore is None and engine._llm is None
    with pytest.raises(BackendUnavailableError):
        engine.ingest_paper(b"not processed", "paper.pdf")


def test_missing_dependency_is_not_a_demo_fallback(monkeypatch, real_configuration, fake_modules):
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state["configured_provider"] == "openai"
    assert state["init_error"] == "missing_dependency"
    assert state["ready"] is False
    assert engine._embeddings is None and engine._vectorstore is None and engine._llm is None


@pytest.mark.parametrize(
    "module,class_name,category",
    [
        ("langchain_huggingface", "HuggingFaceEmbeddings", "embedding_initialization_failed"),
        ("langchain_chroma", "Chroma", "storage_initialization_failed"),
        ("langchain_openai", "ChatOpenAI", "model_initialization_failed"),
    ],
)
def test_partial_initialization_is_reset_and_errors_are_categorical(
    monkeypatch, real_configuration, fake_modules, module, class_name, category, caplog
):
    def fail(**kwargs):
        raise OSError("sensitive backend detail")

    monkeypatch.setattr(fake_modules[module], class_name, fail)
    engine = RAGEngine()
    assert engine.get_readiness()["init_error"] == category
    assert engine._embeddings is None and engine._vectorstore is None and engine._llm is None
    assert "sensitive backend detail" not in caplog.text
    assert f"category={category}" in caplog.text
    assert "exception_type=OSError" in caplog.text
    with pytest.raises(BackendUnavailableError) as error:
        engine.assert_backend_ready()
    assert error.value.category == category


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_client_construction_reports_local_readiness_not_remote_verification(
    monkeypatch, real_configuration, fake_modules, provider
):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state == {
        "configured_provider": provider,
        "configured_model": "fake-test-model",
        "effective_retrieval": "chroma",
        "effective_generation": provider,
        "ready": True,
        "init_error": None,
        "pending_cleanup_ids": [],
        "provider_connection_verified": False,
    }
    assert engine._llm.calls == 0
    assert engine._embeddings.configuration["encode_kwargs"] == {"batch_size": 32}
    if provider == "ollama":
        assert engine._llm.configuration["validate_model_on_init"] is False


def test_readiness_is_cheap_and_does_not_probe_or_bool_test_backends(
    real_configuration, fake_modules
):
    engine = RAGEngine()

    class NoProbes:
        def __bool__(self):
            pytest.fail("Readiness must inspect presence, not call backend behavior")

        def __getattr__(self, name):
            pytest.fail("Readiness must not query a backend")

    engine._embeddings = engine._vectorstore = engine._llm = NoProbes()
    assert engine.get_readiness()["ready"] is True
    assert engine.get_readiness()["provider_connection_verified"] is False


def test_falsey_initialized_backends_still_ingest_query_and_delete(
    monkeypatch, real_configuration, fake_modules
):
    engine = RAGEngine()

    class FalseyBackend:
        def __bool__(self):
            return False

    class Store(FalseyBackend):
        def __init__(self):
            self.rows = []
            self.deleted = []

        def add_texts(self, texts, metadatas, ids):
            self.rows = [
                (SimpleNamespace(page_content=text, metadata=metadata), 0.9)
                for text, metadata in zip(texts, metadatas)
            ]

        def similarity_search_with_relevance_scores(self, question, **kwargs):
            return self.rows

        def delete(self, ids):
            self.deleted = ids

    class Model(FalseyBackend):
        def invoke(self, prompt):
            return SimpleNamespace(content="An actual fake-model invocation")

    engine._embeddings = FalseyBackend()
    engine._vectorstore = Store()
    engine._llm = Model()
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 1, "text": "Evidence"}])
    uploaded = engine.ingest_paper(b"falsey backend test", "paper.pdf")
    assert engine._vectorstore.rows and engine.chunks_store == []
    assert engine.query("Evidence")["answer"] == "An actual fake-model invocation"
    assert engine.delete_paper(uploaded["paper_id"]) is True
    assert engine._vectorstore.deleted


@pytest.mark.parametrize("component", ["_embeddings", "_vectorstore", "_llm"])
def test_missing_real_component_blocks_query_and_upload(
    monkeypatch, real_configuration, fake_modules, component
):
    engine = RAGEngine()
    setattr(engine, component, None)
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: pytest.fail("Must fail before parsing"))
    with pytest.raises(BackendUnavailableError):
        engine.query("A question")
    with pytest.raises(BackendUnavailableError):
        engine.ingest_paper(b"not processed", "paper.pdf")
    assert engine.get_readiness()["ready"] is False


def test_generation_error_propagates_without_demo_answer(
    monkeypatch, real_configuration, fake_modules, caplog
):
    engine = RAGEngine()
    engine._vectorstore.results = [
        (SimpleNamespace(page_content="Evidence", metadata={"filename": "paper.pdf", "page": 3}), 0.9)
    ]

    def fail(prompt):
        raise OSError("sensitive provider detail")

    monkeypatch.setattr(engine._llm, "invoke", fail)
    monkeypatch.setattr(engine, "_demo_generate", lambda *_: pytest.fail("No demo fallback"))
    with pytest.raises(GenerationError) as error:
        engine.query("A question")
    assert "sensitive provider detail" not in str(error.value)
    assert "sensitive provider detail" not in caplog.text
    assert "category=generation_failed" in caplog.text
    assert "exception_type=OSError" in caplog.text


def test_real_empty_retrieval_abstains_without_claiming_model_use(real_configuration, fake_modules):
    engine = RAGEngine()
    result = engine.query("A question not covered by retrieved passages")
    assert result["citations"] == []
    assert result["model_used"] == "not-invoked"
    assert "insufficient evidence" in result["answer"]
    assert engine._llm.calls == 0


def test_explicit_demo_remains_available_without_model_dependencies(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)
    engine = RAGEngine()
    state = engine.get_readiness()
    assert state["ready"] is True and state["init_error"] is None
    assert state["effective_retrieval"] == "memory-keyword"
    assert state["effective_generation"] == "demo"
    assert engine.query("No documents")["model_used"] == "not-invoked"


def test_pending_failed_upload_is_visible_with_metadata_and_not_counted_ready(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    engine = RAGEngine()
    monkeypatch.setattr(engine, "_extract_pdf", lambda _: [{"page": 4, "text": "Evidence text"}])
    existing = engine.ingest_paper(b"existing", "existing.pdf")

    def fail(**kwargs):
        raise OSError("Simulated storage failure")

    engine._vectorstore = SimpleNamespace(add_texts=fail, delete=fail)
    engine._embeddings = object()
    with pytest.raises(StorageMutationError) as failure:
        engine.ingest_paper(b"failed upload", "failed.pdf")
    pending_id = failure.value.paper_id
    default_rows = engine.list_papers()
    assert [paper["paper_id"] for paper in default_rows] == [existing["paper_id"]]
    assert "status" not in default_rows[0]
    visible = {paper["paper_id"]: paper for paper in engine.list_papers(include_pending=True)}
    assert visible[existing["paper_id"]]["status"] == "ready"
    assert visible[pending_id]["status"] == "pending_cleanup"
    assert visible[pending_id]["filename"] == "failed.pdf"
    assert visible[pending_id]["pages"] == 1
    assert visible[pending_id]["uploaded_at"]
    assert engine.get_stats()["papers_loaded"] == 1
    assert engine.get_readiness()["pending_cleanup_ids"] == [pending_id]
    assert engine.get_readiness()["ready"] is False
    engine._vectorstore.delete = lambda **_: None
    assert engine.delete_paper(pending_id) is True
    assert pending_id not in engine._pending_metadata
    assert engine.get_readiness()["ready"] is True


def test_pending_id_without_metadata_is_still_discoverable(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "demo")
    engine = RAGEngine()
    engine._pending_cleanup["orphan"] = ["orphan-0"]
    assert engine.list_papers() == []
    assert engine.list_papers(include_pending=True) == [{
        "paper_id": "orphan", "filename": None, "pages": None,
        "chunks": 1, "uploaded_at": None, "status": "pending_cleanup",
    }]
