"""Real Chroma/local-embedding process restart checks. No provider client/calls.

Uses only a newly created temporary corpus and the two digest-pinned public
fixtures. Run inside the real-profile test image with --network none.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_retrieval import FIXTURES


def child(args):
    from app.config import settings
    from app.rag_engine import RAGEngine
    # Real initialization (including ledger/token bound), but model construction
    # is replaced before initialization. There is no network-capable LLM object.
    settings.LLM_PROVIDER = "openai"
    settings.LLM_MODEL = "gpt-4o-mini"
    settings.OPENAI_API_KEY = "offline-fixture-not-a-key"
    settings.EMBEDDING_MODEL = args.model
    settings.PAPER_STORE_PATH = str(args.directory / "papers.sqlite3")
    settings.MODEL_CALL_LEDGER_PATH = str(args.directory / "ledger.sqlite3")
    settings.MAX_MODEL_CALLS_PER_DAY = settings.MAX_MODEL_CALLS_TOTAL = 1
    settings.MAX_MODEL_TOKENS_PER_DAY = settings.MAX_MODEL_TOKENS_TOTAL = 10000
    RAGEngine._init_llm = lambda self: setattr(self, "_llm", object())
    engine = RAGEngine()
    assert engine.get_readiness()["ready"], engine.get_readiness()
    manifest = args.directory / "expected.json"

    def snapshot():
        vectors = engine._vectorstore.get(include=["documents", "metadatas"])
        return {
            "papers": engine.list_papers(), "chunks": engine.chunks_store,
            "pages": {pid: {str(page): text for page, text in pages.items()}
                      for pid, pages in engine._page_texts.items()},
            "vectors": {cid: [text, meta] for cid, text, meta in zip(
                vectors["ids"], vectors["documents"], vectors["metadatas"])},
        }

    if args.phase == "populate":
        for filename, digest in FIXTURES.items():
            blob = (args.fixtures / filename).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == digest
            engine.ingest_paper(blob, filename)
        manifest.write_text(json.dumps(snapshot()), encoding="utf-8")
    elif args.phase in {"restart", "repair"}:
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        assert snapshot() == expected
        for filename, digest in FIXTURES.items():
            blob = (args.fixtures / filename).read_bytes()
            assert engine._paper_store.read_pdf(digest) == blob
            assert engine.ingest_paper(blob, "renamed.pdf")["filename"] == filename
        captured = []

        def capture(question, passages, cap, request_id):
            captured.extend(passages)
            return "Offline evidence capture", "not-invoked", None

        engine._generate_bounded = capture
        selected = FIXTURES["retrieval-augmented-generation.pdf"]
        result = engine.query("Which pre-trained models are used as the retriever and generator?", selected)
        assert result["citations"] and all(c["paper_id"] == selected for c in result["citations"])
        for passage in captured:
            assert passage["text"] in engine._page_texts[selected][passage["page"]]
        context = " ".join(p["text"] for p in captured)
        assert "DPR" in context and "BART" in context
        if args.phase == "restart":
            # Simulate loss of a subset of the derived index. Next process must
            # restore it from canonical chunks, without re-uploading the PDFs.
            engine._vectorstore.delete(ids=list(expected["vectors"])[:3])
        else:
            assert engine.delete_paper(selected)
    elif args.phase == "delete_survives":
        selected = FIXTURES["retrieval-augmented-generation.pdf"]
        assert engine._paper_store.read_pdf(selected) is None
        assert engine.get_stats()["papers_loaded"] == 1
        assert not engine._vectorstore.get(where={"paper_id": selected})["ids"]
        assert not engine.query("retriever model", selected)["citations"]
        for paper in engine.list_papers():
            assert engine.delete_paper(paper["paper_id"])
    elif args.phase == "empty_restart":
        assert engine.get_stats() == {"papers_loaded": 0, "total_chunks": 0}
        assert engine._vectorstore._collection.count() == 0
    assert engine._ledger.summary()["calls_total"] == 0
    engine.close()
    print(json.dumps({"phase": args.phase, "passed": True, "provider_calls": 0}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "/opt/models/all-MiniLM-L6-v2"))
    parser.add_argument("--phase", choices=["populate", "restart", "repair", "delete_survives", "empty_restart"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.phase:
        child(args)
        return
    environment = {**os.environ, "OPENAI_API_KEY": "", "PAPER_STORE_PATH": "",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "ANONYMIZED_TELEMETRY": "false", "TOKENIZERS_PARALLELISM": "false"}
    with tempfile.TemporaryDirectory(prefix="rag-durability-") as directory:
        for phase in ["populate", "restart", "repair", "delete_survives", "empty_restart"]:
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--phase", phase,
                "--directory", directory, "--fixtures", str(args.fixtures.resolve()), "--model", args.model,
            ], env=environment, text=True, capture_output=True, timeout=300)
            if completed.returncode:
                # Only public fixtures and fake credentials are used by children.
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(f"Durability check failed: {phase}")
            print(completed.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
