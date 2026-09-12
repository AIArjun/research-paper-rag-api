"""Small public-paper evidence regression, real embeddings, zero provider calls.

Run with the real-profile test image and mount frontend/public/samples at /fixtures.
This checks evidence coverage, not model answers or general retrieval accuracy.
The first eight questions were visible during development, not held-out benchmarks.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "demo"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.rag_engine import RAGEngine

FIXTURES = {
    "attention-is-all-you-need.pdf": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
    "retrieval-augmented-generation.pdf": "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233",
}
A, R = FIXTURES
CASES = [
    ("attention-scaling", A, "Why does scaled dot-product attention divide by the square root of d_k?",
     ["dot products grow large in magnitude", "extremely small gradients"]),
    ("rag-sequence-token", R, "What distinguishes RAG-Sequence from RAG-Token in how retrieved documents are used?",
     ["same document", "different latent document for each target token", "marginalized"]),
    ("rag-model-names", R, "In the RAG paper, which pre-trained models are used as the retriever and as the generator?",
     ["pre-trained bi-encoder from DPR", "BERT document encoder", "based on BERT", "BART-large"]),
    ("encoder-layers", A, "How many identical layers does the Transformer encoder stack use, and what are the two sub-layers in each?",
     ["N = 6", "self-attention", "feed-forward"]),
    ("bleu", A, "What BLEU score did the big Transformer model reach on WMT 2014 English-to-German?", ["28.4"]),
    ("training-hardware", A, "What hardware was used to train the Transformer models and how long did the big model train?", ["P100", "3.5"]),
    ("rag-retriever-training", R, "During RAG fine-tuning, which retriever encoder is updated and which is kept fixed?", ["document encoder", "query encoder"]),
    ("rag-wikipedia-index", R, "What Wikipedia snapshot and passage length are used in the document index?", ["December 2018", "100"]),
    ("absent-topic", A, "What dosage of amoxicillin is recommended for toddlers?", []),
]


def inspect_context(text, markers):
    normalized = " ".join(text.lower().split())
    return {marker: marker.lower() in normalized for marker in markers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model", default=settings.EMBEDDING_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    engine = RAGEngine()  # demo initialization never constructs a provider client
    engine._embeddings = HuggingFaceEmbeddings(
        model_name=args.model, model_kwargs={"device": "cpu"}, encode_kwargs={"batch_size": 32},
    )
    engine._vectorstore = Chroma(collection_name="evidence_regression", embedding_function=engine._embeddings)
    for filename, digest in FIXTURES.items():
        blob = (args.fixtures / filename).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest, f"Fixture changed: {filename}"
        engine.ingest_paper(blob, filename)

    # Capture the exact passages handed to generation. No model implementation exists.
    captured = {}
    engine._llm = object()

    def capture(question, passages, cap, request_id):
        captured["context"] = engine._build_context(passages, cap)
        captured["passages"] = passages
        return "Offline capture; no answer generated.", "offline-capture", None

    engine._generate_bounded = capture
    output = {"provider_calls": 0, "fixture_sha256": FIXTURES, "corpus": engine.get_stats(),
              "scope": "eight development regressions plus one absent topic; evidence only", "cases": []}
    for name, filename, question, markers in CASES:
        dense = engine._vectorstore.similarity_search_with_score(question, k=5, filter={"paper_id": FIXTURES[filename]})
        old_passages = [{"text": doc.page_content, "paper": filename, "page": doc.metadata["page"]} for doc, _ in dense]
        old_context = engine._build_context(old_passages, settings.MAX_CONTEXT_CHARS)
        result = engine.query(question, paper_id=FIXTURES[filename], top_k=5)
        context = captured["context"]
        assert len(context) <= settings.MAX_CONTEXT_CHARS
        assert all(c["paper_id"] == FIXTURES[filename] for c in result["citations"])
        for passage in captured["passages"]:
            assert passage["text"] in engine._page_texts[FIXTURES[filename]][passage["page"]]
        output["cases"].append({
            "id": name, "question": question, "filename": filename,
            "old": {"markers": inspect_context(old_context, markers), "pages": [p["page"] for p in old_passages], "context": old_context},
            "new": {"markers": inspect_context(context, markers), "pages": [p["page"] for p in captured["passages"]], "context": context},
        })
    answerable = [case for case in output["cases"] if case["new"]["markers"]]
    output["coverage"] = {variant: sum(all(c[variant]["markers"].values()) for c in answerable) for variant in ("old", "new")}
    output["coverage"]["total"] = len(answerable)
    output["absent_topic_note"] = "Still retrieves unrelated chunks; model abstention requires a separate live check."
    serialized = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({"coverage": output["coverage"], "provider_calls": 0, "output": str(args.output)}))
    else:
        print(serialized)
    if output["coverage"]["new"] != len(answerable):
        raise SystemExit("Expected source evidence missing; inspect report before deployment.")


if __name__ == "__main__":
    main()
