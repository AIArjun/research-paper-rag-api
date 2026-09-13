# 📚 Research Paper RAG API

A research **Retrieval-Augmented Generation (RAG)** API prototype. Upload research papers (PDF) and ask questions with source passages and page references. Real-model answers require a separately configured and verified backend.

**Open the app:** [Research Observatory](https://research-observatory-gamma.vercel.app) · [About the product / request access](https://arjunworks.se/research)

The browser application runs on Vercel. The [Render service](https://research-paper-rag-api.onrender.com) is its backend; [API documentation](https://research-paper-rag-api.onrender.com/docs) is intended for developers. The live app is a passcode-protected, shared public-paper research preview. Paper persistence is opt-in through PAPER_STORE_PATH; check corpus_storage on /health and the deployment restart evidence. See [durable corpus setup](docs/DURABLE-CORPUS.md). See [frontend setup and limits](frontend/README.md).

Built with **LangChain + FastAPI + ChromaDB + OpenAI/Ollama + Docker**.

**Deployment profiles:** `Dockerfile` installs lightweight demo dependencies. Demo mode uses keyword retrieval and template answers. `Dockerfile.real` provides a separate pinned Linux/Python 3.11 CPU profile with local embeddings and Chroma; its build, tests and resource measurements are described in [the real-profile guide](docs/REAL-PROFILE.md). Adding a provider key to the demo image does not install real-RAG libraries. Stage 2c adds shared-token access control, request and corpus bounds, admission control and a persistent model-call ledger for one protected demo; see [Stage 2c: protected demo](docs/STAGE2C.md). The optional [durable corpus](docs/DURABLE-CORPUS.md) saves original PDFs, extraction, metadata and mutation recovery state. [Retrieval checks](docs/RETRIEVAL_QUALITY.md) cover a small public-paper sample, not general answer accuracy. The public deployment may lag this branch; verify its deployed commit before relying on new behavior.

## Protected demo (Stage 2c)

Every route except `/`, `/docs`, `/redoc`, `/openapi.json`, `/health` and `/ready` requires `Authorization: Bearer <DEMO_ACCESS_TOKEN>`. Without a configured token, protected routes answer a categorical 503 in every mode; with a missing or wrong token they answer 401 with a `WWW-Authenticate` challenge, before any body is parsed. The token is compared in constant time and is never rendered in responses, docs or logs.

Environment variable names (values are examples, never real secrets):

| Variable | Purpose | Safe example |
|---|---|---|
| `DEMO_ACCESS_TOKEN` | Shared bearer token, 32–512 printable ASCII characters, no whitespace | output of `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ALLOWED_ORIGINS` | Explicit CORS origins, comma-separated; empty means none; wildcards are rejected | `https://demo.example` |
| `MAX_FILE_SIZE_MB`, `MAX_PDF_PAGES`, `MAX_CHUNKS_PER_PAPER`, `MAX_PAPERS`, `MAX_TOTAL_CHUNKS` | Upload and corpus ceilings (defaults 10 MiB, 60, 600, 20, 3000) | defaults |
| `MAX_CONTEXT_CHARS`, `MAX_CONCURRENT_QUERIES` | Prompt context ceiling and query slots (defaults 6000, 2); uploads are one at a time | defaults |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_OUTPUT_TOKENS` | Provider call bounds (defaults 30, 400); retries are always zero | defaults |
| `MODEL_CALL_LEDGER_PATH` | SQLite ledger file on durable storage; required for real providers | `/var/data/model-calls.sqlite3` |
| `MAX_MODEL_CALLS_PER_DAY`, `MAX_MODEL_CALLS_TOTAL`, `MAX_MODEL_TOKENS_PER_DAY`, `MAX_MODEL_TOKENS_TOTAL` | Call and token allowances; real generation stays disabled until all four are positive | `20`, `100`, `40000`, `200000` |

Fixed request bounds: questions up to 2000 characters, `top_k` at most 5, JSON bodies at most 32 KiB (enough for a 2000-character question in any escaped encoding), upload requests at most the file ceiling plus a 16 KiB multipart allowance. Total request bytes are enforced at the ASGI receive boundary, so a missing or misleading `Content-Length` cannot bypass them. Over-limit uploads answer 413 (size, pages, chunks) or 409 (paper count, corpus capacity) before embeddings or storage. Only one upload is admitted at a time, and the admission is taken before any body byte is received, so a second simultaneous upload answers 429 with `Retry-After` without being read. Every attempted real-model call is reserved in the ledger before the provider is contacted, using an explicit model-supported token bound (the exact tiktoken count plus framing and the output cap for OpenAI models), and stays counted if it fails or times out; abstentions consume nothing; an exhausted allowance answers 429 with no provider call. Ollama is an unsupported protected configuration: it fails closed as `token_bound_unavailable` because the server-side Modelfile `TEMPLATE`/`SYSTEM` adds input this client cannot bound. An existing empty ledger file is refused rather than reinitialized. These ceilings bound usage; they are not a verified dollar cap. The demo must run as one worker on one instance.

Run it locally in demo mode:

```bash
export LLM_PROVIDER=demo
export DEMO_ACCESS_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uvicorn app.main:app --port 8001 --workers 1
curl -H "Authorization: Bearer $DEMO_ACCESS_TOKEN" http://localhost:8001/papers
```

## Foundation behavior

- Chunk size and overlap come from configuration; invalid values are rejected and every chunk advances within its original physical PDF page.
- Paper IDs are full-document SHA-256 digests. Uploading identical bytes again is idempotent and returns the original filename, ID and metadata. Different PDFs sharing the same header remain distinct. Clients must use returned IDs rather than assume the old 12-character format.
- A paper is marked ready only after successful indexing. Failed partial indexing triggers cleanup. When cleanup cannot be confirmed, querying is blocked until the affected deletion succeeds; the API reports an explicit storage failure, and the paper list exposes a pending-cleanup status.
- A failed deletion is not reported as success. Retry the affected deletion to recover.
- Generation receives complete retrieved passages; response previews remain short. Citations include stable paper/chunk IDs for newly indexed content. Retrieved passages still require manual claim-and-page verification.

With PAPER_STORE_PATH configured, the registry and mutation journal survive restart; missing known vectors are rebuilt from canonical chunks. Without it, these protections remain process-local. Use a disposable local/test corpus for failure testing and follow the [migration guide](docs/DURABLE-CORPUS.md) for existing deployments.

## Readiness and failures

`GET /ready` reports configured provider/model, effective retrieval/generation components, initialization error category, pending-cleanup IDs, whether the access token is configured, model-call accounting state and the effective limits. It returns 503 when local components are unavailable, storage cleanup is pending, or no valid access token is configured. `GET /health` follows that readiness status; its `llm_provider` field describes effective generation rather than echoing the requested provider. Neither endpoint makes provider, embedding or vector-search calls (the model-budget summary is a local read-only ledger query run off the event loop; a ledger that becomes unusable after startup makes the API unready), and `provider_connection_verified=false` makes that limit explicit. A separate controlled query is required to establish provider connectivity and answer quality.

Invalid provider/chunk/limit configuration, missing keys or dependencies, unconfigured or unusable model-call accounting, and initialization failures are reported as unready. Real-mode queries cannot silently fall back to template answers: missing backends return 503, generation failures return a sanitized 502, and empty retrieval returns an explicit abstention marked `model_used=not-invoked`. A transient generation error does not rewrite local initialization status. Every error, including framework-raised 404/405/parse failures, carries a `category` and a `request_id` (also sent as `X-Request-ID`). Error responses and logs never include raw exceptions, prompts, passages, question text, client filenames or credentials; successful responses do return the caller's own question, short citation previews and the stored filename, which is the API's purpose.

`GET /papers` includes `status=ready` or `pending_cleanup`. Metadata unavailable for an incomplete record is null; active counts exclude pending records. Pending mutations are journaled when PAPER_STORE_PATH is set. Deletion requires usable storage so the recovery record cannot be discarded before vector removal is confirmed.

---

## How It Works

```
Upload PDF → Extract Text → Chunk → Embed → Store in ChromaDB
                                                    ↓
Ask Question → Semantic Search → Retrieve Top Chunks → LLM Generates Answer with Citations
```

## Features

- **PDF Processing** — Extract text from research papers, split into overlapping chunks
- **Vector Storage** — Embed chunks using sentence-transformers, store in ChromaDB
- **Semantic Search** — Find the most relevant passages for any question
- **LLM Answers** — Generate answers using OpenAI (GPT-4o-mini) or Ollama (llama3, mistral)
- **Page-Level Citations** — Every answer includes source paper and page number
- **Multi-Paper Support** — Upload multiple papers, query across all or filter by paper
- **Demo Mode** — Works without API keys for testing (keyword matching + template answers)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT-4o-mini / Ollama (llama3, mistral) |
| PDF Extraction | pdfplumber / pypdf |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

## Project Structure

```
research-paper-rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI endpoints
│   ├── rag_engine.py       # Core RAG pipeline
│   └── config.py           # Environment configuration
├── tests/
│   └── test_api.py         # Test suite
├── vectorstore/             # ChromaDB persistence (gitignored)
├── uploads/                 # Uploaded PDFs (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### Option 1: Local Development

```bash
git clone https://github.com/AIArjun/research-paper-rag-api.git
cd research-paper-rag-api

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements-deploy.txt

# Run in demo mode (no API key needed)
uvicorn app.main:app --reload --port 8001

```

On Windows, activate the virtual environment with `venv\Scripts\Activate.ps1`. This quick start runs the portable demo profile. For the Linux x86_64 real profile, use the separate container and measurement guide below; `requirements.txt` now points to the same hash-locked real dependencies.

### Option 2: Docker

```bash
docker-compose up --build
```

This uses the demo Dockerfile. To build and measure the real local embedding/retrieval pipeline without paid calls, follow [Stage 2b: real profile](docs/REAL-PROFILE.md). Its container runs without an external network during measurement and uses a placeholder key only to construct the client.

### Access

- **Landing Page:** http://localhost:8001
- **Swagger Docs:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

---

## API Endpoints

### `POST /papers/upload` — Upload a paper
```bash
curl -X POST http://localhost:8001/papers/upload \
  -H "Authorization: Bearer $DEMO_ACCESS_TOKEN" \
  -F "file=@my_paper.pdf"
```

### `POST /query` — Ask a question
```bash
curl -X POST http://localhost:8001/query \
  -H "Authorization: Bearer $DEMO_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What accuracy did the model achieve?", "top_k": 5}'
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "question": "What accuracy did the model achieve?",
  "answer": "Based on the paper, the model achieved 95% accuracy on the benchmark dataset (Source: ml_paper.pdf, Page 1).",
  "citations": [
    {
      "text": "Results show 95% accuracy on the benchmark dataset...",
      "page": 1,
      "paper": "ml_paper.pdf",
      "relevance_score": 0.8723
    }
  ],
  "retrieval_time_ms": 12.5,
  "generation_time_ms": 850.3,
  "total_time_ms": 862.8,
  "model_used": "gpt-4o-mini",
  "model_usage": {"accounting": "measured", "input_tokens": 812, "output_tokens": 96, "tokens_charged": 908, "tokens_reserved": 1450, "context_chars": 2400}
}
```

`model_usage.accounting` is `measured` when the provider reported token counts and `reserved` when only the conservative pre-call estimate is known. Demo answers and abstentions have no `model_usage`.

### `GET /papers` — List uploaded papers (token required)
### `DELETE /papers/{paper_id}` — Remove a paper (token required)
### `GET /health` — System health check (public)
### `GET /ready` — Local component readiness, access and budget state (public, no paid calls)

---

## LLM Configuration

### OpenAI (Recommended)
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-your-key
# Real generation stays disabled until accounting is explicit:
export MODEL_CALL_LEDGER_PATH=/var/data/model-calls.sqlite3
export MAX_MODEL_CALLS_PER_DAY=20 MAX_MODEL_CALLS_TOTAL=100
export MAX_MODEL_TOKENS_PER_DAY=40000 MAX_MODEL_TOKENS_TOTAL=200000
```

### Ollama (Free, Local)
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

export LLM_PROVIDER=ollama
export LLM_MODEL=llama3
```

Under the Stage 2c protections this configuration fails closed (`/ready` reports `init_error: token_bound_unavailable` and no model is invoked): Ollama applies the model's Modelfile `TEMPLATE`/`SYSTEM` outside the supplied prompt, so the input cannot be bounded from this client until a model/template-specific bound is implemented.

### Demo Mode (No API Key)
```bash
export LLM_PROVIDER=demo
```

---

## Running Tests

```bash
pip install -r requirements-deploy.txt pytest httpx reportlab "tiktoken==0.14.0"
export LLM_PROVIDER=demo
pytest tests/ -v
```

CI runs deterministic demo/fake-backend tests on Python 3.11, matching Docker, and builds/smoke-tests the demo image with an explicit fake access token. The suite sets its own fake token in `tests/conftest.py`. The token-bound tests use the real pinned tiktoken encoding (downloaded once on the CI runner, baked into the real image). These checks do not establish real-provider compatibility or end-to-end model accuracy.

---

## Architecture

```
Client
  │
  ├── POST /papers/upload
  │     │
  │     ├── PDF Text Extraction (pdfplumber/pypdf)
  │     ├── Recursive Text Chunking (500 chars, 100 overlap)
  │     ├── Embedding Generation (sentence-transformers)
  │     └── ChromaDB Vector Storage
  │
  └── POST /query
        │
        ├── Question Embedding
        ├── Semantic Similarity Search (ChromaDB)
        ├── Context Assembly (top-k chunks)
        ├── LLM Generation (OpenAI/Ollama)
        └── Response with Citations
```

---

## Evaluation

Legacy figures below were previously recorded for 5 ML/CV research papers (8–25 pages each). **They are unverified:** this repository does not include the corpus, labeled questions, outputs or evaluation runner needed to reproduce them. Do not use these figures as evidence of the current deployment's accuracy or latency.

| Metric | Demo Mode | OpenAI (GPT-4o-mini) |
|--------|-----------|---------------------|
| Avg. ingestion time (per paper) | 120 ms | 120 ms |
| Avg. chunking (chunks/paper) | 42 | 42 |
| Avg. retrieval latency | 8 ms | 45 ms |
| Avg. generation latency | 2 ms | 920 ms |
| End-to-end query latency | ~10 ms | ~965 ms |
| Citation accuracy (manual eval, 20 queries) | 60% (keyword only) | 85% |
| Correct source paper identified | 80% | 95% |

**Notes:**
- Retrieval latency scales with corpus size; tested with <250 chunks total.
- Citation accuracy evaluated manually: does the cited page contain the claimed information?
- Demo mode uses keyword matching (no semantic understanding), so accuracy is lower but latency is near-instant.
- OpenAI mode provides reasoning-based answers with substantially better citation quality.

---

## Known Limitations

- **No chunk re-ranking** — Retrieved chunks are ranked by embedding similarity only. Adding a cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) would improve relevance.
- **No hybrid search** — Currently uses pure semantic search. Combining BM25 keyword search with vector search (reciprocal rank fusion) would improve recall for exact-match queries.
- **No cross-paper answer synthesis** — When querying multiple papers, the system retrieves chunks independently but does not synthesize conflicting findings across papers.
- **No hallucination detection** — The LLM may generate plausible but unsupported claims. A verification layer comparing generated claims against retrieved chunks would reduce hallucination.
- **Scanned PDFs not supported** — Text extraction relies on embedded text layers. Scanned/image-only PDFs require OCR preprocessing (e.g., Tesseract) which is not yet integrated.
- **Paper persistence is opt-in** — Set PAPER_STORE_PATH on a persistent local disk; otherwise the registry is disposable. One worker/instance only. Verify platform retention with a restart. Backups and per-user document isolation are separate work.
- **Ledger durability depends on the deployment** — The model-call ledger is a SQLite file. On an ephemeral filesystem a replaced container loses it and starts counting from zero, so durable storage or a provider-enforced backstop is required before paid use.
- **Ceilings are not a price cap** — Call and token allowances bound attempted usage; the money spent depends on the provider's price list and on measured usage, which the demo cannot verify.

---

## Author

**Arjun Ponnaganti**
- MSc Image Analysis & Machine Learning — Uppsala University, Sweden
- 4 peer-reviewed publications including IEEE
- [LinkedIn](https://linkedin.com/in/arjun-ponnaganti)
- [GitHub](https://github.com/AIArjun)

## License

MIT License
