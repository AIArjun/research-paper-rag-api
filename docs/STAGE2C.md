# Stage 2c: protecting the shared public-paper demo

Stage 2c protects one shared demonstration of the RAG API: one shared credential, one API worker, one instance, public research papers only. It does not add private client workspaces, durable paper metadata, verified answer quality or production readiness. Every ceiling below is an initial bound chosen for the measured 2 GiB / 1 CPU real profile, not a proven capacity.

## Controls

| Concern | Control | Where |
|---|---|---|
| Access | `Authorization: Bearer <DEMO_ACCESS_TOKEN>` on upload, query, list and delete; constant-time comparison; 401 + `WWW-Authenticate` for missing/wrong tokens; categorical 503 in every mode when no valid token (32–512 printable ASCII characters, no whitespace) is configured; enforced in a pure ASGI middleware before any body is read | `app/protection.py` (`AccessTokenMiddleware`) |
| Public surface | `/`, `/docs`, `/redoc`, `/openapi.json`, `/health`, `/ready` stay open and never render the token; readiness reports `access_configured` truthfully and is 503 until a valid token exists; its model-budget summary is a read-only local ledger query run in a worker thread, and a ledger that becomes unusable after startup makes the API unready | `app/main.py` |
| CORS | Explicit `ALLOWED_ORIGINS` list, default empty; wildcards and malformed entries are `invalid_configuration`; CORS is not authorization: an allowed origin without the token still gets 401 | `app/config.py`, `app/main.py` |
| Request bytes | Per-route caps enforced while receiving: upload requests up to `MAX_FILE_SIZE_MB` MiB + 16 KiB multipart allowance, everything else 32 KiB (a 2000-character question fits in any escaped encoding); an absent or misleading `Content-Length` does not bypass the cap; a declared length over the cap is refused before reading | `RequestBodyLimitMiddleware` |
| Upload ceilings | File bytes (413), page count checked before any page text is extracted (413), chunk count checked before embeddings (413), paper count (409) and total corpus chunks (409) checked before extraction and again atomically before storage; pending-cleanup rows occupy capacity; identical bytes stay idempotent even at capacity; rejected uploads store nothing | `RAGEngine._ingest_paper` |
| Query bounds | Question ≤ 2000 characters, `top_k` ≤ 5, `paper_id` ≤ 128 characters (422 without echoing input); prompt context ≤ `MAX_CONTEXT_CHARS` in rank order | `QueryRequest`, `RAGEngine.query` |
| Concurrency | One admitted upload or delete at a time; `MAX_CONCURRENT_QUERIES` query slots (default 2); queries during an ingestion, or any request when slots are full, answer 429 with `Retry-After: 5`. An upload takes the mutation slot in `AdmissionMiddleware` after authentication and before a single body byte is received (FastAPI would otherwise parse and spool the multipart body before route code runs), so a second simultaneous upload gets 429 without `receive()` ever being called; the same admission is handed to the route and transferred to the worker thread, and it is released by whichever side finishes last (parse failure, oversize body, disconnect or cancellation before dispatch release it immediately; a running worker keeps it until it ends) | `AdmissionMiddleware`, `AdmissionSlot`, `Admission` |
| Event loop | Extraction, chunking, embeddings, retrieval, deletion and provider calls run in worker threads; the registry uses a short-held state lock so `/health` and `/ready` answer during slow work; a separate work lock serializes storage work and is never held during provider calls | `app/rag_engine.py` |
| Provider bounds | `LLM_TIMEOUT_SECONDS` (default 30), `LLM_MAX_OUTPUT_TOKENS` (default 400), zero automatic retries, for both supported providers (OpenAI: `timeout`, `max_retries=0`, `max_tokens`; Ollama: `num_predict`, client `timeout`); a client that cannot accept these bounds fails construction and readiness reports `model_initialization_failed` | `RAGEngine._init_llm` |
| Accounting | Persistent SQLite ledger with daily and lifetime call/token allowances; every attempted call is reserved inside one `BEGIN IMMEDIATE` transaction before the provider is contacted and stays counted if the call fails, times out or hangs; provider-reported usage replaces the reservation when available (`accounting: measured`), otherwise the pre-call bound stands (`accounting: reserved`); abstention and demo answers consume nothing; exhaustion is 429 `budget_exhausted` with `Retry-After` for daily scopes | `app/ledger.py`, `RAGEngine._generate_bounded` |
| Token bound | The reservation is an explicit model-supported upper bound, never a character heuristic: OpenAI models use the exact tiktoken encoding for the configured model (o200k_base for gpt-4o-mini) plus 16 framing tokens for the chat envelope plus the enforced output cap; the request carries exactly the prompt built here, so counting it bounds the billed input. Ollama is an unsupported protected configuration and fails closed (`token_bound_unavailable`, no client constructed, no backend loaded): the server applies the model's Modelfile `TEMPLATE`/`SYSTEM` text outside the supplied prompt ([docs.ollama.com/modelfile](https://docs.ollama.com/modelfile); the `raw` mode of [docs.ollama.com/api/generate](https://docs.ollama.com/api/generate) is not used), so no client-side count can bound the input until a model/template-specific bound exists. A model tiktoken cannot map, an encoding that cannot be loaded, or a missing tiktoken package fails closed the same way (`token_bound_unavailable` / `missing_dependency`). `/ready` and `model_usage.reservation_bound` name the bound in use | `app/tokens.py`, `Dockerfile.real` (baked encoding cache) |
| Fail closed | Real generation is disabled until the ledger path and all four allowances are explicit (`budget_not_configured`); a missing directory, uninitialized, corrupt or incompatible ledger is `ledger_unavailable` and nothing is reset; an existing zero-byte ledger file is refused as truncated history rather than reinitialized, and a new ledger is created only when the path does not exist (schema staged in a private file and published with an exclusive link); malformed numbers never crash import and become `invalid_configuration`; a ledger write failure blocks the call (`503 ledger_unavailable`) | `app/config.py`, `app/ledger.py` |
| Sanitization | Categorical `detail.category` plus `request_id` (also `X-Request-ID`) on every error, including framework-raised 404/405/parse failures; validation errors list field locations and error types only; no `str(e)`, prompts, passages, question text, client filenames, Authorization headers, keys or tracebacks in error responses or logs (successful responses do echo the caller's own question, short citation previews and the stored filename); Stage 2b allowlisted diagnostics retained and extended | `app/main.py`, `app/diagnostics.py` |

## What the ledger does and does not guarantee

- It bounds attempted calls and charged tokens per UTC day and for the lifetime of the ledger file. It is not a verified dollar cap: money follows the provider's price list and measured usage, neither of which the demo knows or enforces. Do not present these ceilings as a spend guarantee.
- Token reservations are computed from a model-supported bound: for OpenAI, the exact tiktoken count of the prompt plus 16 framing tokens plus the output cap (verified against tiktoken 0.14.0: 2000 CJK characters or 2000 emoji tokenize to about 2060 input tokens, where a three-characters-per-token heuristic reserved about 762 for the whole prompt). Provider-reported usage replaces the reservation after the call where the client exposes `usage_metadata` (OpenAI chat models); a measured total larger than the reservation is charged in full and logged as an overshoot so later calls see it. Ollama has no supported bound and is refused (see the Token bound row); it is not silently accounted at a guess.
- A truncated or emptied ledger file cannot replenish the budget: an existing zero-byte file fails closed (`ledger_unavailable`) until an operator restores or deliberately removes it. Only a path that does not exist is initialized, atomically.
- It requires durable backing storage. On an ephemeral filesystem a replaced container starts a fresh ledger at zero, and a process-local counter would be worse. `/ready` exposes `ledger_created_at` so an unexpected reset is visible. After startup the ledger is opened read-write or read-only only, never created: a file that disappears at runtime makes calls fail closed (`503 ledger_unavailable`) and the API unready rather than silently starting a new ledger.
- Before Stage 3 spending, require an explicit small verification budget and either durable enforcement (ledger on a persistent disk) or a verified provider-enforced backstop configured in the provider's own console. This repository does not invent provider hard-limit features or prices.

## Environment variables

Names only; values below are examples and never real secrets.

```sh
DEMO_ACCESS_TOKEN=<python -c "import secrets; print(secrets.token_urlsafe(48))">
ALLOWED_ORIGINS=                      # or https://demo.example,https://another.example
MAX_FILE_SIZE_MB=10  MAX_PDF_PAGES=60  MAX_CHUNKS_PER_PAPER=600  MAX_PAPERS=20  MAX_TOTAL_CHUNKS=3000
MAX_CONTEXT_CHARS=6000  MAX_CONCURRENT_QUERIES=2
LLM_TIMEOUT_SECONDS=30  LLM_MAX_OUTPUT_TOKENS=400
MODEL_CALL_LEDGER_PATH=/var/data/model-calls.sqlite3
MAX_MODEL_CALLS_PER_DAY=20  MAX_MODEL_CALLS_TOTAL=100  MAX_MODEL_TOKENS_PER_DAY=40000  MAX_MODEL_TOKENS_TOTAL=200000
```

Fixed constants: 2000 question characters, `top_k` 5, 32 KiB JSON bodies, 16 KiB multipart allowance, `Retry-After: 5`.

## HTTP contract summary

| Situation | Status | `detail.category` |
|---|---|---|
| No/invalid configured token | 503 | `access_not_configured` |
| Missing or wrong bearer token | 401 (+ `WWW-Authenticate`) | `unauthorized` |
| Body over the route cap | 413 | `request_too_large` |
| File, page or chunk ceiling | 413 | `file_too_large`, `too_many_pages`, `too_many_chunks` |
| Paper count or corpus capacity | 409 | `paper_limit_reached`, `corpus_capacity_reached` |
| Busy (ingestion in progress or slots full) | 429 (+ `Retry-After`) | `busy` |
| Allowance exhausted | 429 (+ `Retry-After` for daily) | `budget_exhausted` |
| Ledger unavailable at call time | 503 | `ledger_unavailable` |
| Backend/config not ready | 503 | `invalid_configuration`, `missing_api_key`, `budget_not_configured`, `ledger_unavailable`, ... |
| Validation | 422 | `invalid_request` (locations and types only) |
| Framework errors (unknown path, wrong method, unparseable body) | 404 / 405 / 400 | `not_found`, `method_not_allowed`, `invalid_request` |
| Provider failure | 502 | `generation_failed` |
| Unexpected failure | 500 | `ingestion_failed`, `query_failed`, `deletion_failed` |

## Verification

Local, at the tested commit (see the pull request for the exact SHA): the full suite passed with the demo profile on Python 3.11 (`pytest tests/`), including the 100 Stage 2b tests with authenticated fixtures and the focused tests covering authentication and invalid configuration, receive-boundary body caps (ASGI-level and against a real uvicorn socket with chunked encoding for both h11 and httptools), each ceiling rejecting before expensive work, duplicate uploads at capacity, admission and cancellation races, responsive `/health` and `/ready` under slow fake ingestion, sanitized errors and logs, provider bounds, and ledger accounting across fresh connections, a separate process, concurrent threads, exhaustion, corruption and restart. The review corrections added reproductions of the three reported defects: a truncated ledger that must not replenish the budget, two simultaneous uploads of which only one may ever read its body, and reservation counts checked against the real pinned tiktoken encoding on diverse Unicode inputs (`tests/test_token_bounds.py`, which fails rather than skips inside the real image where the encodings are baked in).

Docker is not available in the authoring environment, so the demo image smoke test and the real-profile measurement run only in GitHub Actions. The pull request records the actual CI outcomes; a pending run is not a pass. The real-profile workflow keeps the 2 GiB / 1 CPU embedding and Chroma check with limits enabled, the runtime network disabled and a placeholder provider key; it now also exercises the protected API with a fixed fake token and a one-call allowance, and asserts that no call is consumed. No real generation, semantic relevance or citation accuracy has been verified.

## Deployment checklist (not performed in this stage)

1. Keep one Uvicorn worker and one instance. The admission slots, registry and ledger transactions assume a single process on one volume; `WEB_CONCURRENCY` must not raise the worker count (the real image pins `--workers 1`).
2. Choose the compute class from the Stage 2b evidence: the measured workload does not fit 512 MiB; the 2 GiB / 1 CPU class passed two-paper ingestion. Upgrading the Render service is a separate, deliberate decision.
3. Enter secrets directly in the Render dashboard, never in the repository: `DEMO_ACCESS_TOKEN` (generated, at least 32 characters) and the provider key. Rotate the token by replacing the variable and redeploying.
4. Attach a persistent disk and point `MODEL_CALL_LEDGER_PATH` and `VECTORSTORE_PATH` at it; verify after a restart that `/ready` still reports the earlier `ledger_created_at` and counts.
5. Set all four allowances to the agreed small verification budget; leave them at zero (generation disabled) until that budget exists.
6. Align the service with the real profile: `./Dockerfile.real`, clear or match the start-command override (`--workers 1`, `PORT`), readiness path `/ready`.
7. Set `ALLOWED_ORIGINS` only when a browser client exists; leave it empty otherwise.
8. After deploy, confirm: `/ready` is 200 with `access_configured: true` and `model_budget.state: ok`; an unauthenticated `/papers` is 401; an oversized upload is 413; the token does not appear in `/openapi.json`, `/ready` or logs.
9. Run a single controlled query against a known public paper only after steps 3–5, and record the measured `model_usage`.

## Rollback reference

- `main` remains `61d77430ed75f53fa04b535149875edf77ea528c` and auto-deploys to Render; nothing in this stage changes it. The existing deployment (`main`/`61d7743`, `./Dockerfile`, command override) is the rollback target.
- The PR stack is #2 foundation → `main`, #3 readiness → foundation, #4 real profile → readiness, and this stage → `codex/rag-real-profile`. Reverting Stage 2c means closing its pull request or resetting `codex/rag-protection` to `2b1124823623fedf5f268b7108539749b8dd8e4d`, the Stage 2b head.
- A deployed protected demo can be disabled without a code change by removing `DEMO_ACCESS_TOKEN` (all protected routes become 503) or by setting any model-call allowance to zero (generation becomes `budget_not_configured`).

## Remaining Stage 3 needs

- A concrete, explicit verification budget (calls and tokens per day and in total) agreed before any paid call.
- The service compute upgrade to the measured 2 GB class, decided and purchased separately.
- A durable ledger location (persistent disk) and/or a verified provider-enforced spending backstop; confirmation that the ledger survives a deploy.
- Secrets entered directly in Render (`DEMO_ACCESS_TOKEN`, provider key); none exist in this repository.
- Configuration and start-command alignment for `Dockerfile.real` (`PORT`, `--workers 1`, readiness path, limits).
- Controlled answer and citation verification with real generation on known public papers, recording measured usage.
- Later milestones, not Stage 3 prerequisites: a persistent paper registry with recovery, and an upload/chat interface.
