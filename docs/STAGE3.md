# Stage 3: deploying and verifying the protected demo

Stage 3 is one controlled live demonstration of the protected API with two public papers on the existing Render service. It is not a frontend, not durable paper metadata and not a production deployment. Nothing here changes `main`'s behavior until the service is deliberately pointed at this branch; secrets, the deploy itself and the inspection of live evidence stay with the operator.

## Decision: the ledger lives on the disk, the corpus does not

- Only the model-call ledger persists. `MODEL_CALL_LEDGER_PATH` points into the mounted disk. `VECTORSTORE_PATH` stays at the image default `/app/vectorstore`, which is ephemeral on Render, and uploads stay in memory. A restart or deploy therefore clears the corpus and the two papers must be re-uploaded; ledger identity (`ledger_created_at`) and counts must survive. Persisting Chroma alone would leave a vector store that disagrees with the in-memory paper registry after a restart, so the Stage 2c checklist line that pointed `VECTORSTORE_PATH` at the disk is superseded.
- The runtime user cannot rely on the mount's ownership. Render's disk documentation states which paths can be mounted and that a disk removes zero-downtime deploys, but says nothing about the owner or mode of the mount point, and the API runs as UID 10001 (`app`). The ledger refuses to start without a writable directory (`ledger_unavailable`, API unready), so an unwritable mount would fail the deploy. `Dockerfile.real`'s default `runtime` target now starts `scripts/real_entrypoint.sh` as root; the script creates the ledger directory, hands that directory and any existing ledger/SQLite sidecar files to `app` (never a recursive change over the whole mount), and then replaces itself with the server command under `setpriv --reuid=app --regid=app --init-groups --inh-caps=-all --bounding-set=-all --no-new-privs`. The server process is UID 10001 with an empty effective, inheritable, ambient and bounding capability set and cannot regain privileges, which is stricter than the previous plain `USER app`. Started as a non-root user (the `test` target, or a platform that forbids root) the script is a plain `exec`. The `--target test` image still runs the suite as `app`. The real-profile measurement now mounts a root-owned `0755` tmpfs at `/var/data`, puts the ledger under `/var/data/ledger/`, and asserts that PID 1 has UID 10001 and `CapEff` zero, so CI shows the mechanism working against a hostile mount.

## Render settings

All values are entered in the Render dashboard; none are committed. Names are exact.

| Setting | Value |
|---|---|
| Branch | `codex/rag-stage3`; deploy the exact commit reported in the pull request, and confirm it in the deploy's Events entry |
| Auto-Deploy | Off (confirmed); every Stage 3 deploy is a manual deploy |
| Dockerfile Path | `./Dockerfile.real` |
| Docker Build Context Directory | `.` |
| Docker Command | empty. Clear the existing `uvicorn ... --port 8001` override so the image `ENTRYPOINT` and `CMD` run (`--workers 1`, `PORT` honored). An override would bypass neither the entrypoint nor `USER`, but it would pin the wrong port and lose the one-worker guarantee if edited later |
| Health Check Path | `/ready` (any `2xx`/`3xx` is healthy; `/ready` answers 503 until access, ledger and backend are all ready) |
| Instance Type | the already purchased 2 GB / 1 CPU class |
| Instances | 1 (a disk forbids more, and the admission slots and ledger assume one process) |
| Disk | already added: 1 GB, mount path `/var/data` |
| Environment | see below; leave everything else unset |

Environment variables to create:

```sh
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<provider key, dashboard only>
DEMO_ACCESS_TOKEN=<generated 32-512 printable ASCII characters, dashboard only>
MODEL_CALL_LEDGER_PATH=/var/data/ledger/model-calls.sqlite3
MAX_MODEL_CALLS_PER_DAY=20
MAX_MODEL_CALLS_TOTAL=20
MAX_MODEL_TOKENS_PER_DAY=100000
MAX_MODEL_TOKENS_TOTAL=100000
LLM_TIMEOUT_SECONDS=30
LLM_MAX_OUTPUT_TOKENS=400
```

Do not create `ALLOWED_ORIGINS` (unset means the empty list: no browser origin is allowed, which is correct without a frontend), `VECTORSTORE_PATH` (must stay ephemeral), `PORT` (Render supplies it and the image binds to it), `WEB_CONCURRENCY` (the command pins one worker regardless), or any embedding/offline variable (baked into the image). Retries are fixed at zero in code. A ledger subdirectory rather than the mount root keeps the entrypoint's ownership change off the mount point itself.

Sizing note, not a price claim: one call reserves the exact tiktoken count of the prompt (at most `MAX_CONTEXT_CHARS` 6000 characters of context plus the question and instructions) plus 16 framing tokens plus the 400-token output cap, so a call stays a few thousand tokens and 20 calls stay well inside 100000. Money follows the provider's price list and measured usage; nothing here enforces a dollar amount.

Expected `/ready` after the deploy (record the whole body):

- `ready: true`, `access_configured: true`, `configured_provider: openai`, `configured_model: gpt-4o-mini`, `effective_retrieval: chroma`, `effective_generation: openai`.
- `model_budget.state: ok`, `configured: true`, `token_bound: tiktoken/o200k_base`, `usage.calls_total: 0` on a fresh ledger, and a `usage.ledger_created_at` value: write it down, it is the ledger's identity for the restart check.
- `provider_connection_verified: false` is normal: readiness never contacts the provider, so a model name there is not evidence that a paid call works.

If `/ready` is 503, `init_error` says why: `ledger_unavailable` (mount or entrypoint problem; check the deploy logs for the entrypoint), `budget_not_configured` (an allowance is missing or zero), `missing_api_key`, `token_bound_unavailable` (model not mappable), `invalid_configuration` (a malformed number). `access_configured: false` means the token is missing or malformed.

## Verification helper

`scripts/verify_live_demo.py` needs only Python 3.11 and the standard library. The access token is read from `DEMO_ACCESS_TOKEN` in the environment, never from an argument, and the helper refuses to write any evidence file in which the token would appear. Keep the token out of shell history:

```sh
read -rs DEMO_ACCESS_TOKEN && export DEMO_ACCESS_TOKEN   # paste, Enter; nothing is echoed
```

Safe run (no paid call, nothing deleted): readiness before, missing and wrong token must be 401, authenticated listing, download and digest-check the two public PDFs, upload them, one empty-filter query that must abstain with `model_used: not-invoked`, readiness after with an unchanged call count.

```sh
python scripts/verify_live_demo.py --base-url https://research-paper-rag-api.onrender.com \
  --output stage3-evidence --fixture-dir stage3-fixtures --download-fixtures
```

Live run (explicit opt-in, at most 3 calls here and never more than 5 per invocation, one attempt per question, no retry, stops at the first non-200):

```sh
python scripts/verify_live_demo.py --base-url https://research-paper-rag-api.onrender.com \
  --output stage3-evidence --skip-uploads --live --max-live-calls 3
```

Each run writes `stage3-evidence/stage3-evidence-<UTC>.json` (every response, request ids, ledger snapshots before and after each call) and a `.md` rendering. Exit code 0 means every recorded check passed: the safe checks including declared page/chunk counts of the uploads, the ledger comparison when requested (evaluated on the first `/ready` snapshot, before any upload or paid call), and with `--live` every call answered 200 from the configured model with `measured` usage that the ledger delta confirms (`checks.live_phase_passed`, per-call `accounted` and `failure_reasons`). 1 means a check failed or a precondition aborted the run; 2 means the token was missing or would have leaked. A failed live call is never retried, stops the live phase and fails the run; semantic answer quality and citation support remain a manual judgment. Uploads use the fixture manifest of `scripts/measure_real_profile.py`: `attention-is-all-you-need.pdf` from https://arxiv.org/pdf/1706.03762 (SHA-256 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`, 15 pages, 110 chunks) and `retrieval-augmented-generation.pdf` from https://arxiv.org/pdf/2005.11401 (SHA-256 `23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233`, 19 pages, 188 chunks). A digest mismatch aborts before any upload. The control behavior (no generation call without `--live`, the cap, no retry, token handling, digest refusal, ledger comparison) is covered by `tests/test_verify_live_demo.py` against a fake server; no credential is used in tests.

## Evidence rubric

The default live questions are specific enough to check against the PDFs. "Page" in a citation is the 1-based physical page of the uploaded PDF, not a printed page label.

| Id | Question | Expected support (verify in the PDF) |
|---|---|---|
| `transformer-encoder-layers` | How many identical layers does the Transformer encoder stack use, and what are the two sub-layers in each encoder layer? | Attention Is All You Need, Section 3.1: N = 6; multi-head self-attention and a position-wise feed-forward network |
| `transformer-bleu-en-de` | What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German translation task? | Attention Is All You Need, abstract and Table 2: 28.4 BLEU |
| `rag-retriever-generator` | In the RAG paper, which pre-trained models are used as the retriever and as the generator? | Retrieval-Augmented Generation, Section 2: a DPR bi-encoder retriever and a BART-large generator |
| (always run) | any question with `paper_id` set to a value that cannot exist | 200, `citations: []`, `model_used: not-invoked`, `model_usage: null`, ledger unchanged |

For each live answer, record and judge:

1. `model_used` is `gpt-4o-mini` and `model_usage.accounting` is `measured` with non-null `input_tokens` and `output_tokens`; `reservation_bound` is `tiktoken/o200k_base`; `tokens_charged` is the measured total and is at most `tokens_reserved` unless an overshoot warning appears in the logs.
2. The ledger delta between the bracketing `/ready` snapshots is exactly one call and `tokens_charged` tokens. That delta, not the model name, is the evidence that a provider call happened and was accounted for.
3. Every citation names one of the two uploaded filenames and a page; open the PDF at that page and confirm the preview text is there and supports the claim the answer attributes to it. Previews are truncated to 300 characters, so the full passage may continue beyond the preview.
4. The answer's factual claims match the expected support; note any claim the cited pages do not contain (an unsupported claim is a finding, not a formatting issue).
5. Timing: `generation_time_ms` must be under the 30-second provider timeout; a 502 `generation_failed` or timeout still consumes one reservation and is recorded as such.

Known limitations to state with the evidence: three questions are a demonstration, not a quality measurement; relevance scores are Chroma's and were not calibrated; the 400-token output cap can truncate an answer; the answer text is free-form and cites what the model chose to cite; nothing here verifies the provider's billing.

## Restart verification sequence

1. Safe run, then live run as above. Keep the JSON of the last run (call it A) and note `ledger_created_at`, `calls_total` and `tokens_charged_total` from its `ready_final`.
2. In Render: Manual Deploy → Restart service. Render documents a restart as a manual deploy of the same commit and configuration; because a disk is attached the running instance is stopped before the new one starts, so expect a short outage rather than a zero-downtime swap.
3. When `/ready` is 200 again, run the normal safe run with the comparison (uploads included; the comparison is evaluated on the first `/ready` snapshot, before any upload):

   ```sh
   python scripts/verify_live_demo.py --base-url https://research-paper-rag-api.onrender.com \
     --output stage3-evidence --fixture-dir stage3-fixtures --compare-ledger stage3-evidence/<A>.json
   ```

   Expected: `checks.ledger_persisted: true` (same `ledger_created_at`, counts not lower), `papers_before` is `[]` because the corpus is disposable, both papers ingest again with 110 and 188 chunks, and every check passes. Do not use `--skip-uploads` for this step: on the empty corpus after a restart the abstention query is correctly refused with `400 empty_corpus`, so that variant exits 1 on `abstention_not_invoked` even though `ledger_persisted` is `true`. A different `ledger_created_at` means a fresh ledger was created somewhere else (wrong path, wrong mount, or the disk was not attached) and the budget history was lost; stop and investigate before any further paid call.
4. Optionally one more live call, then a final `/ready` showing counts that only grew.

Observed on the deployed `main` `1d49927` service: the Render restart kept the ledger identity (same `ledger_created_at`, 3 calls, 2487 tokens, zero unsettled) and cleared the corpus, as designed.

## Rollback and disable, and what is not assumed

- A deploy whose `/ready` never turns 200 is cancelled by Render after its health-check window. With a disk attached the previous instance was already stopped, so do not assume the previous version keeps serving or comes back on its own; the fetched documentation describes traffic continuing to existing instances only for zero-downtime deploys, which a disk disables. Treat a failed deploy as an outage that ends with a rollback or a fixed deploy, and confirm what the dashboard actually did.
- Rollback: Render's rollback reuses a previous deploy's build artifact (if still retained) with that deploy's environment variables, keeps the current disk and compute plan, and turns auto-deploy off. Whether the rollback target's Docker command override is restored is not stated in the fetched documentation. The rollback target is the demo deploy of `main`/`61d7743` (`./Dockerfile`, demo mode, no papers). This has not been tested in Stage 3 unless the operator performs it; record the outcome either way.
- Disable without a deploy of new code: set `MAX_MODEL_CALLS_PER_DAY=0` and choose "Save and deploy" (existing build, new environment): the API becomes unready with `budget_not_configured` and every protected route answers 503, so no paid call is possible. Removing `DEMO_ACCESS_TOKEN` the same way makes every protected route 503 `access_not_configured`. Both are redeploys and therefore brief outages on a disk service.
- Emergency stop: suspending the service stops serving immediately. What suspension does to the disk and the URL is not covered by the pages fetched for this document; the operator should check the dashboard's suspend dialog before relying on it.
- The ledger itself is the backstop for paid calls: 20 calls and 100000 tokens for the day and for the file's lifetime, enforced before the provider is contacted, with failed calls still counted. Increasing an allowance later is a deliberate environment change.

## Evidence status

Filled in by the operator from actual runs. Until then Stage 3 is prepared, not complete: no live answer, citation or restart claim is made here.

## Retrieval defect found by the live verification, and the extraction fix

Live evidence at `main` `1d49927` (kept as recorded): three accounted `gpt-4o-mini` calls; all 15 citation previews matched their source chunks, pages and hashes exactly, so the defect is retrieval and entailment, not page coordinates; the attention-scaling question was answered correctly from page 4, but "What distinguishes RAG-Sequence from RAG-Token in how retrieved documents are used?" retrieved pages 17, 8, 7, 17, 1 (never page 3, where Section 2.1 defines both) and the answer was reversed, and the retriever/generator question retrieved pages 1, 5, 3, 7, 9 and answered generically. Citation previews showed run-together words such as `memoryisapre-trainedseq2seqmodel`.

Diagnosis (offline, `docs/evidence/stage3-retrieval-offline.json`): pdfplumber's `extract_text()` groups characters into words with a fixed 3 pt tolerance. Both fixtures are pdfTeX output that positions words by offset without space glyphs, and their 10 pt body text has an inter-word gap of about 2.5 pt, so nearly every line was extracted as one word: 750 tokens longer than 25 characters in the RAG paper and 407 in the Transformer paper, against 74 and 1 from pypdf on the same pages. The embeddings were computed on that noise. Rebuilding the index offline with the deployed extraction reproduces the live pages exactly (17, 8, 7, 17, 1 and 1, 5, 3, 7, 9).

Fix (`app/rag_engine.py`, `WORD_GAP_RATIO = 0.15`): pdfplumber's `x_tolerance_ratio` scales the word tolerance with the glyph size, 1.5 pt at 10 pt. Kerning gaps inside a word are near 0 em and the narrowest justified inter-word gap of a Times-style font is about 0.17 em, so 0.15 em separates words without splitting them; on the fixtures it leaves 73 and 1 long tokens (the remainder are bibliography lines that pypdf joins the same way). No dependency, provider or protection changed; the pypdf fallback is untouched. `tests/test_pdf_word_spacing.py` reproduces the layout with reportlab (words drawn 2.5 pt apart with no space glyphs), shows the default tolerance joining them, and checks the engine keeps the boundaries while a tightly kerned word stays whole.

Expected chunk counts change because the corrected text is longer: Attention Is All You Need 97 → 110 chunks, Retrieval-Augmented Generation 171 → 188 chunks (pages unchanged, 15 and 19; both far inside the 600-per-paper and 3000-total ceilings). The fixture manifests in `scripts/measure_real_profile.py` and `scripts/verify_live_demo.py` carry the new counts.

Offline reproduction of the three live requests exactly as sent (top-5, paper-filtered, `docs/evidence/stage3-retrieval-offline.json`). "Defining sentences" means the sentence itself is inside the supplied 6000-character context, not merely its page:

| Live request (filter) | Before, `main` `1d49927` | After, 500/100 (this change) | After, 1000/200 (comparison only) |
|---|---|---|---|
| 1. "Why does scaled dot-product attention divide by the square root of d_k?" (Transformer) | 4, 4, 4, 4, 7; run-together text, no defining sentence | 4, 4, 4, 4, 4; both defining sentences present | 4, 4, 4, 4, 9; both present, lower scores |
| 2. "What distinguishes RAG-Sequence from RAG-Token in how retrieved documents are used?" (RAG) | 17, 8, 7, 17, 1; none | 8, 7, **3**, 8, **3**; both defining sentences present ("uses the same retrieved document to generate the complete sequence"; "draw a different latent document for each target token") | 8, 3, 7, 3, 2; RAG-Token sentence present, RAG-Sequence sentence lost at a chunk boundary |
| 3. "In the RAG paper, which pre-trained models are used as the retriever and as the generator?" (RAG) | 1, 5, 3, 7, 9; none | 6, 7, 10, 17, 9; none (page 2 at rank 6, page 3 at ranks 9 and 11) | 2, 17, 9, 3, 18; "BERT" present but neither "pre-trained bi-encoder from DPR" nor "we use BART-large" |

The offline "before" column matches the live pages exactly, which is what makes the diagnosis a reproduction rather than an inference. Extra rubric questions run without a paper filter are recorded separately in the evidence file and are not the live requests.

Bounded chunking comparison (requested after the fix): larger page-local chunks with the existing `CHUNK_SIZE`/`CHUNK_OVERLAP` configuration (1000/200) would give 54 and 93 chunks, but they do not put the defining DPR or BART-large sentences into the context for request 3, they drop the RAG-Sequence defining sentence for request 2, and every top score falls (the embedding model truncates at 256 word pieces, so a 1000-character chunk is only partly embedded). The defaults therefore stay at 500/100 and no chunking change is included.

Honest limits and the remaining blocker: the fix removes the root cause; requests 1 and 2 now have their defining sentences in the supplied context, so a live repeat should use those two (plus the empty-filter abstention), not request 3. Request 3 stays a known missing-context case: its defining Section 2.2/2.3 sentences are not retrieved by this embedding model under either chunking, because the Section 2.2/2.3 chunks mix the "DPR"/"BART" sentences with formula fragments and `(cid:NN)` glyph artifacts under the fixed 500-character chunking; a paragraph-aware chunker or artifact stripping is a separate, larger change and was not attempted here. No relevance threshold for abstention was added: the offline top scores of wrong and right pages overlap (0.41 versus 0.44), so a threshold would need calibration on more questions. Answer quality after the fix is not verified until the operator re-runs the live rubric; a redeploy is recommended only after the real-profile CI shows the new chunk counts.
