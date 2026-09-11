# Real CPU profile and bounded resource measurement

Stage 2b builds the actual local embedding and Chroma stack. A successful measurement does not establish real-provider generation, retrieval relevance, citation accuracy, authenticated access, durability or production readiness.

## Reproducible inputs

- Linux x86_64, CPython 3.11.16: official Python image pinned by manifest digest in `Dockerfile.real`.
- `requirements-real.in` lists direct package pins. `requirements-real.lock.txt` resolves transitive dependencies with SHA-256 hashes; the PyTorch wheel is explicitly CPU-only from the official CPU index. GPU/CUDA packages are not needed.
- `requirements-real-test.in` and its lock add test tools while constraining the runtime package versions. `requirements.txt` is a compatibility entry point to the real lock; it is not portable to other Python/OS/CPU combinations. `requirements-deploy.txt` and the existing `Dockerfile` remain the demo profile.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The build fetches tokenizer/config files and `model.safetensors` into `/opt/models/all-MiniLM-L6-v2`. Runtime uses this local path with offline flags. The embedding batch size is 32.
- Modular integrations: `langchain-huggingface`, `langchain-chroma`, `langchain-openai` and `langchain-ollama`. Explicit demo mode does not load these heavy backends.

Regenerate the Linux locks with uv 0.12.13 (from the repository root):

```sh
uv pip compile requirements-real.in --python-platform x86_64-manylinux_2_28 --python-version 3.11 --generate-hashes --no-annotate --output-file requirements-real.lock.txt
uv pip compile requirements-real-test.in --constraint requirements-real.lock.txt --python-platform x86_64-manylinux_2_28 --python-version 3.11 --generate-hashes --no-annotate --output-file requirements-real-test.lock.txt
```

Review pins and regenerate hashes deliberately for upgrades. The container uses `pip --require-hashes --only-binary=:all:` and `pip check`; a dependency resolution alone is not proof of import/runtime compatibility.

## Run the same checks as CI

Docker must be running. The build downloads dependencies and the pinned model. The measurement host downloads two public PDFs and verifies their recorded SHA-256 digests. A changed upstream PDF is a fixture failure, not silently accepted data.

```sh
docker build -f Dockerfile.real --target runtime -t rag-real-profile:measurement .
docker build -f Dockerfile.real --target test -t rag-real-profile:test .
docker run --rm --network none --memory 2g --memory-swap 2g --cpus 1 -e LLM_PROVIDER=demo -e OPENAI_API_KEY= rag-real-profile:test
python scripts/measure_real_profile.py --skip-build --image rag-real-profile:measurement --output measurement-artifacts
```

The runtime container is network-isolated. No host port is exposed; checks run through localhost inside the container. A placeholder API key constructs a client but cannot authorize a paid call. Since Stage 2c the container also receives a fixed fake `DEMO_ACCESS_TOKEN`, an empty `ALLOWED_ORIGINS`, a throwaway ledger path under `/tmp` and a one-call allowance, so the measurement runs through the protected API: unauthenticated and wrong-token requests must answer 401, readiness must report the access token and accounting as configured without exposing the token, the non-existent paper filter must retrieve nothing and return `model_used=not-invoked`, and the ledger must still show zero calls afterwards. Both PDFs must ingest with expected identities/counts within the default limits (10 MiB, 60 pages, 600 chunks per paper), and re-upload must preserve canonical metadata.

Profiles: 512 MiB / 0.1 CPU and 2 GiB / 1 CPU, with swap disabled. Both are measured and failures retained. This changes CPU as well as memory; results are an isolated CI experiment, not a controlled CPU comparison or a measurement on Render. The readiness deadline is at most 300 seconds per case. The 2 GiB case must pass for overall success.

Artifacts record the exact Git commit, image identity/size, time to ready, app-process RSS/high-water RSS, cgroup current/peak usage, CPU/memory settings, upload counts, readiness/query responses, logs and OOM/exit state. Temporary HTTP and observer processes consume memory inside the same cgroup; total memory therefore includes this measurement overhead. App-process RSS and cgroup totals are different metrics. Large corpora/concurrent traffic require further measurement.

## Startup, diagnosis and deployment boundary

The real image starts one Uvicorn worker and honors `PORT`. Its default build target runs the API; `--target test` runs tests. The measurement overrides the port and tests that one worker is still selected when `WEB_CONCURRENCY` suggests more. The image runs as a non-root user (UID 10001).

Initialization/generation logs now include allowlisted exception type, module, missing dependency and HTTP status where available. They do not stringify exception bodies, keys, URLs or tracebacks. HTTP responses remain categorical. Stage 2c replaced question-prefix logging and raw exception responses with categorical errors and request correlation (see [STAGE2C.md](STAGE2C.md)); this is still not a whole-application logging/privacy review.

Current Render settings still build `./Dockerfile`, not this real profile. Its dashboard start-command override wins over a Dockerfile CMD. A later deliberate deployment must select `./Dockerfile.real` and clear the override or explicitly align `PORT` and `--workers 1`, set the appropriate readiness path, and configure protection/secrets directly in Render. Do not change the existing public service merely to run these resource tests.

Disk persistence is separate work: the paper registry and pending recovery metadata remain in process memory. Mounting a disk for Chroma alone does not establish correct cross-restart state.

## Evidence status

Measured 11 September 2026 at code commit `1512585932ee952428eeb629286a16e01aa448c5`. Both the demo CI and real-profile CI passed. The real dependency image ran 100 tests successfully (one existing AnyIO warning).

| Configuration | Result | Time to local readiness | Workload evidence |
|---|---|---|---|
| 512 MiB / 0.1 CPU, no swap | Failed: OOM kill / exit 137 during first upload | 130.00 s | Startup reached ready, but no upload completed. Last observed pre-upload cgroup peak was 457.32 MiB; final peak after the kill was unavailable. |
| 2 GiB / 1 CPU, no swap | Passed | 11.47 s | Both papers: 97 + 171 chunks; duplicate/list checks; real empty-filter retrieval, no model invocation; no OOM. |

For the passing case, cgroup idle usage was 452.41 MiB and observed peak was 919.63 MiB. App-process RSS was 573.55 MiB at idle; high-water RSS reached 1,130.31 MiB. These metrics have different accounting and must not be treated as interchangeable. Container image size was 2,052,790,318 bytes (about 1.91 GiB).

The 512 MiB failure establishes that this tested workload does not fit that limit. The 2 GiB result supports using a 2 GB hosting class for the protected demonstration; it does not establish capacity for concurrent clients or larger corpora. Do not spend on a 512 MB paid plan to address this observed memory failure.

Permanent raw evidence: [stage2b-1512585.json](evidence/stage2b-1512585.json). [Real-profile CI](https://github.com/AIArjun/research-paper-rag-api/actions/runs/34576253008) includes build, test and container logs; [demo CI](https://github.com/AIArjun/research-paper-rag-api/actions/runs/34576252970) passed separately. Both refer to the same measured code commit. Subsequent documentation commits do not imply a new measurement.

## Primary references

- [uv dependency locking](https://docs.astral.sh/uv/pip/compile/) and [CPU-only PyTorch sources](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) and [Sentence Transformers integration](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers)
- [Embedding model and revision history](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
