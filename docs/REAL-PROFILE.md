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

The runtime container is network-isolated. No host port is exposed; checks run through localhost inside the container. A placeholder API key constructs a client but cannot authorize a paid call. The non-existent paper filter must retrieve nothing and return `model_used=not-invoked`. Both PDFs must ingest with expected identities/counts, and re-upload must preserve canonical metadata.

Profiles: 512 MiB / 0.1 CPU and 2 GiB / 1 CPU, with swap disabled. Both are measured and failures retained. This changes CPU as well as memory; results are an isolated CI experiment, not a controlled CPU comparison or a measurement on Render. The readiness deadline is at most 300 seconds per case. The 2 GiB case must pass for overall success.

Artifacts record the exact Git commit, image identity/size, time to ready, app-process RSS/high-water RSS, cgroup current/peak usage, CPU/memory settings, upload counts, readiness/query responses, logs and OOM/exit state. Temporary HTTP and observer processes consume memory inside the same cgroup; total memory therefore includes this measurement overhead. App-process RSS and cgroup totals are different metrics. Large corpora/concurrent traffic require further measurement.

## Startup, diagnosis and deployment boundary

The real image starts one Uvicorn worker and honors `PORT`. Its default build target runs the API; `--target test` runs tests. The measurement overrides the port and tests that one worker is still selected when `WEB_CONCURRENCY` suggests more. The image runs as a non-root user (UID 10001).

Initialization/generation logs now include allowlisted exception type, module, missing dependency and HTTP status where available. They do not stringify exception bodies, keys, URLs or tracebacks. HTTP responses remain categorical. Existing question-prefix logging and generic HTTP error sanitization remain Stage 2c work; this is not a whole-application logging/privacy review.

Current Render settings still build `./Dockerfile`, not this real profile. Its dashboard start-command override wins over a Dockerfile CMD. A later deliberate deployment must select `./Dockerfile.real` and clear the override or explicitly align `PORT` and `--workers 1`, set the appropriate readiness path, and configure protection/secrets directly in Render. Do not change the existing public service merely to run these resource tests.

Disk persistence is separate work: the paper registry and pending recovery metadata remain in process memory. Mounting a disk for Chroma alone does not establish correct cross-restart state.

## Evidence status

The local demo/fake-backend suite passes 100 tests at implementation time. The real Docker build and memory checks must complete in the associated pull request before this milestone is recorded as verified. Refer to its CI artifact for actual numbers and failures; do not infer a hosting recommendation from the configured memory limits alone.

## Primary references

- [uv dependency locking](https://docs.astral.sh/uv/pip/compile/) and [CPU-only PyTorch sources](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) and [Sentence Transformers integration](https://docs.langchain.com/oss/python/integrations/embeddings/sentence_transformers)
- [Embedding model and revision history](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
