# 📚 Research Paper RAG API

A production-oriented **Retrieval-Augmented Generation (RAG)** API with modular architecture, containerization, CI/CD, and structured observability. Upload research papers (PDF) and ask questions with **cited answers**.

Built with **LangChain + FastAPI + ChromaDB + OpenAI/Ollama + Docker**.

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

pip install -r requirements.txt

# Run in demo mode (no API key needed)
uvicorn app.main:app --reload --port 8001

# Or with OpenAI
export OPENAI_API_KEY=sk-your-key
export LLM_PROVIDER=openai
uvicorn app.main:app --reload --port 8001
```

### Option 2: Docker

```bash
docker-compose up --build
```

### Access

- **Landing Page:** http://localhost:8001
- **Swagger Docs:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

---

## API Endpoints

### `POST /papers/upload` — Upload a paper
```bash
curl -X POST http://localhost:8001/papers/upload \
  -F "file=@my_paper.pdf"
```

### `POST /query` — Ask a question
```bash
curl -X POST http://localhost:8001/query \
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
  "model_used": "gpt-4o-mini"
}
```

### `GET /papers` — List uploaded papers
### `DELETE /papers/{paper_id}` — Remove a paper
### `GET /health` — System health check

---

## LLM Configuration

### OpenAI (Recommended)
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-your-key
```

### Ollama (Free, Local)
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

export LLM_PROVIDER=ollama
export LLM_MODEL=llama3
```

### Demo Mode (No API Key)
```bash
export LLM_PROVIDER=demo
```

---

## Running Tests

```bash
pip install pytest httpx reportlab
pytest tests/ -v
```

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

Benchmarked on 5 ML/CV research papers (8–25 pages each) using the demo and OpenAI pipelines:

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
- **No persistent paper metadata** — Paper metadata is stored in memory. Restarting the server loses the paper registry (ChromaDB vectors persist, but the paper list does not).

---

## Author

**Arjun Ponnaganti**
- MSc Image Analysis & Machine Learning — Uppsala University, Sweden
- 4 peer-reviewed publications including IEEE
- [LinkedIn](https://linkedin.com/in/arjun-ponnaganti)
- [GitHub](https://github.com/AIArjun)

## License

MIT License
