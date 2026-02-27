# 📚 Research Paper RAG API

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you upload research papers (PDF) and ask questions with **cited answers**.

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

## Author

**Arjun Ponnaganti**
- MSc Image Analysis & Machine Learning — Uppsala University, Sweden
- 4 peer-reviewed publications including IEEE
- [LinkedIn](https://linkedin.com/in/arjun-ponnaganti)
- [GitHub](https://github.com/AIArjun)

## License

MIT License
