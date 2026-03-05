# ─── Research Paper RAG API ───
# Production Docker image
# Author: Arjun Ponnaganti

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/vectorstore /app/uploads

# Pre-download the embedding model during build so it's cached in the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

EXPOSE 8001

# Single worker to stay within memory limits on free-tier hosts (Render, Railway, Fly)
# timeout-keep-alive prevents premature connection drops on slow uploads
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--timeout-keep-alive", "120"]
