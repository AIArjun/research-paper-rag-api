# ─── Research Paper RAG API ───
# Lightweight deployment for Render free tier (512MB)
# Uses demo mode — no heavy ML models loaded at startup
# For full deployment with embeddings, use requirements.txt instead
# Author: Arjun Ponnaganti

FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

RUN mkdir -p /app/vectorstore /app/uploads

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
