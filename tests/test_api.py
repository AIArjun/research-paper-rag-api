"""
Tests for Research Paper RAG API
==================================
Run: pytest tests/ -v
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app, rag
from app.rag_engine import RAGEngine
from app.config import Settings


client = TestClient(app)


# ─── Helper ───
def _create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF for testing."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    # Add text via annotation workaround — use reportlab if available
    buf = io.BytesIO()
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        c = canvas.Canvas(buf, pagesize=letter)
        c.drawString(72, 700, "Abstract: This paper presents a novel approach to machine learning.")
        c.drawString(72, 680, "We propose a deep learning model for image classification.")
        c.drawString(72, 660, "Results show 95% accuracy on the benchmark dataset.")
        c.drawString(72, 640, "Keywords: machine learning, deep learning, image classification")
        c.drawString(72, 600, "1. Introduction")
        c.drawString(72, 580, "Machine learning has revolutionized computer vision in recent years.")
        c.drawString(72, 560, "Our approach uses convolutional neural networks with attention mechanisms.")
        c.drawString(72, 520, "2. Methodology")
        c.drawString(72, 500, "We trained a ResNet-50 model on ImageNet with data augmentation.")
        c.drawString(72, 480, "The model was optimized using Adam optimizer with learning rate 0.001.")
        c.save()
        return buf.getvalue()
    except ImportError:
        writer.write(buf)
        return buf.getvalue()


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "papers_loaded" in data
        assert "total_chunks" in data
        assert "llm_provider" in data
        assert "timestamp" in data


class TestLandingPage:
    def test_root_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "RAG API" in response.text


class TestPaperUpload:
    def test_upload_pdf(self):
        pdf_bytes = _create_minimal_pdf()
        response = client.post(
            "/papers/upload",
            files={"file": ("test_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "paper_id" in data
        assert "chunks" in data
        assert data["chunks"] >= 0
        assert data["filename"] == "test_paper.pdf"

    def test_upload_rejects_non_pdf(self):
        response = client.post(
            "/papers/upload",
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_rejects_wrong_extension(self):
        response = client.post(
            "/papers/upload",
            files={"file": ("test.docx", io.BytesIO(b"data"), "application/pdf")},
        )
        assert response.status_code == 400


class TestListPapers:
    def test_list_papers(self):
        response = client.get("/papers")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestQuery:
    def test_query_without_papers_returns_400(self):
        # Clear all papers first
        for paper_id in list(rag.papers.keys()):
            rag.delete_paper(paper_id)

        response = client.post(
            "/query",
            json={"question": "What is machine learning?"},
        )
        assert response.status_code == 400

    def test_query_with_paper(self):
        # Upload a paper first
        pdf_bytes = _create_minimal_pdf()
        upload = client.post(
            "/papers/upload",
            files={"file": ("ml_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload.status_code == 200

        # Query it
        response = client.post(
            "/query",
            json={"question": "What accuracy did the model achieve?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "retrieval_time_ms" in data
        assert "generation_time_ms" in data
        assert "model_used" in data

    def test_query_response_has_citations(self):
        response = client.post(
            "/query",
            json={"question": "What deep learning approach was used?", "top_k": 3},
        )
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["citations"], list)
            for citation in data["citations"]:
                assert "text" in citation
                assert "paper" in citation
                assert "relevance_score" in citation


class TestDeletePaper:
    def test_delete_nonexistent_paper(self):
        response = client.delete("/papers/nonexistent123")
        assert response.status_code == 404

    def test_delete_existing_paper(self):
        # Upload first
        pdf_bytes = _create_minimal_pdf()
        upload = client.post(
            "/papers/upload",
            files={"file": ("to_delete.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        paper_id = upload.json()["paper_id"]

        # Delete
        response = client.delete(f"/papers/{paper_id}")
        assert response.status_code == 200


class TestRAGEngineUnit:
    def test_chunk_text(self):
        engine = RAGEngine()
        pages = [{"page": 1, "text": "This is a test sentence. " * 50}]
        chunks = engine._chunk_text(pages, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all("text" in c for c in chunks)
        assert all("page" in c for c in chunks)

    def test_get_stats_empty(self):
        engine = RAGEngine()
        stats = engine.get_stats()
        assert stats["papers_loaded"] == 0
        assert stats["total_chunks"] == 0

    def test_demo_retrieve_empty(self):
        engine = RAGEngine()
        results = engine._demo_retrieve("test query", None, 5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_demo_generate_no_citations(self):
        engine = RAGEngine()
        answer = engine._demo_generate("test question", [])
        assert "could not find" in answer.lower()


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.LLM_PROVIDER == "demo"
        assert s.CHUNK_SIZE == 500
        assert s.CHUNK_OVERLAP == 100
        assert s.MAX_FILE_SIZE_MB == 20
        assert s.PORT == 8001
