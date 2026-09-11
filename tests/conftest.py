"""Deterministic test environment: a fake shared token and demo provider defaults.

The token below is a test fixture, not a credential. It is set before any
application module is imported so protected routes can be exercised.
"""

import io
import os

TEST_ACCESS_TOKEN = "test-only-fake-demo-access-token-0123456789abcdef"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_ACCESS_TOKEN}"}

os.environ["DEMO_ACCESS_TOKEN"] = TEST_ACCESS_TOKEN
os.environ.setdefault("LLM_PROVIDER", "demo")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ALLOWED_ORIGINS", "")


class CharacterBound:
    """Test-only token bound: one token per character plus the output cap."""

    name = "test/one-token-per-character"

    def count(self, text: str) -> int:
        return len(text)

    def reservation(self, prompt: str, max_output_tokens: int) -> int:
        return len(prompt) + max_output_tokens


def multi_page_pdf(pages: int, lines_per_page: int = 3) -> bytes:
    """A small text PDF with the requested number of pages (reportlab, or pypdf blank pages)."""
    buffer = io.BytesIO()
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(buffer, pagesize=letter)
        for page in range(1, pages + 1):
            for line in range(lines_per_page):
                pdf.drawString(72, 700 - 20 * line, f"Page {page} line {line}: evidence about attention mechanisms.")
            pdf.showPage()
        pdf.save()
    except ImportError:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        writer.write(buffer)
    return buffer.getvalue()


def minimal_pdf() -> bytes:
    """One text page; distinct from the legacy helper in test_api so identities differ."""
    return multi_page_pdf(1, lines_per_page=6)
