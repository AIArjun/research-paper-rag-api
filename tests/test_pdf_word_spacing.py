"""PDF word boundaries survive extraction when the PDF positions words without space glyphs.

pdfTeX and similar producers place each word by its x offset and emit no space
character; with pdfplumber's fixed default tolerance the gap of a 10 pt body
font is narrower than the tolerance, so a whole line came out as one word and
was embedded as noise (observed live on both Stage 3 fixtures). The PDF below
reproduces that layout with reportlab by drawing each word at its own
position, 2.5 pt apart, exactly like the fixtures' body text.
"""

import io

import pytest

from app.rag_engine import RAGEngine, WORD_GAP_RATIO

SENTENCE = "The RAG-Sequence model uses the same retrieved document to generate the complete sequence."


def positional_words_pdf(sentence: str, font: str = "Times-Roman", size: float = 10, gap: float = 2.5) -> bytes:
    reportlab_pdfgen = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.pdfbase.pdfmetrics import stringWidth

    buffer = io.BytesIO()
    pdf = reportlab_pdfgen.Canvas(buffer, pagesize=(612, 792))
    pdf.setFont(font, size)
    x = 72
    for word in sentence.split():
        pdf.drawString(x, 700, word)
        x += stringWidth(word, font, size) + gap
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_the_fixed_default_tolerance_reproduces_the_run_together_words():
    pdfplumber = pytest.importorskip("pdfplumber")
    with pdfplumber.open(io.BytesIO(positional_words_pdf(SENTENCE))) as pdf:
        default = pdf.pages[0].extract_text() or ""
    # The defect: pdfplumber's default joins the line into one token.
    assert "RAG-Sequencemodelusesthesame" in default.replace(" ", "")
    assert " " not in default.strip()


def test_extraction_keeps_word_boundaries_of_positionally_spaced_text():
    engine = RAGEngine()
    pages = engine._extract_pdf(positional_words_pdf(SENTENCE))
    assert [page["page"] for page in pages] == [1]
    words = pages[0]["text"].split()
    assert words == SENTENCE.split()
    assert max(len(word) for word in words) < 25


def test_a_wide_gap_is_still_a_word_boundary_and_kerning_is_not():
    engine = RAGEngine()
    generous = engine._extract_pdf(positional_words_pdf(SENTENCE, font="Helvetica", size=12, gap=3.3))
    assert generous[0]["text"].split() == SENTENCE.split()
    # A slightly negative offset (tight kerning) must not split characters inside a word.
    tight = engine._extract_pdf(positional_words_pdf("Transformer", size=10, gap=-0.2))
    assert tight[0]["text"].strip() == "Transformer"


def test_ratio_is_below_the_narrowest_justified_word_gap():
    # A justified Times line may shrink its word gap to about 0.17 em; kerning stays near 0 em.
    assert 0 < WORD_GAP_RATIO < 0.17
