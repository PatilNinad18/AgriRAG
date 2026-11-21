"""
tests/conftest.py
------------------
Shared pytest fixtures.

Strategy:
- Mock heavy services (Qdrant, language model, EmbeddingService) so unit tests
  run without GPU, a running Qdrant instance, or model downloads.
- Use FastAPI TestClient for integration-style route tests.
- Fixtures are function-scoped by default (fresh mock per test).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Return a FastAPI TestClient with all heavy services mocked.
    Startup lifespan is bypassed so tests don't need real ML models.
    """
    # Patch services before importing app to avoid side effects
    with (
        patch("app.services.qdrant_service.qdrant_service.connect"),
        patch("app.services.qdrant_service.qdrant_service._client", MagicMock()),
        patch("app.rag.embedder.embedding_service.load"),
        patch("app.rag.embedder.embedding_service._model", MagicMock()),
        patch("app.llm.agriparam.agriparam_service.load"),
        patch("app.llm.agriparam.agriparam_service._model", MagicMock()),
        patch("app.llm.agriparam.agriparam_service._tokenizer", MagicMock()),
    ):
        from app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Sample data ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF file for ingestion tests."""
    try:
        from fpdf import FPDF  # optional dep for test PDF generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Black soil is ideal for cotton cultivation.", ln=True)
        pdf.cell(200, 10, txt="Add nitrogen fertiliser at 120 kg/ha for wheat.", ln=True)
        pdf_path = tmp_path / "test_agri.pdf"
        pdf.output(str(pdf_path))
        return pdf_path
    except ImportError:
        # Fallback: create a minimal valid PDF manually
        pdf_content = b"""%PDF-1.4
1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj
2 0 obj<</Type /Pages /Kids[3 0 R] /Count 1>>endobj
3 0 obj<</Type /Page /Parent 2 0 R /MediaBox[0 0 612 792]
/Contents 4 0 R /Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Black soil is ideal for cotton.) Tj ET
endstream endobj
5 0 obj<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>endobj
xref 0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000370 00000 n
trailer<</Size 6 /Root 1 0 R>>startxref 441 %%EOF"""
        pdf_path = tmp_path / "test_agri.pdf"
        pdf_path.write_bytes(pdf_content)
        return pdf_path


@pytest.fixture
def mock_query_result():
    """A realistic mock RAGResponse for testing the API layer."""
    from app.rag.pipeline import RAGResponse, SourceCitation
    return RAGResponse(
        question="Which crop is suitable for black soil?",
        answer=(
            "Black soil (Regur soil) is ideal for cotton cultivation. "
            "It retains moisture well and is rich in calcium carbonate."
        ),
        sources=[
            SourceCitation(
                source="soil_guide.pdf",
                page=3,
                score=0.85,
                excerpt="Black soil is high in clay content and ideal for cotton.",
            )
        ],
        retrieval_time_ms=45.2,
        generation_time_ms=1823.5,
        total_time_ms=1868.7,
        has_sources=True,
    )
