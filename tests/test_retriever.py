"""Unit tests for app/rag/retriever.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.retriever import MAX_CONTEXT_CHARS, Retriever, RetrievedDocument


@pytest.fixture
def retriever():
    return Retriever()


@pytest.fixture
def sample_hits():
    return [
        {
            "text": "Black soil retains moisture and is ideal for cotton.",
            "source": "soil_guide.pdf",
            "page": 3,
            "chunk_index": 0,
            "score": 0.87,
        },
        {
            "text": "Regur soil has high clay content and nutrient richness.",
            "source": "soil_guide.pdf",
            "page": 4,
            "chunk_index": 1,
            "score": 0.72,
        },
    ]


class TestRetriever:
    def test_returns_no_results_gracefully(self, retriever):
        with (
            patch("app.rag.retriever.embedding_service.encode_query", return_value=[0.1] * 384),
            patch("app.rag.retriever.qdrant_service.search", return_value=[]),
        ):
            result = retriever.retrieve("what is black soil?")
            assert result.has_results is False
            assert result.documents == []
            assert result.context == ""

    def test_returns_documents_on_hits(self, retriever, sample_hits):
        with (
            patch("app.rag.retriever.embedding_service.encode_query", return_value=[0.1] * 384),
            patch("app.rag.retriever.qdrant_service.search", return_value=sample_hits),
        ):
            result = retriever.retrieve("Which crop for black soil?")
            assert result.has_results is True
            assert len(result.documents) == 2
            assert all(isinstance(d, RetrievedDocument) for d in result.documents)

    def test_context_contains_source_headers(self, retriever, sample_hits):
        with (
            patch("app.rag.retriever.embedding_service.encode_query", return_value=[0.1] * 384),
            patch("app.rag.retriever.qdrant_service.search", return_value=sample_hits),
        ):
            result = retriever.retrieve("soil type for cotton")
            assert "[1]" in result.context
            assert "soil_guide.pdf" in result.context
            assert "Score:" in result.context

    def test_context_within_max_chars(self, retriever):
        """Context must not exceed MAX_CONTEXT_CHARS even for many large chunks."""
        big_hits = [
            {
                "text": "x" * 1000,
                "source": f"doc{i}.pdf",
                "page": i,
                "chunk_index": 0,
                "score": 0.8,
            }
            for i in range(20)
        ]
        with (
            patch("app.rag.retriever.embedding_service.encode_query", return_value=[0.1] * 384),
            patch("app.rag.retriever.qdrant_service.search", return_value=big_hits),
        ):
            result = retriever.retrieve("big context test")
            assert len(result.context) <= MAX_CONTEXT_CHARS + 200  # small tolerance for headers

    def test_document_score_preserved(self, retriever, sample_hits):
        with (
            patch("app.rag.retriever.embedding_service.encode_query", return_value=[0.1] * 384),
            patch("app.rag.retriever.qdrant_service.search", return_value=sample_hits),
        ):
            result = retriever.retrieve("cotton soil")
            assert result.documents[0].score == 0.87
            assert result.documents[1].score == 0.72
