"""
Integration tests for FastAPI routes.

Heavy services (Qdrant, AgriParam, Embedder) are mocked via conftest fixtures.
Tests validate: request validation, response schema, error handling.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "components" in data
        assert set(data["components"].keys()) >= {"qdrant", "embedding_model", "llm"}

    def test_health_status_values(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] in {"healthy", "degraded", "unhealthy"}


class TestQueryEndpoint:
    def test_empty_question_rejected(self, client):
        resp = client.post("/api/v1/query", json={"question": ""})
        assert resp.status_code == 422   # Pydantic validation error

    def test_too_short_question_rejected(self, client):
        resp = client.post("/api/v1/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_missing_question_field_rejected(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_valid_query_accepted(self, client, mock_query_result):
        """Mock the pipeline and verify the response shape."""
        with patch(
            "app.api.routes.rag_pipeline.query", return_value=mock_query_result
        ):
            resp = client.post(
                "/api/v1/query",
                json={"question": "Which crop is suitable for black soil?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "has_sources" in data
        assert "total_time_ms" in data

    def test_query_response_source_structure(self, client, mock_query_result):
        with patch(
            "app.api.routes.rag_pipeline.query", return_value=mock_query_result
        ):
            resp = client.post(
                "/api/v1/query",
                json={"question": "Which crop suits black soil?"},
            )
        sources = resp.json()["sources"]
        assert len(sources) >= 1
        for src in sources:
            assert "source" in src
            assert "page" in src
            assert "score" in src
            assert "excerpt" in src

    def test_top_k_override_accepted(self, client, mock_query_result):
        with patch(
            "app.api.routes.rag_pipeline.query", return_value=mock_query_result
        ):
            resp = client.post(
                "/api/v1/query",
                json={"question": "What fertilizer for cotton?", "top_k": 3},
            )
        assert resp.status_code == 200


class TestIngestEndpoint:
    def test_non_pdf_rejected(self, client):
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400

    def test_empty_file_rejected(self, client):
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_valid_pdf_upload_accepted(self, client, sample_pdf):
        from app.rag.pipeline import IngestResponse
        mock_ingest = IngestResponse(
            filename="test_agri.pdf",
            chunks_created=5,
            vectors_stored=5,
            success=True,
        )
        with patch("app.api.routes.rag_pipeline.ingest", return_value=mock_ingest):
            with open(sample_pdf, "rb") as f:
                resp = client.post(
                    "/api/v1/ingest",
                    files={"file": ("test_agri.pdf", f, "application/pdf")},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["chunks_created"] == 5


class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "docs" in data
