"""
streamlit_app/api_client.py
----------------------------
HTTP client for communicating with the FastAPI backend.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import requests

try:
    from streamlit_app.config import UI_CONFIG
except ModuleNotFoundError:
    from config import UI_CONFIG


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SourceCitation:
    source: str
    page: int
    score: float
    excerpt: str


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[SourceCitation]
    has_sources: bool
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    error: str | None = None


@dataclass
class IngestResult:
    filename: str
    chunks_created: int
    vectors_stored: int
    success: bool
    error: str | None = None


class AgriAPIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        # NOTE: Do NOT set Content-Type header globally on the session.
        # Multipart file uploads need requests to set it automatically
        # with the correct boundary. A global Content-Type breaks this.
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def health(self) -> dict[str, Any]:
        try:
            resp = self._session.get(self._url("/health"), timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot reach the API server. Is FastAPI running?")
        except requests.exceptions.HTTPError as exc:
            raise APIError(str(exc), status_code=exc.response.status_code)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        payload: dict[str, Any] = {"question": question}
        if top_k:
            payload["top_k"] = top_k

        try:
            resp = self._session.post(
                self._url("/query"),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=UI_CONFIG["query_timeout_seconds"],
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot reach the API server. Is FastAPI running?")
        except requests.exceptions.Timeout:
            raise APIError(f"Query timed out after {UI_CONFIG['query_timeout_seconds']}s.")
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc))
            raise APIError(detail, status_code=exc.response.status_code)

        data = resp.json()
        return QueryResult(
            question=data["question"],
            answer=data["answer"],
            sources=[
                SourceCitation(
                    source=s["source"],
                    page=s["page"],
                    score=s["score"],
                    excerpt=s["excerpt"],
                )
                for s in data.get("sources", [])
            ],
            has_sources=data["has_sources"],
            total_time_ms=data["total_time_ms"],
            retrieval_time_ms=data["retrieval_time_ms"],
            generation_time_ms=data["generation_time_ms"],
            error=data.get("error"),
        )

    def ingest(self, filename: str, file_bytes: bytes) -> IngestResult:
        """
        Upload a PDF for ingestion.

        IMPORTANT: Do NOT set Content-Type header here.
        requests sets it automatically as multipart/form-data with
        the correct boundary when you use the 'files' parameter.
        Setting it manually breaks the boundary and FastAPI receives None.
        """
        if not file_bytes:
            raise APIError("File bytes are empty — nothing to upload.")

        try:
            resp = self._session.post(
                self._url("/ingest"),
                files={
                    "file": (filename, io.BytesIO(file_bytes), "application/pdf")
                },
                # NO headers= here — let requests handle Content-Type automatically
                timeout=UI_CONFIG["ingest_timeout_seconds"],
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise APIError("Cannot reach the API server. Is FastAPI running?")
        except requests.exceptions.Timeout:
            raise APIError("PDF ingestion timed out.")
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            raise APIError(detail, status_code=exc.response.status_code)

        data = resp.json()
        return IngestResult(
            filename=data["filename"],
            chunks_created=data["chunks_created"],
            vectors_stored=data["vectors_stored"],
            success=data["success"],
            error=data.get("error"),
        )

    def collection_info(self) -> dict[str, Any]:
        try:
            resp = self._session.get(self._url("/collection"), timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise APIError(f"Failed to get collection info: {exc}")