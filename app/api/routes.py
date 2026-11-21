"""
app/api/routes.py
------------------
FastAPI route handlers.

Design decisions:
- All I/O-bound heavy work (model inference, Qdrant calls) is dispatched with
  asyncio.to_thread() so FastAPI's async event loop is never blocked.
- Uploaded PDFs are saved to a temp file before processing so the pipeline
  can use file-path-based APIs (pypdf, etc.).
- HTTP error codes: 400 for client errors (bad input), 503 for service
  unavailability (model not loaded), 500 for unexpected server errors.
- The /health endpoint checks all three components independently so ops teams
  can pinpoint which subsystem is down.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.schemas import (
    CollectionInfoResponse,
    ComponentHealth,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceCitationResponse,
)
from app.llm.agriparam import agriparam_service
from app.rag.embedder import embedding_service
from app.rag.pipeline import rag_pipeline
from app.services.qdrant_service import qdrant_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """
    Check availability of all system components.

    Returns 200 regardless — use the 'status' field to determine health.
    """
    qdrant_ok = await asyncio.to_thread(qdrant_service.health_check)
    embedding_ok = embedding_service._model is not None
    llm_ok = agriparam_service._model is not None

    components = ComponentHealth(
        qdrant=qdrant_ok,
        embedding_model=embedding_ok,
        llm=llm_ok,
    )

    collection_info = None
    if qdrant_ok:
        try:
            collection_info = await asyncio.to_thread(qdrant_service.collection_info)
        except Exception:
            pass

    if all([qdrant_ok, embedding_ok, llm_ok]):
        status_str = "healthy"
    elif any([qdrant_ok, embedding_ok, llm_ok]):
        status_str = "degraded"
    else:
        status_str = "unhealthy"

    return HealthResponse(
        status=status_str,
        components=components,
        collection_info=collection_info,
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer an agriculture question using RAG.

    The response always includes source citations and timing breakdowns.
    """
    if agriparam_service._model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Language model is not loaded. Please wait for startup to complete.",
        )

    logger.info("Query received", question=request.question[:80])

    try:
        result = await asyncio.to_thread(
            rag_pipeline.query, request.question
        )
    except Exception as exc:
        logger.error("Query failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {exc}",
        ) from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceCitationResponse(
                source=s.source,
                page=s.page,
                score=s.score,
                excerpt=s.excerpt,
            )
            for s in result.sources
        ],
        has_sources=result.has_sources,
        retrieval_time_ms=result.retrieval_time_ms,
        generation_time_ms=result.generation_time_ms,
        total_time_ms=result.total_time_ms,
        error=result.error,
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload and ingest a PDF into the Qdrant knowledge base.

    Accepts multipart/form-data with a single 'file' field.
    Supported format: PDF only.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    logger.info("Ingest request received", filename=file.filename)

    # Save to a temp file so the synchronous pipeline can use Path APIs
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", prefix=file.filename, delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await asyncio.to_thread(rag_pipeline.ingest, tmp_path)
    except Exception as exc:
        logger.error("Ingest failed", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        filename=file.filename,
        chunks_created=result.chunks_created,
        vectors_stored=result.vectors_stored,
        success=result.success,
        error=result.error,
    )


# ── Collection info ───────────────────────────────────────────────────────────

@router.get("/collection", response_model=CollectionInfoResponse, tags=["system"])
async def collection_info() -> CollectionInfoResponse:
    """Return current Qdrant collection statistics."""
    try:
        info = await asyncio.to_thread(qdrant_service.collection_info)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unavailable: {exc}",
        ) from exc

    return CollectionInfoResponse(**info)


# ── Reset (admin) ─────────────────────────────────────────────────────────────

@router.delete("/collection", tags=["system"])
async def reset_collection() -> dict:
    """
    Wipe all vectors from the collection and recreate it.
    Use with caution — this deletes ALL ingested data.
    """
    await asyncio.to_thread(qdrant_service.delete_all)
    return {"message": "Collection reset successfully."}
