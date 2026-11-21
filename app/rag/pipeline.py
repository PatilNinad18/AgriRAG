"""
app/rag/pipeline.py
--------------------
Top-level RAG pipeline: orchestrates retrieval + generation.

Design decisions:
- RAGPipeline is the single entry point for both the query path and the
  ingestion path.  FastAPI routes import this, not individual services.
- Query flow:  question → embed → search Qdrant → build context → language model
- Ingest flow: PDF → chunks → embed → upsert Qdrant
- Returns typed response objects (RAGResponse, IngestResponse) that map
  directly to Pydantic API schemas, keeping the API layer thin.
- Errors are caught here and surfaced as structured error fields so the API
  can return a 200 with an error message rather than a 500 that crashes the
  Streamlit UI mid-stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.llm.agriparam import agriparam_service
from app.rag.embedder import embedding_service
from app.rag.ingestion import Document, ingest_pdf
from app.rag.retriever import RetrievedDocument, retriever
from app.services.qdrant_service import qdrant_service
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class SourceCitation:
    source: str
    page: int
    score: float
    excerpt: str   # First 200 chars of the chunk for UI display


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[SourceCitation]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    has_sources: bool
    error: str | None = None


@dataclass
class IngestResponse:
    filename: str
    chunks_created: int
    vectors_stored: int
    success: bool
    error: str | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Orchestrates retrieval and generation for the AgriRAG assistant."""

    # ── Query path ────────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG query: retrieve relevant documents, then generate an answer.

        Parameters
        ----------
        question : The user's agriculture question.

        Returns
        -------
        RAGResponse with the generated answer and source citations.
        """
        overall_start = time.perf_counter()
        logger.info("RAG query started", question=question[:100])

        # ── 1. Retrieval ──────────────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        try:
            result = retriever.retrieve(question)
        except Exception as exc:
            logger.error("Retrieval failed", error=str(exc), exc_info=True)
            return RAGResponse(
                question=question,
                answer="",
                sources=[],
                retrieval_time_ms=0,
                generation_time_ms=0,
                total_time_ms=0,
                has_sources=False,
                error=f"Retrieval failed: {exc}",
            )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # ── 2. Generation ─────────────────────────────────────────────────────
        generation_start = time.perf_counter()
        try:
            answer = agriparam_service.generate(
                question=question,
                context=result.context,
            )
        except Exception as exc:
            logger.error("Generation failed", error=str(exc), exc_info=True)
            return RAGResponse(
                question=question,
                answer="",
                sources=[],
                retrieval_time_ms=retrieval_ms,
                generation_time_ms=0,
                total_time_ms=(time.perf_counter() - overall_start) * 1000,
                has_sources=result.has_results,
                error=f"Answer generation failed: {exc}",
            )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - overall_start) * 1000

        # ── 3. Format sources ─────────────────────────────────────────────────
        sources = [
            SourceCitation(
                source=doc.source,
                page=doc.page,
                score=doc.score,
                excerpt=doc.text[:200] + ("…" if len(doc.text) > 200 else ""),
            )
            for doc in result.documents
        ]

        logger.info(
            "RAG query complete",
            retrieval_ms=round(retrieval_ms),
            generation_ms=round(generation_ms),
            total_ms=round(total_ms),
            sources=len(sources),
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_time_ms=round(retrieval_ms, 1),
            generation_time_ms=round(generation_ms, 1),
            total_time_ms=round(total_ms, 1),
            has_sources=bool(sources),
        )

    # ── Ingest path ───────────────────────────────────────────────────────────

    def ingest(self, pdf_path: Path) -> IngestResponse:
        """
        Ingest a PDF file into the Qdrant knowledge base.

        Parameters
        ----------
        pdf_path : Path to the uploaded/local PDF.

        Returns
        -------
        IngestResponse
        """
        logger.info("Ingestion started", file=pdf_path.name)

        # 1. Extract + chunk
        try:
            documents: list[Document] = ingest_pdf(pdf_path)
        except Exception as exc:
            logger.error("PDF ingestion failed", file=pdf_path.name, error=str(exc))
            return IngestResponse(
                filename=pdf_path.name,
                chunks_created=0,
                vectors_stored=0,
                success=False,
                error=str(exc),
            )

        # 2. Embed
        try:
            vectors = embedding_service.encode_documents(
                documents, batch_size=32, show_progress=True
            )
        except Exception as exc:
            logger.error("Embedding failed", file=pdf_path.name, error=str(exc))
            return IngestResponse(
                filename=pdf_path.name,
                chunks_created=len(documents),
                vectors_stored=0,
                success=False,
                error=str(exc),
            )

        # 3. Upsert to Qdrant
        try:
            ids = [doc.doc_id for doc in documents]
            payloads = [
                {
                    "text": doc.text,
                    "source": doc.source,
                    "page": doc.page,
                    "chunk_index": doc.chunk_index,
                }
                for doc in documents
            ]
            stored = qdrant_service.upsert_vectors(ids, vectors, payloads)
        except Exception as exc:
            logger.error("Qdrant upsert failed", file=pdf_path.name, error=str(exc))
            return IngestResponse(
                filename=pdf_path.name,
                chunks_created=len(documents),
                vectors_stored=0,
                success=False,
                error=str(exc),
            )

        logger.info(
            "Ingestion complete",
            file=pdf_path.name,
            chunks=len(documents),
            stored=stored,
        )

        return IngestResponse(
            filename=pdf_path.name,
            chunks_created=len(documents),
            vectors_stored=stored,
            success=True,
        )


# Module-level singleton
rag_pipeline = RAGPipeline()
