"""
app/rag/retriever.py
---------------------
Semantic retrieval and RAG context construction.

Design decisions:
- Retriever is intentionally thin: it coordinates EmbeddingService and
  QdrantService but adds no additional ML logic.
- build_context() formats retrieved chunks into a prompt-friendly string,
  numbering each source so the LLM can cite them as [1], [2], etc.
- The returned RetrievalResult carries both the formatted context string and
  the raw retrieved documents so the API can surface citations in the UI.
- Context is capped at MAX_CONTEXT_CHARS to avoid exceeding the model's
  context window.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.embedder import embedding_service
from app.services.qdrant_service import qdrant_service
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_CONTEXT_CHARS = 3_500   # ~900 tokens; keeps prompt + generation within 2048


@dataclass
class RetrievedDocument:
    """A single document chunk returned by semantic search."""

    text: str
    source: str
    page: int
    chunk_index: int
    score: float

    @property
    def citation_label(self) -> str:
        return f"{self.source} (p.{self.page})"


@dataclass
class RetrievalResult:
    """Full output of a retrieval operation."""

    query: str
    documents: list[RetrievedDocument]
    context: str          # Formatted context string for the LLM prompt
    has_results: bool


class Retriever:
    """Coordinates embedding + vector search to retrieve relevant documents."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Execute semantic retrieval for a natural language query.

        Parameters
        ----------
        query           : User's agriculture question.
        top_k           : Override RAG_TOP_K from config.
        score_threshold : Override RAG_SCORE_THRESHOLD from config.

        Returns
        -------
        RetrievalResult
        """
        logger.info("Starting retrieval", query=query[:80])

        # 1. Embed the query
        query_vector = embedding_service.encode_query(query)

        # 2. Vector search
        raw_hits = qdrant_service.search(
            query_vector=query_vector,
            top_k=top_k or settings.rag_top_k,
            score_threshold=score_threshold or settings.rag_score_threshold,
        )

        if not raw_hits:
            logger.info("No documents retrieved above score threshold", query=query[:80])
            return RetrievalResult(
                query=query,
                documents=[],
                context="",
                has_results=False,
            )

        # 3. Wrap raw dicts as typed objects
        documents = [
            RetrievedDocument(
                text=hit["text"],
                source=hit["source"],
                page=hit["page"],
                chunk_index=hit["chunk_index"],
                score=hit["score"],
            )
            for hit in raw_hits
        ]

        # 4. Build context string
        context = self._build_context(documents)

        logger.info(
            "Retrieval complete",
            documents=len(documents),
            context_chars=len(context),
        )

        return RetrievalResult(
            query=query,
            documents=documents,
            context=context,
            has_results=True,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(documents: list[RetrievedDocument]) -> str:
        """
        Format retrieved chunks into a numbered context block.

        Format:
            [1] Source: crop_guide.pdf (p.3) | Score: 0.82
            <text of chunk 1>

            [2] Source: soil_manual.pdf (p.7) | Score: 0.75
            <text of chunk 2>
            ...
        """
        parts: list[str] = []
        total_chars = 0

        for idx, doc in enumerate(documents, start=1):
            header = (
                f"[{idx}] Source: {doc.source} (p.{doc.page}) "
                f"| Score: {doc.score:.2f}"
            )
            chunk = f"{header}\n{doc.text}"

            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                logger.debug(
                    "Context truncated to fit window",
                    included=idx - 1,
                    total=len(documents),
                )
                break

            parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n".join(parts)


# Module-level singleton
retriever = Retriever()
