"""
app/services/qdrant_service.py
-------------------------------
Qdrant vector database client wrapper.

Design decisions:
- Wraps qdrant-client behind a thin service class so the rest of the codebase
  is not coupled to Qdrant's SDK types — easier to swap or mock.
- The collection is created lazily on first use (idempotent).
- Retry logic (tenacity) handles transient connection failures at startup.
- All public methods are synchronous; FastAPI routes run them in a thread pool
  via asyncio.to_thread() inside the endpoint layer.
"""

from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantService:
    """Manages a single Qdrant collection for the agriculture knowledge base."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.embedding_dimension

    # ── Connection ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def connect(self) -> None:
        """
        Establish connection to Qdrant and ensure the collection exists.
        Called once at application startup.
        """
        logger.info(
            "Connecting to Qdrant",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Ensure local Qdrant traffic bypasses any corporate or Docker proxy.
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            if "localhost" not in no_proxy:
                os.environ["NO_PROXY"] = f"{no_proxy},localhost,127.0.0.1"
        else:
            os.environ["NO_PROXY"] = "localhost,127.0.0.1"

        self._client = QdrantClient(
            url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
        self._ensure_collection()
        logger.info("Qdrant connection established", collection=self.collection_name)

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        assert self._client is not None

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in existing:
            logger.debug("Collection already exists", collection=self.collection_name)
            return

        logger.info("Creating Qdrant collection", collection=self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
            # Optimiser settings tuned for a small-medium agriculture knowledge base
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=10_000,
            ),
        )
        logger.info("Collection created", collection=self.collection_name)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError(
                "QdrantService not connected. Call connect() first."
            )
        return self._client

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        """
        Upsert document chunks into the collection.

        Parameters
        ----------
        ids      : Unique string IDs (will be hashed to UUIDs internally).
        vectors  : Embedding vectors matching EMBEDDING_DIMENSION.
        payloads : Metadata dicts — must contain at least 'text' and 'source'.

        Returns
        -------
        int : Number of points upserted.
        """
        import uuid

        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id)),
                vector=vector,
                payload=payload,
            )
            for doc_id, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Vectors upserted", count=len(points))
        return len(points)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search against the collection.

        Returns
        -------
        List of dicts with keys: text, source, page, score, chunk_index.
        """
        k = top_k or settings.rag_top_k
        threshold = score_threshold or settings.rag_score_threshold

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )

        documents = []
        for hit in results:
            payload = hit.payload or {}
            documents.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "score": round(hit.score, 4),
                }
            )

        logger.debug("Search complete", results=len(documents), top_k=k)
        return documents

    # ── Admin ─────────────────────────────────────────────────────────────────

    def collection_info(self) -> dict[str, Any]:
        """Return basic info about the collection (point count, status)."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": self.vector_size,
        }

    def delete_all(self) -> None:
        """Wipe all vectors from the collection (useful for re-ingestion)."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.warning("Collection wiped and recreated", collection=self.collection_name)

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable and the collection exists."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info is not None
        except Exception:
            return False


# Module-level singleton — imported by FastAPI lifespan and services
qdrant_service = QdrantService()
