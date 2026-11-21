"""
app/rag/embedder.py
-------------------
Embedding generation using sentence-transformers.

Design decisions:
- sentence-transformers/all-MiniLM-L6-v2 is the default because it:
    * Has dimension 384 (small → fast Qdrant indexing)
    * Performs well on domain-specific short texts
    * Runs entirely on CPU in < 1 s per batch
- The model is loaded once and cached as a module-level singleton.
  Loading SentenceTransformer takes ~2 s; we do it at startup, not per query.
- Batched encoding (encode_documents) is used for bulk ingestion to leverage
  parallelism in the transformer; single-vector encode_query for search.
- The embedding dimension is validated against settings on load to catch
  model/config mismatches early.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.rag.ingestion import Document
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps SentenceTransformer to generate embeddings for documents and queries."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Download (or load from cache) the embedding model.
        Must be called before any encode_* methods.
        """
        logger.info(
            "Loading embedding model",
            model=settings.embedding_model_name,
            device=settings.agriparam_device,
        )

        self._model = SentenceTransformer(
            settings.embedding_model_name,
            device=settings.agriparam_device,
        )

        # Validate dimension matches config
        test_vec = self._model.encode(["dimension check"])
        actual_dim = test_vec.shape[1]
        if actual_dim != settings.embedding_dimension:
            raise ValueError(
                f"Embedding model {settings.embedding_model_name!r} produces "
                f"vectors of dimension {actual_dim}, but EMBEDDING_DIMENSION "
                f"is set to {settings.embedding_dimension}. Update your .env."
            )

        logger.info(
            "Embedding model loaded",
            model=settings.embedding_model_name,
            dimension=actual_dim,
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            raise RuntimeError(
                "EmbeddingService not loaded. Call load() first."
            )
        return self._model

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_documents(
        self,
        documents: list[Document],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Encode a list of Document objects into embedding vectors.

        Parameters
        ----------
        documents     : Document chunks to embed (text field used).
        batch_size    : Number of texts per model forward pass.
        show_progress : Print tqdm progress bar (useful during offline ingestion).

        Returns
        -------
        List of float lists; i-th vector corresponds to i-th document.
        """
        texts = [doc.text for doc in documents]
        logger.info("Encoding documents", count=len(texts), batch_size=batch_size)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # Cosine similarity = dot product on unit vectors
            convert_to_numpy=True,
        )

        logger.info("Document encoding complete", vectors=len(vectors))
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        """
        Encode a single query string for similarity search.

        Uses the same normalisation as encode_documents so cosine scores are
        directly comparable.
        """
        if not query.strip():
            raise ValueError("Query text must not be empty.")

        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].tolist()


# Module-level singleton
embedding_service = EmbeddingService()
