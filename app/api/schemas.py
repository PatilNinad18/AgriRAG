"""
app/api/schemas.py
-------------------
Pydantic models for FastAPI request validation and response serialisation.

All response models are strict (no extra fields) so the API contract is
explicit and frontend consumers can rely on the shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Query endpoint ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Agriculture question from the user.",
        examples=["Which crop is suitable for black soil?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override default number of retrieved documents.",
    )


class SourceCitationResponse(BaseModel):
    source: str
    page: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceCitationResponse]
    has_sources: bool
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    error: str | None = None


# ── Ingest endpoint ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    vectors_stored: int
    success: bool
    error: str | None = None


# ── Health endpoint ───────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    qdrant: bool
    embedding_model: bool
    llm: bool


class HealthResponse(BaseModel):
    status: str          # "healthy" | "degraded" | "unhealthy"
    components: ComponentHealth
    collection_info: dict | None = None


# ── Collection info ───────────────────────────────────────────────────────────

class CollectionInfoResponse(BaseModel):
    name: str
    points_count: int | None
    status: str
    vector_size: int
