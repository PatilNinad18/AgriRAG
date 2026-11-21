"""
app/main.py
-----------
FastAPI application factory.

Windows fix: uvicorn on Windows with structlog can silently exit during
lifespan async context. Solution: use @app.on_event (deprecated but
stable on Windows) instead of the asynccontextmanager lifespan pattern.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.llm.agriparam import agriparam_service
from app.rag.embedder import embedding_service
from app.services.qdrant_service import qdrant_service
from app.utils.config import settings
from app.utils.logger import configure_logging, get_logger

# Configure logging immediately at import time
configure_logging(
    log_level=settings.log_level,
    json_logs=settings.json_logs,
)
logger = get_logger("startup")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Agriculture RAG Assistant — answers crop, soil, and fertiliser "
            "questions using semantic retrieval and a configurable language model."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.on_event("startup")
    async def startup_event():
        import asyncio
        logger.info("Starting AgriRAG Assistant", env=settings.app_env)

        logger.info("Connecting to Qdrant...")
        await asyncio.to_thread(qdrant_service.connect)

        logger.info("Loading embedding model...")
        await asyncio.to_thread(embedding_service.load)

        logger.info("Loading language model (may take a minute)...")
        await asyncio.to_thread(agriparam_service.load)

        logger.info("All services ready — application started")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down AgriRAG Assistant")

    @app.get("/", tags=["root"])
    async def root():
        return {
            "service": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_config=None,
    )