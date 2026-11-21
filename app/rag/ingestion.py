"""
app/rag/ingestion.py
---------------------
PDF ingestion pipeline: load → clean → chunk → return Document objects.

Design decisions:
- Uses pypdf (pure-Python, no system deps) for text extraction.
- LangChain's RecursiveCharacterTextSplitter is the only LangChain component
  used — it produces better agriculture-domain chunks than naive splitting
  because it respects paragraph/sentence boundaries.
- Each Document carries rich metadata (source filename, page number, chunk
  index) so that source citations in the UI are precise.
- The pipeline is stateless and functional: ingest_pdf(path) → [Document].
  The caller is responsible for persisting to Qdrant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """A single text chunk with its provenance metadata."""

    text: str
    source: str            # Original filename (e.g. "soil_health_guide.pdf")
    page: int              # 1-based page number in the source PDF
    chunk_index: int       # Position of this chunk within the page/document
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Stable unique identifier for this chunk."""
        return f"{self.source}::page{self.page}::chunk{self.chunk_index}"


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract (page_number, text) tuples from a PDF.

    Returns
    -------
    List of (1-based page number, raw text) pairs.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append((page_num, raw))

    logger.debug(
        "PDF pages extracted",
        file=pdf_path.name,
        page_count=len(pages),
    )
    return pages


def _clean_text(text: str) -> str:
    """
    Normalise whitespace and remove artefacts common in PDF extractions.

    - Collapse multiple spaces / tabs into one space.
    - Collapse 3+ consecutive newlines into two (preserve paragraph breaks).
    - Strip leading/trailing whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build the text splitter with configured chunk_size and chunk_overlap.

    RecursiveCharacterTextSplitter tries successively smaller separators
    (\n\n → \n → " " → "") so it respects natural text boundaries.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: Path) -> list[Document]:
    """
    Full ingestion pipeline for a single PDF file.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the PDF.

    Returns
    -------
    List[Document]
        All text chunks extracted from the PDF, ready for embedding.

    Raises
    ------
    FileNotFoundError  : If the PDF does not exist.
    ValueError         : If the PDF yields no extractable text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Starting PDF ingestion", file=pdf_path.name)

    pages = _extract_pages(pdf_path)
    splitter = _build_splitter()
    documents: list[Document] = []

    for page_num, raw_text in pages:
        cleaned = _clean_text(raw_text)
        if not cleaned:
            logger.debug("Skipping empty page", page=page_num, file=pdf_path.name)
            continue

        chunks = splitter.split_text(cleaned)
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 20:   # Skip meaningless micro-chunks
                continue

            documents.append(
                Document(
                    text=chunk,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_index=chunk_idx,
                    metadata={"file_path": str(pdf_path)},
                )
            )

    if not documents:
        raise ValueError(
            f"No text could be extracted from {pdf_path.name}. "
            "The file may be scanned images without OCR."
        )

    logger.info(
        "PDF ingestion complete",
        file=pdf_path.name,
        chunks=len(documents),
    )
    return documents


def ingest_directory(directory: Path) -> list[Document]:
    """
    Ingest all PDFs in a directory recursively.

    Parameters
    ----------
    directory : Path
        Directory to scan for .pdf files.

    Returns
    -------
    Flat list of all Document chunks across all PDFs.
    """
    pdf_files = sorted(directory.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in directory", path=str(directory))
        return []

    all_documents: list[Document] = []
    for pdf_path in pdf_files:
        try:
            docs = ingest_pdf(pdf_path)
            all_documents.extend(docs)
        except Exception as exc:
            logger.error(
                "Failed to ingest PDF",
                file=pdf_path.name,
                error=str(exc),
            )

    logger.info(
        "Directory ingestion complete",
        directory=str(directory),
        files=len(pdf_files),
        total_chunks=len(all_documents),
    )
    return all_documents
