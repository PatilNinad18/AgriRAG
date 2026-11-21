# 🌾 AgriTrackr AI — AgriRAG Assistant

> **An intelligent, domain-specialized Retrieval-Augmented Generation (RAG) assistant for Indian agriculture, farming practices, soil health, crop management, and fertilizer recommendations.**

---

## 📌 Overview

**AgriTrackr AI (AgriRAG Assistant)** is an end-to-end RAG system designed to empower farmers, agronomists, and agricultural extension workers with accurate, verifiable, and context-aware agricultural insights.

The platform pairs a compact instruction-tuned language model (**Qwen2.5-0.5B-Instruct** by default) with a vector search engine (**Qdrant**) and a semantic embedder (**Sentence-Transformers**). It features a **dual-mode operational engine**:
1. **📄 Grounded RAG Mode**: When relevant documents (crop manuals, soil advisories, government extension bulletins) are found in the knowledge base, answers are strictly grounded in the literature and accompanied by page-level citations and similarity scores.
2. **🧠 Domain Fallback Mode**: When the query falls outside the ingested documents or below the cosine similarity threshold, the configured language model answers from its pretrained knowledge.

---

## ✨ Key Features

- **Dual-Mode Response Engine**: Dynamic switching between grounded document citations and domain-expert LLM fallback.
- **Smart PDF Ingestion**: Extracts text using `pypdf`, normalizes formatting, and chunks documents using `RecursiveCharacterTextSplitter` with full provenance tracking (source file, page number, chunk index).
- **Sub-Second Semantic Retrieval**: High-speed vector indexing and cosine similarity search powered by Qdrant and `sentence-transformers/all-MiniLM-L6-v2` (384-d).
- **GPU & CPU Flexibility**: Optimized for NVIDIA GPUs (RTX series with FP16 precision and VRAM tracking) with CPU fallback support.
- **Interactive Streamlit Web UI**:
  - Live system telemetry (status monitors for Qdrant, Embedder, and LLM).
  - Drag-and-drop PDF ingestion with instant chunk and vector feedback.
  - Expandable source citation cards with similarity scores and excerpts.
  - Granular latency profiling (retrieval time, generation time, and total latency).
  - Pre-configured example questions for rapid testing.
- **Production-Grade FastAPI Backend**: Asynchronous thread offloading (`asyncio.to_thread`), typed Pydantic v2 schemas, automated Swagger/ReDoc docs, and structured logging with `structlog`.
- **Full Test Suite**: Unit and integration test coverage for API endpoints, ingestion pipelines, and retrieval logic with mock fixtures.

---

## 🏛️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               User Interface                 │
                    │      (Streamlit Web App on :8501)            │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP REST / Uploads
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │             FastAPI Backend (:8000)          │
                    │       app/api/routes.py & schemas.py         │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │     RAG Pipeline      │
                               │  app/rag/pipeline.py  │
                               └───────────┬───────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
                 ▼                                                   ▼
     ┌───────────────────────┐                           ┌───────────────────────┐
     │   Ingestion Pipeline  │                           │   Retriever Service   │
     │  (PDF -> Clean ->     │                           │  (Embed Query ->      │
     │   Chunk -> Embed)     │                           │   Search Qdrant)      │
     └───────────┬───────────┘                           └───────────┬───────────┘
                 │                                                   │
                 │ Vectors + Payload                                 │ Similarity Hits
                 ▼                                                   ▼
     ┌───────────────────────────────────────────────────────────────────────────┐
     │                         Qdrant Vector Database (:6333)                    │
     │                         Collection: agri_knowledge                        │
     └───────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │          Prompt Builder & Logic           │
                     │  Context Found?                           │
                     │  ├── Yes ──> RAG Prompt Template          │
                     │  └── No  ──> Domain Fallback Template     │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │          Language Model Service           │
                     │     (Qwen/Qwen2.5-0.5B-Instruct)          │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │   Structured Answer + Provenance Sources  │
                     │            Returned to Client             │
                     └───────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
agritrackr-ai/
├── app/                              # Core FastAPI backend application
│   ├── __init__.py
│   ├── main.py                       # FastAPI application factory & startup hooks
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                 # Endpoint route handlers (/health, /query, /ingest, etc.)
│   │   └── schemas.py                # Pydantic v2 request and response schemas
│   ├── llm/
│   │   ├── __init__.py
│   │   └── agriparam.py              # Configurable LLM loader and inference
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py               # Sentence-Transformers embedding service (384-d)
│   │   ├── ingestion.py              # PDF extraction, text normalization & chunking
│   │   ├── pipeline.py               # Top-level orchestrator for query & ingestion workflows
│   │   └── retriever.py              # Qdrant search caller & prompt context constructor
│   ├── services/
│   │   ├── __init__.py
│   │   └── qdrant_service.py         # Qdrant client wrapper, retry logic & vector upserting
│   └── utils/
│       ├── __init__.py
│       ├── config.py                 # Pydantic-Settings environment configuration
│       └── logger.py                 # Structlog logger initialization
│
├── streamlit_app/                    # Streamlit frontend user interface
│   ├── __init__.py
│   ├── app.py                        # Streamlit UI application with chat & file uploader
│   ├── api_client.py                 # Typed HTTP client communicating with FastAPI
│   └── config.py                     # UI settings, limits & sample questions
│
├── data/                             # Data and cache storage (git-ignored)
│   ├── raw/                          # Directory for unprocessed agricultural PDF files
│   ├── processed/                    # Processed document chunks and staging
│   ├── embeddings/                   # Saved offline embeddings
│   └── .hf_cache/                    # Local Hugging Face cache for models and weights
│
├── qdrant/
│   └── docker-compose.yml            # Docker Compose service definition for Qdrant
│
├── tests/                            # Automated testing suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures and mock service patches
│   ├── test_api.py                   # Integration tests for FastAPI endpoints
│   ├── test_ingestion.py             # Unit tests for text cleaning and PDF chunking
│   └── test_retriever.py             # Tests for retrieval filtering and context construction
│
├── .env.example                      # Template for environment configuration
├── .gitignore                        # Git ignore rules
├── diagnose.py                       # Diagnostic script verifying GPU, CUDA, Qdrant & models
├── Makefile                          # Unified build and automation commands
├── pytest.ini                        # Pytest configuration settings
├── requirements.txt                  # Python dependencies
└── test_run.py                       # Lightweight sanity check script
```

---

## ⚙️ Prerequisites & Tech Stack

- **Operating System**: Linux, macOS, or Windows 10/11 (includes Windows-specific optimizations).
- **Python**: Version `3.10` to `3.13` (Prebuilt wheels configured for Python 3.13).
- **Docker & Docker Compose**: For hosting the local Qdrant vector database.
- **Hardware Recommendations**:
  - **GPU**: NVIDIA GPU with at least 6 GB VRAM (e.g., RTX 3060/4050/4060 or higher) for FP16 CUDA acceleration.
  - **CPU**: Multi-core processor (if running LLM in CPU mode).
  - **RAM**: Minimum 8 GB RAM (16 GB recommended).

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Python Environment

```bash
# Clone repository
git clone <repository-url>
cd agritrackr-ai

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch**: The default `requirements.txt` includes CUDA 12.6 wheels (`torch==2.6.0+cu126`). If you are running on a machine without an NVIDIA GPU, uncomment the CPU wheel index in `requirements.txt` or install standard PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 3. Configure Environment Variables

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

Adjust the key settings as needed:

| Parameter | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Runtime environment (`development`, `production`) |
| `API_PORT` | `8000` | Port for the FastAPI server |
| `STREAMLIT_API_BASE_URL` | `http://localhost:8000` | Address Streamlit uses to reach FastAPI |
| `QDRANT_HOST` | `localhost` | Hostname of Qdrant instance |
| `QDRANT_PORT` | `6333` | REST port for Qdrant |
| `QDRANT_COLLECTION_NAME` | `agri_knowledge` | Qdrant collection name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer embedding model |
| `EMBEDDING_DIMENSION` | `384` | Embedding dimensionality |
| `AGRIPARAM_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model identifier for answer generation |
| `AGRIPARAM_DEVICE` | `cpu` | Device target: `cuda`, `cpu`, or `mps` |
| `AGRIPARAM_MAX_NEW_TOKENS` | `192` | Max tokens generated per answer |
| `RAG_TOP_K` | `5` | Top documents retrieved per query |
| `RAG_SCORE_THRESHOLD` | `0.35` | Minimum cosine similarity score for context inclusion |
| `CHUNK_SIZE` | `800` | Chunk size (characters) for document splitter |
| `CHUNK_OVERLAP` | `100` | Overlap (characters) between adjacent chunks |

> The first API startup downloads the language model (about 1 GB) and the embedding model. Ensure the drive configured by `HF_HOME` has several GB free; the API only begins accepting requests after both models finish loading.

### 4. Start Qdrant Vector Store

Start Qdrant containerized using Docker:

```bash
# Using Makefile
make qdrant

# Or directly using Docker Compose
docker compose -f qdrant/docker-compose.yml up -d
```

Verify that Qdrant is running by browsing to: [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 5. Run System Diagnostics (Recommended)

Run `diagnose.py` to ensure GPU/CUDA drivers, PyTorch, Qdrant connectivity, and model checkpoints load properly:

```bash
python diagnose.py
```

### 6. Start the FastAPI Backend

```bash
# Using Makefile
make api

# Or directly with Uvicorn
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000). Interactive API documentation can be accessed at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 7. Start the Streamlit Frontend

In a separate terminal (with the virtual environment activated):

```bash
# Using Makefile
make ui

# Or directly with Streamlit
streamlit run streamlit_app/app.py --server.port 8501
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 📚 Knowledge Base Ingestion

You can populate the Qdrant vector database with PDFs (e.g. ICAR handbooks, state agricultural guides, fertilizer schedules) through three methods:

### Method 1: Via Streamlit UI
1. Open the Streamlit dashboard ([http://localhost:8501](http://localhost:8501)).
2. Look at the left sidebar under **"📄 Add Knowledge"**.
3. Drag and drop any agricultural `.pdf` file (up to 50 MB).
4. Click **"📥 Ingest into knowledge base"**.
5. Once processed, the app confirms chunk counts and indexed vectors.

### Method 2: Via REST API Endpoint
Use `curl` or any HTTP client to POST a PDF to the `/api/v1/ingest` endpoint:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/raw/cotton_cultivation_guide.pdf"
```

Response:
```json
{
  "filename": "cotton_cultivation_guide.pdf",
  "chunks_created": 42,
  "vectors_stored": 42,
  "success": true,
  "error": null
}
```

### Method 3: Batch Ingestion Script
Place all PDF files into `data/raw/` and run the Makefile ingestion helper:

```bash
make ingest
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service name, version, and documentation link |
| `GET` | `/api/v1/health` | Subsystem status (`qdrant`, `embedding_model`, `llm`, vector count) |
| `POST` | `/api/v1/query` | Submit natural language query and obtain RAG-augmented response |
| `POST` | `/api/v1/ingest` | Upload a PDF file for chunking, embedding, and indexing |
| `GET` | `/api/v1/collection`| Fetch collection statistics (total points, vector size, status) |
| `DELETE` | `/api/v1/collection`| Wipe all stored vectors and reset collection (Admin) |

### Sample Query Request & Response

**Request (`POST /api/v1/query`):**
```json
{
  "question": "Which crops are best suited for black soil and what fertilizer should be used?"
}
```

**Response:**
```json
{
  "question": "Which crops are best suited for black soil and what fertilizer should be used?",
  "answer": "1. Overview / Introduction:\nBlack soils (Regur soils) are rich in clay and have high moisture retention capacity...\n2. Recommended Crops:\nCotton, soybean, wheat, and sorghum thrive in black soil...\n3. Fertilizer Recommendations:\nApply balanced NPK with special attention to phosphorus and zinc deficiency...",
  "sources": [
    {
      "source": "soil_and_crop_management.pdf",
      "page": 14,
      "score": 0.8421,
      "excerpt": "Black soils are predominantly found in the Deccan plateau. They are ideal for cotton cultivation due to high moisture retention..."
    }
  ],
  "has_sources": true,
  "retrieval_time_ms": 38.4,
  "generation_time_ms": 1420.2,
  "total_time_ms": 1458.6,
  "error": null
}
```

---

## 🧪 Testing

The repository uses `pytest` with mock fixtures so that the entire test suite can be executed rapidly without downloading multi-gigabyte models or running a live Qdrant container.

```bash
# Run test suite
make test
# Or:
pytest tests/ -v --tb=short

# Run test suite with coverage report
make test-coverage
```

---

## 🛠️ Helpful Commands (`Makefile`)

| Command | Action |
|---|---|
| `make setup` | Install all dependencies and copy `.env.example` to `.env` |
| `make qdrant` | Start local Qdrant container in background |
| `make qdrant-stop` | Stop local Qdrant container |
| `make api` | Run FastAPI server with auto-reload |
| `make ui` | Run Streamlit user interface |
| `make test` | Execute test suite |
| `make lint` | Run `ruff` code linter |
| `make ingest` | Batch ingest all PDFs found under `data/raw/` |
| `make clean` | Clean up `__pycache__` and pytest caches |

---

## 🛡️ License & Acknowledgements

- **Qwen2.5**: Default compact instruction model from [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- **Qdrant**: Open-source vector similarity search engine by [Qdrant](https://qdrant.tech/).
- **Sentence-Transformers**: Open-source framework for state-of-the-art sentence embeddings.
