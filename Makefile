# ════════════════════════════════════════════════════════════════════════════
#  AgriRAG Assistant — Makefile
# ════════════════════════════════════════════════════════════════════════════

.PHONY: help setup qdrant api ui test lint ingest clean

help:
	@echo ""
	@echo "  AgriRAG Assistant — Development Commands"
	@echo "  ──────────────────────────────────────────"
	@echo "  make setup    Install Python dependencies"
	@echo "  make qdrant   Start Qdrant via Docker Compose"
	@echo "  make api      Start FastAPI backend"
	@echo "  make ui       Start Streamlit frontend"
	@echo "  make test     Run pytest suite"
	@echo "  make lint     Run ruff linter"
	@echo "  make ingest   Ingest all PDFs in data/raw/"
	@echo "  make clean    Remove __pycache__ and temp files"
	@echo ""

setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ Dependencies installed. Edit .env to configure."

qdrant:
	docker compose -f qdrant/docker-compose.yml up -d
	@echo "✅ Qdrant running at http://localhost:6333"

qdrant-stop:
	docker compose -f qdrant/docker-compose.yml down

api:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run streamlit_app/app.py --server.port 8501

test:
	python -m pytest tests/ -v --tb=short

test-coverage:
	python -m pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	python -m ruff check app/ streamlit_app/ tests/

ingest:
	@echo "Ingesting all PDFs in data/raw/ via API..."
	@for pdf in data/raw/*.pdf; do \
		echo "  → Ingesting $$pdf"; \
		curl -s -X POST http://localhost:8000/api/v1/ingest \
			-F "file=@$$pdf" | python -m json.tool; \
	done

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"
