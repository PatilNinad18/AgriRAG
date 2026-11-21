import sys
import traceback

def p(msg):
    print(msg, flush=True)

p("=" * 60)
p("AgriRAG Startup Diagnostics v2")
p("=" * 60)

# Step 1: Config
try:
    from app.utils.config import settings
    p(f"[OK] Config loaded")
    p(f"     AGRIPARAM_DEVICE = {settings.agriparam_device}")
    p(f"     AGRIPARAM_MODEL  = {settings.agriparam_model_name}")
except Exception as e:
    p(f"[FAIL] Config: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Torch + CUDA
try:
    p("[....] Importing torch...")
    import torch
    p(f"[OK] Torch {torch.__version__}")
    p(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p(f"     GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        p(f"     VRAM free: {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")
except Exception as e:
    p(f"[FAIL] Torch: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Qdrant
try:
    p("[....] Connecting to Qdrant...")
    from app.services.qdrant_service import qdrant_service
    qdrant_service.connect()
    p("[OK] Qdrant connected")
except Exception as e:
    p(f"[FAIL] Qdrant: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Embedding model - broken into sub-steps
try:
    p("[....] Importing SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    p("[OK] SentenceTransformer imported")
except Exception as e:
    p(f"[FAIL] SentenceTransformer import: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    p("[....] Loading embedding model from HuggingFace...")
    model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.agriparam_device,
    )
    p("[OK] Embedding model object created")
except Exception as e:
    p(f"[FAIL] SentenceTransformer load: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    p("[....] Running test embedding...")
    import numpy as np
    vec = model.encode(["test agriculture query"], normalize_embeddings=True)
    p(f"[OK] Test embedding shape: {vec.shape}")
except Exception as e:
    p(f"[FAIL] Test embedding: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    p("[....] Loading embedding service...")
    from app.rag.embedder import embedding_service
    embedding_service.load()
    p("[OK] Embedding service loaded")
except Exception as e:
    p(f"[FAIL] Embedding service: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: AgriParam
try:
    p("[....] Importing AgriParam service...")
    from app.llm.agriparam import agriparam_service
    p("[OK] AgriParam service imported")
except Exception as e:
    p(f"[FAIL] AgriParam import: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    p("[....] Loading AgriParam model (30-60s)...")
    agriparam_service.load()
    p("[OK] AgriParam loaded")
except Exception as e:
    p(f"[FAIL] AgriParam load: {e}")
    traceback.print_exc()
    sys.exit(1)

p("=" * 60)
p("ALL SYSTEMS OK")
p("=" * 60)
