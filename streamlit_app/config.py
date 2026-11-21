"""UI-level configuration (not duplicating backend settings)."""

import os

API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")

UI_CONFIG = {
    "query_timeout_seconds": 180,
    "ingest_timeout_seconds": 300,
    "max_upload_mb": 50,
    "example_questions": [
        "Which crop is suitable for black soil?",
        "How do I increase nitrogen content in my soil?",
        "What fertilizer is recommended for sugarcane?",
        "What causes yellow leaves in cotton plants?",
        "Which crops are suitable during Kharif season in Maharashtra?",
        "How to control aphids in wheat?",
        "What is the ideal pH for growing rice?",
        "How much water does cotton require per hectare?",
    ],
}
