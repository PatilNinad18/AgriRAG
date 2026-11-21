"""
streamlit_app/app.py
---------------------
Agriculture RAG Assistant — Streamlit chat UI.
"""

from __future__ import annotations
import streamlit as st

try:
    from streamlit_app.api_client import AgriAPIClient, APIError, QueryResult
    from streamlit_app.config import API_BASE_URL, UI_CONFIG
except ModuleNotFoundError:
    from api_client import AgriAPIClient, APIError, QueryResult
    from config import API_BASE_URL, UI_CONFIG

st.set_page_config(
    page_title="AgriRAG Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.source-card {
    background: #f8fdf5; border-left: 4px solid #2e7d32;
    border-radius: 4px; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem;
}
.source-card-header { font-weight: 600; color: #1b5e20; margin-bottom: 4px; }
.score-badge {
    display: inline-block; background: #e8f5e9; color: #2e7d32;
    border-radius: 12px; padding: 1px 8px; font-size: 0.78rem; margin-left: 8px;
}
.timing-meta { font-size: 0.78rem; color: #757575; margin-top: 6px; }
.rag-badge {
    display: inline-block; background: #e3f2fd; color: #1565c0;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; margin-bottom: 6px;
}
.fallback-badge {
    display: inline-block; background: #fff8e1; color: #f57f17;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; margin-bottom: 6px;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "client" not in st.session_state:
    st.session_state.client = AgriAPIClient(API_BASE_URL)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "_uploaded_name" not in st.session_state:
    st.session_state["_uploaded_name"] = None
if "_uploaded_bytes" not in st.session_state:
    st.session_state["_uploaded_bytes"] = None
if "_uploaded_size" not in st.session_state:
    st.session_state["_uploaded_size"] = 0
if "_ingest_result" not in st.session_state:
    st.session_state["_ingest_result"] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def render_source_citations(result: QueryResult) -> None:
    # Show RAG mode or fallback mode badge
    if result.has_sources and result.sources:
        st.markdown(
            f'<span class="rag-badge">📄 RAG Mode — answer grounded in {len(result.sources)} PDF source(s)</span>',
            unsafe_allow_html=True,
        )
        with st.expander(f"📚 Sources ({len(result.sources)} documents retrieved)", expanded=False):
            for i, src in enumerate(result.sources, 1):
                st.markdown(
                    f"""<div class="source-card">
                        <div class="source-card-header">
                            [{i}] {src.source}
                            <span class="score-badge">score: {src.score:.2f}</span>
                            &nbsp;·&nbsp; page {src.page}
                        </div>
                        <div>{src.excerpt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<span class="fallback-badge">🧠 Fallback Mode — answered from AgriParam model knowledge (no matching PDF found)</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="timing-meta">⏱ Retrieval: {result.retrieval_time_ms:.0f} ms &nbsp;|&nbsp; '
        f'Generation: {result.generation_time_ms:.0f} ms &nbsp;|&nbsp; '
        f'Total: {result.total_time_ms:.0f} ms</div>',
        unsafe_allow_html=True,
    )


def submit_question(question: str) -> None:
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    try:
        with st.spinner("🌾 Thinking… retrieving documents and generating answer…"):
            result = st.session_state.client.query(question)

        if result.error:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ An error occurred: {result.error}",
                "result": None,
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "result": result,
            })
    except APIError as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ API Error: {exc}",
            "result": None,
        })

    st.session_state.pending_question = ""


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌾 AgriRAG Assistant")
    st.caption("Powered by AgriParam LLM + Qdrant Vector Search")
    st.divider()

    # System status
    st.subheader("System Status")
    try:
        health = st.session_state.client.health()
        status_emoji = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(
            health.get("status", ""), "⚪"
        )
        st.write(f"{status_emoji} **{health.get('status', 'unknown').title()}**")
        comps = health.get("components", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Qdrant", "✓" if comps.get("qdrant") else "✗")
        col2.metric("Embedder", "✓" if comps.get("embedding_model") else "✗")
        col3.metric("LLM", "✓" if comps.get("llm") else "✗")
        info = health.get("collection_info")
        if info:
            count = info.get("points_count", 0)
            st.caption(f"📦 {count:,} vectors indexed")
            if count == 0:
                st.warning("⚠️ Knowledge base is empty. Upload a PDF to enable RAG mode.")
    except APIError as exc:
        st.error(f"⚠️ {exc}")

    st.divider()

    # PDF upload
    st.subheader("📄 Add Knowledge")
    uploaded_file = st.file_uploader(
        "Upload an agriculture PDF",
        type=["pdf"],
        help="Upload crop guides, soil manuals, fertiliser charts, etc.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if file_bytes:
            st.session_state["_uploaded_name"] = uploaded_file.name
            st.session_state["_uploaded_bytes"] = file_bytes
            st.session_state["_uploaded_size"] = len(file_bytes)
            st.session_state["_ingest_result"] = None

    if st.session_state["_uploaded_bytes"] is not None:
        fname = st.session_state["_uploaded_name"]
        fbytes = st.session_state["_uploaded_bytes"]
        fsize = st.session_state["_uploaded_size"]
        max_bytes = UI_CONFIG["max_upload_mb"] * 1024 * 1024

        st.caption(f"📎 **{fname}** ({fsize / 1024:.0f} KB ready)")

        if fsize > max_bytes:
            st.error(f"File too large. Max: {UI_CONFIG['max_upload_mb']} MB")
        else:
            if st.button("📥 Ingest into knowledge base", use_container_width=True):
                with st.spinner(f"Ingesting {fname}…"):
                    try:
                        result = st.session_state.client.ingest(fname, fbytes)
                        st.session_state["_ingest_result"] = result
                        if result.success:
                            st.session_state["_uploaded_bytes"] = None
                            st.session_state["_uploaded_name"] = None
                    except APIError as exc:
                        st.session_state["_ingest_result"] = None
                        st.error(f"❌ {exc}")

        res = st.session_state.get("_ingest_result")
        if res is not None:
            if res.success:
                st.success(
                    f"✅ **{res.filename}** ingested!\n\n"
                    f"Chunks: {res.chunks_created} → Vectors: {res.vectors_stored}"
                )
            else:
                st.error(f"❌ Ingestion failed: {res.error}")

    st.divider()

    # Example questions
    st.subheader("💡 Example Questions")
    for example in UI_CONFIG["example_questions"]:
        if st.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state.pending_question = example

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat ──────────────────────────────────────────────────────────────────

st.header("Ask an Agriculture Question")
st.caption("Query crop suitability, soil management, fertilisers, pest control, seasons, and more.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌾"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            render_source_citations(msg["result"])

default_input = st.session_state.pop("pending_question", "") or ""
user_input = st.chat_input(
    "Type your agriculture question here…",
)

question_to_ask = user_input or (default_input if default_input else None)
if question_to_ask:
    submit_question(question_to_ask)
    st.rerun()

if not st.session_state.messages:
    st.info(
        "**How this works:**\n\n"
        "📄 **RAG Mode** — If your question matches content in an uploaded PDF, "
        "the answer is grounded in that document with source citations.\n\n"
        "🧠 **Fallback Mode** — If no matching PDF content is found, "
        "AgriParam answers from its own agricultural training knowledge.\n\n"
        "👈 Upload agriculture PDFs in the sidebar to expand the knowledge base."
    )
