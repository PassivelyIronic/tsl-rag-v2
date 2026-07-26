# ui.py
import os

import httpx
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSL Legal Assistant",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000/query")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* ── Header ── */
.tsl-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.tsl-header h1 {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    margin: 0;
}
.tsl-header p {
    color: #8b949e;
    font-size: 0.85rem;
    margin: 0.4rem 0 0 0;
    font-weight: 300;
}

/* ── Chat messages ── */
.user-bubble {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px 8px 2px 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #e6edf3;
    font-size: 0.95rem;
    line-height: 1.6;
}
.assistant-bubble {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 0 8px 8px 8px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    color: #e6edf3;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ── Metrics bar ── */
.metrics-bar {
    display: flex;
    gap: 1rem;
    margin: 0.8rem 0 0.4rem 0;
    flex-wrap: wrap;
}
.metric-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    background: #161b22;
    border: 1px solid #30363d;
    color: #8b949e;
}
.metric-pill span {
    color: #3fb950;
    font-weight: 500;
}

/* ── Citation card ── */
.citation-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    display: flex;
    gap: 0.8rem;
    align-items: center;
}
.citation-doc {
    font-family: 'IBM Plex Mono', monospace;
    color: #58a6ff;
    font-size: 0.78rem;
    white-space: nowrap;
}
.citation-article {
    color: #8b949e;
    border-left: 1px solid #30363d;
    padding-left: 0.8rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #8b949e;
}

/* ── Input ── */
.stChatInput textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #3fb950 !important;
    box-shadow: 0 0 0 2px rgba(63,185,80,0.15) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #3fb950 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Parametry zapytania")
    top_k = st.slider("Kandydaci retrieval (top_k)", 5, 50, 20)
    rerank_top_n = st.slider("Wyniki po rerankingu", 1, 10, 5)
    show_debug = st.toggle("Pokaż szczegóły retrieval", value=False)

    st.markdown("---")
    st.markdown("### 📂 Korpus dokumentów")

    # Dynamiczne pobieranie listy dokumentów z backendu
    try:
        docs_resp = httpx.get(f"{API_URL}/documents", timeout=3.0)
        if docs_resp.status_code == 200:
            docs = docs_resp.json()
            for doc_id, title in docs.items():
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                    f'color:#58a6ff;margin-bottom:0.2rem">{doc_id}</div>'
                    f'<div style="font-size:0.75rem;color:#6e7681;margin-bottom:0.6rem">{title}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.75rem;color:#6e7681;">Brak listy (błąd API)</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<div style="font-size:0.75rem;color:#6e7681;">Oczekiwanie na API...</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Health check
    try:
        health = httpx.get("http://localhost:8000/query/health", timeout=3.0).json()
        pg = "🟢" if health.get("postgres") == "ok" else "🔴"
        llm = "🟢" if health.get("ollama") == "ok" else "🔴"
        st.markdown(f"{pg} PostgreSQL / pgvector")
        st.markdown(f"{llm} Ollama (LLM + embed)")
    except Exception:
        st.markdown("🔴 API niedostępne")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="tsl-header">
    <h1>⚖ TSL Legal Assistant</h1>
    <p>Wyszukiwanie semantyczne · EU Transport &amp; Logistics Compliance · EC 561/2006 · AETR · Dir. 2002/15</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render history ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🙋 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

        # Metrics bar
        meta = msg.get("meta", {})
        if meta:
            st.markdown(
                f'<div class="metrics-bar">'
                f'<div class="metric-pill">⏱ <span>{meta.get("latency_ms", "?")} ms</span></div>'
                f'<div class="metric-pill">📄 <span>{meta.get("chunks_in_context", "?")} chunks</span></div>'
                f'<div class="metric-pill">🤖 <span>{meta.get("model", "?")}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Citations
        citations = msg.get("citations", [])
        if citations:
            st.markdown(
                '<div style="font-size:0.78rem;color:#8b949e;margin:0.6rem 0 0.3rem 0">'
                "📚 Źródła</div>",
                unsafe_allow_html=True,
            )
            for cit in citations:
                article = cit.get("article") or "—"
                st.markdown(
                    f'<div class="citation-card">'
                    f'<span class="citation-doc">{cit.get("document_id", "")}</span>'
                    f'<span class="citation-article">Art. {article}</span>'
                    f'<span style="color:#6e7681;font-size:0.75rem;margin-left:auto">'
                    f'{cit.get("document_title","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Debug: retrieved chunks
        if show_debug and msg.get("chunks"):
            with st.expander("🔍 Szczegóły retrieval (debug)"):
                for i, chunk in enumerate(msg["chunks"]):
                    cols = st.columns([3, 1, 1, 1, 1])
                    cols[0].caption(
                        f"**{chunk['chunk']['metadata']['document_id']}** · {chunk['chunk']['metadata'].get('article','')}"
                    )
                    cols[1].caption(f"dense: {chunk['dense_score']:.3f}")
                    cols[2].caption(f"bm25: {chunk['bm25_score']:.2f}")
                    cols[3].caption(f"rrf: {chunk['hybrid_score']:.4f}")
                    cols[4].caption(f"rerank: {chunk.get('rerank_score', 0):.2f}")
                    st.caption(chunk["chunk"]["content"][:300] + "…")
                    if i < len(msg["chunks"]) - 1:
                        st.divider()

# ── Chat input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input("Zadaj pytanie o przepisach transportowych UE…"):
    # Render user bubble immediately
    st.markdown(
        f'<div class="user-bubble">🙋 {prompt}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Przeszukuję przepisy…"):
        try:
            resp = httpx.post(
                API_URL,
                json={
                    "query": prompt,
                    "top_k": top_k,
                    "rerank_top_n": rerank_top_n,
                    "debug": True,
                },
                timeout=120.0,
            )

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "Brak odpowiedzi.")
                citations = data.get("citations", [])
                chunks = data.get("retrieved_chunks", [])
                meta = {
                    "latency_ms": data.get("latency_ms"),
                    "chunks_in_context": data.get("metadata", {}).get("chunks_in_context"),
                    "model": data.get("model_used", "").split(":")[0],
                }

                # Render answer
                st.markdown(
                    f'<div class="assistant-bubble">{answer}</div>',
                    unsafe_allow_html=True,
                )

                # Metrics
                st.markdown(
                    f'<div class="metrics-bar">'
                    f'<div class="metric-pill">⏱ <span>{meta["latency_ms"]} ms</span></div>'
                    f'<div class="metric-pill">📄 <span>{meta["chunks_in_context"]} chunks</span></div>'
                    f'<div class="metric-pill">🤖 <span>{meta["model"]}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Citations
                if citations:
                    st.markdown(
                        '<div style="font-size:0.78rem;color:#8b949e;margin:0.6rem 0 0.3rem 0">'
                        "📚 Źródła</div>",
                        unsafe_allow_html=True,
                    )
                    for cit in citations:
                        article = cit.get("article") or "—"
                        st.markdown(
                            f'<div class="citation-card">'
                            f'<span class="citation-doc">{cit.get("document_id","")}</span>'
                            f'<span class="citation-article">Art. {article}</span>'
                            f'<span style="color:#6e7681;font-size:0.75rem;margin-left:auto">'
                            f'{cit.get("document_title","")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Debug panel
                if show_debug and chunks:
                    with st.expander("🔍 Szczegóły retrieval (debug)"):
                        for i, chunk in enumerate(chunks):
                            cols = st.columns([3, 1, 1, 1, 1])
                            cols[0].caption(
                                f"**{chunk['chunk']['metadata']['document_id']}** · "
                                f"{chunk['chunk']['metadata'].get('article','')}"
                            )
                            cols[1].caption(f"dense {chunk['dense_score']:.3f}")
                            cols[2].caption(f"bm25 {chunk['bm25_score']:.2f}")
                            cols[3].caption(f"rrf {chunk['hybrid_score']:.4f}")
                            cols[4].caption(f"rerank {chunk.get('rerank_score', 0):.2f}")
                            st.caption(chunk["chunk"]["content"][:300] + "…")
                            if i < len(chunks) - 1:
                                st.divider()

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "chunks": chunks,
                        "meta": meta,
                    }
                )

            else:
                st.error(f"Błąd API: {resp.status_code} — {resp.text[:200]}")

        except httpx.ConnectError:
            st.error("❌ Nie można połączyć z API. Upewnij się że `uv run python main.py` działa.")
        except Exception as e:
            st.error(f"Błąd: {e}")
