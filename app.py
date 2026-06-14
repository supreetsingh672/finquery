import streamlit as st
from pathlib import Path

from src.ingestion.indexer import get_embedding_model, get_chroma_collection, _load_parents
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import get_reranker, rerank
from src.generation.query_expander import expand_query
from src.generation.generator import generate_answer, get_citations
from src.config import DOCUMENT_NAMES, TOP_K_RERANK

st.set_page_config(
    page_title="FinQuery — RBI Regulations",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker_model():
    return get_reranker()


def check_index_ready() -> bool:
    try:
        collection = get_chroma_collection()
        return collection.count() > 0
    except Exception:
        return False


def render_sidebar():
    with st.sidebar:
        st.title("🏦 FinQuery")
        st.caption("RAG over RBI Regulatory Documents")
        st.divider()

        st.subheader("Knowledge Base")
        for name in DOCUMENT_NAMES.values():
            st.markdown(f"• {name}")

        st.divider()
        st.subheader("RAG Pipeline")
        st.markdown("""
**Retrieval**
- Hybrid search (Vector + BM25)
- Reciprocal Rank Fusion
- BGE Cross-encoder Reranker

**Generation**
- Multi-query expansion
- Parent-child chunking
- Streaming via Groq
        """)
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def render_citations(citations):
    with st.expander(f"📄 Sources ({len(citations)})", expanded=False):
        for c in citations:
            st.markdown(
                f"**{c['ref']}** — *{c['doc_name']}*  \n"
                f"Page {c['page_number']}  ·  {c['section_hint']}"
            )
            st.divider()


def main():
    render_sidebar()

    st.title("FinQuery")
    st.caption("Ask questions about Indian financial regulations — get answers with citations")

    if not check_index_ready():
        st.error("Index not found. Please run `python ingest.py` first.")
        st.code("python ingest.py", language="bash")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations" not in st.session_state:
        st.session_state.citations = {}

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i in st.session_state.citations:
                render_citations(st.session_state.citations[i])

    question = st.chat_input("Ask about RBI regulations, e.g. What are KYC requirements for NRIs?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    embedder = load_embedder()
    load_reranker_model()

    with st.chat_message("assistant"):
        status = st.empty()

        status.markdown("*Expanding query...*")
        queries = expand_query(question)

        status.markdown(f"*Searching across {len(queries)} query variants...*")
        candidates = hybrid_search(queries, embedder)

        status.markdown("*Re-ranking results...*")
        top_chunks = rerank(question, candidates, top_k=TOP_K_RERANK)

        status.empty()

        answer_box = st.empty()
        full_answer = ""
        for token in generate_answer(question, top_chunks):
            full_answer += token
            answer_box.markdown(full_answer + "▌")
        answer_box.markdown(full_answer)

        citations = get_citations(question, top_chunks)
        msg_index = len(st.session_state.messages)
        st.session_state.citations[msg_index] = citations
        render_citations(citations)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})


if __name__ == "__main__":
    main()
