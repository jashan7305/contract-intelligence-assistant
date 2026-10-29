import streamlit as st
from src.config import load_config
from src.data import load_benchmark
from src.chunking import load_corpus_for_queries
from src.retrieval import Retriever

@st.cache_resource
def load_engine():
    cfg = load_config()
    rows = load_benchmark(cfg["data"]["benchmark_path"])
    passages = load_corpus_for_queries(rows, cfg["data"]["corpus_dir"],
        cfg["retrieval"]["chunk_chars"], cfg["retrieval"]["overlap_chars"])
    return Retriever(passages, cfg["retrieval"]["method"],
        cfg["retrieval"]["bm25_weight"], cfg["retrieval"]["tfidf_weight"]), cfg

st.set_page_config(page_title="Contract Intelligence Assistant", layout="wide")
st.title("Contract Intelligence Assistant")
st.caption("Evidence-first contract passage retrieval. This tool retrieves passages; it does not provide legal advice.")
query = st.text_area("Ask a question about the indexed contracts", placeholder="Does this agreement allow sublicensing to a third-party platform?")
if st.button("Search", type="primary", disabled=not query.strip()):
    engine, cfg = load_engine()
    results = engine.search(query, cfg["retrieval"]["top_k"], cfg["retrieval"]["min_score"])
    if not results:
        st.info("No passage met the configured relevance threshold.")
    for i, r in enumerate(results, 1):
        with st.expander(f"{i}. Score {r['score']:.3f} — {r['doc_path']} [{r['start']}:{r['end']}]", expanded=(i == 1)):
            st.code(f"Document: {r['doc_path']}\\nCharacter offsets: [{r['start']}:{r['end']}]")
            st.write(r["text"])
