import os, tempfile, streamlit as st
from config import Config
from rag_pipeline import RAGPipeline
from utils.loaders import load_pdfs_to_docs
from utils.cache import file_hash, load_docs_cache, save_docs_cache

st.set_page_config(page_title="Production-Ready RAG", layout="wide")
st.title("📚 RAG — Multi-PDF Semantic Search")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.write(f"Embeddings: `{Config.EMBEDDING_MODEL}`")
    st.write(f"LLM: `{Config.LLM_HF_MODEL}`")
    st.write(f"Chunks: {Config.CHUNK_SIZE} (overlap {Config.CHUNK_OVERLAP})")
    if Config.HF_TOKEN:
        st.success("HuggingFace Hub token detected")
    else:
        st.info("No HF token — using local model")

# Ensure folders
os.makedirs(Config.INDEX_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# Session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "docs" not in st.session_state:
    st.session_state.docs = []
if "hash_tag" not in st.session_state:
    st.session_state.hash_tag = None

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

def persist_uploaded(files):
    tmp = tempfile.mkdtemp()
    paths = []
    for f in files:
        path = os.path.join(tmp, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        paths.append(path)
    return paths

build_col, clear_col = st.columns(2)
build_clicked = build_col.button("Build Index")
clear_clicked = clear_col.button("Clear Session")

if clear_clicked:
    st.session_state.clear()
    st.experimental_rerun()

if uploaded_files and build_clicked:
    saved_paths = persist_uploaded(uploaded_files)
    tag = "-".join(file_hash(p) for p in saved_paths)
    st.session_state.hash_tag = tag

    cached = load_docs_cache(Config.CACHE_DIR, tag)
    if cached:
        st.info("Loaded cached documents")
        docs = cached
    else:
        docs = load_pdfs_to_docs(saved_paths)
        if not docs:
            st.error("No text extracted from PDFs")
        else:
            save_docs_cache(Config.CACHE_DIR, tag, docs)
            st.success("Parsed & cached PDFs")
    st.session_state.docs = docs

    st.info("Building FAISS index...")
    ok, err = st.session_state.pipeline.build_index(st.session_state.docs)
    if ok:
        st.success("Index ready!")
    else:
        st.error(f"Failed: {err}")

# Question-answer UI
if st.session_state.docs:
    st.subheader("Ask a question about your PDFs")
    q = st.text_input("Question")
    ask = st.button("Get Answer")

    if ask and q.strip():
        answer, sources, err = st.session_state.pipeline.ask(q)
        if err:
            st.error(err)
        else:
            st.success(answer)
            with st.expander("Sources"):
                for i, src in enumerate(sources, 1):
                    meta = src.metadata or {}
                    st.write(f"{i}. {meta.get('source_file','unknown')} — page {meta.get('page','?')}")
else:
    st.info("Upload PDFs and click 'Build Index' to start.")
