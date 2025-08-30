import os, hashlib, pickle
from typing import List
from langchain.schema import Document

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def docs_cache_path(cache_dir: str, tag: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{tag}_docs.pkl")

def save_docs_cache(cache_dir: str, tag: str, docs: List[Document]) -> None:
    path = docs_cache_path(cache_dir, tag)
    with open(path, "wb") as f:
        pickle.dump(docs, f)

def load_docs_cache(cache_dir: str, tag: str):
    path = docs_cache_path(cache_dir, tag)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
