import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    LLM_HF_MODEL = os.getenv("LLM_HF_MODEL", "google/flan-t5-large")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K = int(os.getenv("TOP_K", 10))
    INDEX_DIR = os.getenv("INDEX_DIR", "storage/index")
    CACHE_DIR = os.getenv("CACHE_DIR", "storage/cache")
