# rag_pipeline.py — modular RAG pipeline using LangChain + HF
import logging
from typing import List, Tuple, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from utils.text import chunk_docs
from config import Config

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


class RAGPipeline:
    def __init__(self):
        # Initialize embeddings
        logger.info("Loading embeddings: %s", Config.EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.qa_chain = None

    def build_index(self, docs: List[Document]) -> Tuple[bool, Optional[str]]:
        """Chunk docs, create FAISS index and prepare retriever + chain."""
        try:
            logger.info("Chunking documents")
            chunks = chunk_docs(docs, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)

            logger.info("Building FAISS index with %d chunks", len(chunks))
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": Config.TOP_K})

            self._build_chain()
            return True, None
        except Exception as e:
            logger.exception("Failed to build index: %s", e)
            return False, str(e)

    def _build_chain(self):
        """Create the LLM and the RetrievalQA chain."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use ONLY the context to answer.\n"
                "If the answer is not in context, say: \"I don't know.\"\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            ),
        )

        # Hugging Face Hub or local fallback
        if Config.HF_TOKEN:
            logger.info("HF token detected: using HuggingFaceHub model: %s", Config.LLM_HF_MODEL)
            llm = HuggingFaceHub(repo_id=Config.LLM_HF_MODEL, client=None, model_kwargs={"temperature": 0})
        else:
            logger.info("No HF token: using local transformers pipeline: %s", Config.LLM_HF_MODEL)
            hf_pipe = pipeline("text2text-generation", model=Config.LLM_HF_MODEL, max_length=256, truncation=True)
            llm = HuggingFacePipeline(pipeline=hf_pipe)

        # Correctly pass retriever (not vectorstore/docstore)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def ask(self, question: str) -> Tuple[Optional[str], Optional[List[Document]], Optional[str]]:
        """Return (answer, source_documents, error_message)"""
        if not self.qa_chain:
            return None, None, "QA chain not initialized. Build index first."
        try:
            out = self.qa_chain({"query": question})
            answer = out.get("result")
            sources = out.get("source_documents", [])
            return answer, sources, None
        except Exception as e:
            logger.exception("Query failed: %s", e)
            return None, None, str(e)
