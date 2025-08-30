import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_pdfs_to_docs(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        loader = PyPDFLoader(path)
        file_docs = loader.load()
        for d in file_docs:
            metadata = dict(d.metadata) if d.metadata else {}
            metadata["source_file"] = os.path.basename(path)
            docs.append(Document(page_content=d.page_content, metadata=metadata))
    return docs
