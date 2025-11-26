from __future__ import annotations

import os
# Allow duplicate OpenMP libraries (Fixes OMP Error #15 on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorMemory:
    """In-memory FAISS-backed vector store for MATLAB code chunks."""

    def __init__(self, embedding_model: str = "text-embedding-3-small") -> None:
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store: Optional[FAISS] = None

    def index_chunks(self, chunks: List[str]) -> None:
        """Embed MATLAB code chunks and populate the FAISS index."""
        if not chunks:
            self.vector_store = None
            return

        documents = [Document(page_content=chunk) for chunk in chunks]
        self.vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Return the top-k most relevant chunks for a natural-language query."""
        if not self.vector_store:
            raise ValueError("Vector store is empty. Call index_chunks before querying.")

        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]