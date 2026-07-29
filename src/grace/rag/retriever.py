"""Local RAG Retriever Engine for Grace.

Implements fast, offline semantic chunking and TF-IDF vector retrieval.
Delivers sub-millisecond document chunk matching with zero cloud dependencies.
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("grace.rag")


class DocumentChunker:
    """Splits raw document text into overlapping semantic chunks."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
        """Split text into overlapping chunks of approximately chunk_size chars."""
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= chunk_size:
            return [cleaned]

        chunks = []
        start = 0
        text_len = len(cleaned)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                # Try breaking at sentence or space boundary
                last_space = cleaned.rfind(" ", start + chunk_size // 2, end)
                if last_space != -1:
                    end = last_space

            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = max(start + 1, end - overlap)

        return chunks


class LocalRagIndex:
    """Local sub-millisecond TF-IDF vector index for document retrieval."""

    def __init__(self):
        self._chunks: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

    @property
    def is_indexed(self) -> bool:
        return len(self._chunks) > 0 and self._tfidf_matrix is not None

    def index_text(self, doc_id: str, text: str, chunk_size: int = 400, overlap: int = 80) -> int:
        """Chunk and index document text. Returns number of indexed chunks."""
        chunks = DocumentChunker.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        self._chunks = chunks
        self._metadata = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        try:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._tfidf_matrix = self._vectorizer.fit_transform(self._chunks)
            logger.info(f"RAG Index created for '{doc_id}': {len(chunks)} chunks indexed.")
            return len(chunks)
        except Exception as e:
            logger.error(f"RAG Indexing failed for '{doc_id}': {e}")
            return 0

    def query(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Query top_k relevant chunks matching query_text."""
        if not self.is_indexed or not query_text.strip():
            return []

        try:
            query_vec = self._vectorizer.transform([query_text])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0.0:  # Only return chunks with non-zero similarity
                    results.append({
                        "chunk": self._chunks[idx],
                        "score": round(score, 4),
                        "doc_id": self._metadata[idx]["doc_id"],
                        "index": self._metadata[idx]["chunk_index"],
                    })
            return results
        except Exception as e:
            logger.error(f"RAG Query failed for '{query_text}': {e}")
            return []

    def get_summary_chunks(self, top_k: int = 3) -> list[str]:
        """Get top representative chunks across the document for summarization."""
        if not self._chunks:
            return []
        if len(self._chunks) <= top_k:
            return self._chunks

        # Pick evenly spaced chunks across start, middle, and end
        step = len(self._chunks) / top_k
        selected = [self._chunks[int(i * step)] for i in range(top_k)]
        return selected
