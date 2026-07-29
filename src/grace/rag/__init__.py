"""RAG (Retrieval-Augmented Generation) Subsystem for Grace.

Provides sub-millisecond local semantic chunking and vector retrieval
to eliminate context prefill latency on long documents and PDFs.
"""

from grace.rag.retriever import LocalRagIndex, DocumentChunker

__all__ = ["LocalRagIndex", "DocumentChunker"]
