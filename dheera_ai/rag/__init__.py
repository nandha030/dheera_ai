"""
DheeraAI RAG (Retrieval Augmented Generation) Module.

Provides an all-in-one API for document ingestion:
Upload -> (OCR) -> Chunk -> Embed -> Vector Store
"""

from dheera_ai.rag.main import aingest, ingest

__all__ = ["ingest", "aingest"]


# Expose at dheera_ai.rag level for convenience
async def arag_ingest(*args, **kwargs):
    """Alias for aingest."""
    return await aingest(*args, **kwargs)


def rag_ingest(*args, **kwargs):
    """Alias for ingest."""
    return ingest(*args, **kwargs)

