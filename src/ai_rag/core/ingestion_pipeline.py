# src/ai_rag/core/ingestion_pipeline.py
"""
Ingestion pipeline: raw text → split → embed → store.
Orchestrates TextSplitter + VectorStore.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ai_rag.core.text_splitter import text_splitter
from ai_rag.core.vector_store import vector_store

logger = logging.getLogger(__name__)


async def ingest_texts(
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    batch_size: int = 100,
) -> dict:
    """
    Full ingestion pipeline.
    Returns stats: {chunks_created, docs_stored, total_in_collection}.
    """
    if not texts:
        return {"chunks_created": 0, "docs_stored": 0, "total_in_collection": await vector_store.get_count()}

    # Step 1: Split
    chunks, chunk_metas = text_splitter.split_documents(texts, metadatas)

    # Step 2: Batch upsert (avoid OOM on large datasets)
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i : i + batch_size]
        batch_metas = chunk_metas[i : i + batch_size]
        stored += await vector_store.add_documents(batch_texts, batch_metas)

    stats = {
        "chunks_created": len(chunks),
        "docs_stored": stored,
        "total_in_collection": await vector_store.get_count(),
    }
    logger.info("?? Ingestion complete: %s", stats)
    return stats