# src/ai_rag/services.py
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

_parsers_cache: Dict[str, Any] = {}


def _get_parser(ext: str):
    """延迟导入并缓存解析器"""
    if ext not in _parsers_cache:
        if ext == ".pdf":
            from src.ai_rag.parsers.pdf_parser import PDFParser
            _parsers_cache[ext] = PDFParser()
        elif ext == ".docx":
            from src.ai_rag.parsers.docx_parser import DocxParser
            _parsers_cache[ext] = DocxParser()
        elif ext in (".xlsx", ".xls"):
            from src.ai_rag.parsers.excel_parser import ExcelParser
            _parsers_cache[ext] = ExcelParser()
        elif ext in (".txt", ".md", ".csv", ".log"):
            from src.ai_rag.parsers.txt_parser import TxtParser
            _parsers_cache[ext] = TxtParser()
        else:
            raise ValueError(f"暂不支持的文件类型: {ext}")
    return _parsers_cache[ext]


def ingest_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """ETL pipeline: parse → chunk → dedup → embed → upsert via public API."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    logger.info("🔄 开始 ETL 管道: %s", file_path)
    metadata = metadata or {}

    # === Step 1: 文档解析 ===
    ext = os.path.splitext(file_path)[1].lower()
    parser = _get_parser(ext)
    raw_text = parser.parse_file(file_path)
    if not raw_text or not raw_text.strip():
        logger.warning("⚠️ 文件内容为空: %s", file_path)
        return 0

    # === Step 2: 文本分块 ===
    chunk_size = 500
    chunk_overlap = 50
    chunks: List[str] = [
        raw_text[i : i + chunk_size]
        for i in range(0, len(raw_text), chunk_size - chunk_overlap)
    ]
    logger.info("📦 分块完成: %d chunks", len(chunks))

    # === ✅ Step 2.5: 入库前去重 ===
    from src.ai_rag.utils.dedup import deduplicate_chunks

    chunk_dicts = [{"text": c} for c in chunks]
    unique_chunk_dicts = deduplicate_chunks(chunk_dicts, key_field="text")
    chunks = [d["text"] for d in unique_chunk_dicts]

    if not chunks:
        logger.warning("⚠️ 去重后无有效chunk: %s", file_path)
        return 0

    # === Step 3: 向量化 + 写入（统一走公开 API）===
    task_id = metadata.get("task_id", "unknown")
    ids = [f"{task_id}-chunk-{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    # ✅ 通过公开接口写入，内部已包含批次处理 + 异常上抛
    # Celery task 层负责重试，此处不再重复实现
    from src.ai_rag.core.vector_store import get_vector_store

    vector_store = get_vector_store()
    added = vector_store.add_documents(
        texts=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info("✅ 入库完成: %d chunks -> %s", added, file_path)
    return added