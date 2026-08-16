# src/ai_rag/tasks/etl_service.py
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

_parsers_cache: Dict[str, Any] = {}


def _get_parser(ext: str):
    """延迟导入并缓存解析器实例，避免重复初始化开销。"""
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


async def ingest_document(file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """
    ETL 管道：解析 → 分块 → 去重 → 向量化 → 写入向量库。

    ⚠️ 幂等性保证：基于 task_id 生成确定性 chunk ID，
    配合 vector_store.add_documents 的 upsert 语义，确保重试不会产生重复数据。

    Args:
        file_path: 待处理文件的绝对路径。
        metadata: 附加元数据，必须包含 task_id 用于生成唯一 chunk ID。

    Returns:
        成功写入的 chunk 数量。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    logger.info("🔄 开始 ETL 管道: %s", file_path)
    metadata = metadata or {}

    # === Step 1: 文档解析（CPU 密集型，释放事件循环）===
    ext = os.path.splitext(file_path)[1].lower()
    parser = _get_parser(ext)
    
    # ✅ 加固1: CPU 密集操作放入线程池，防止阻塞 asyncio 事件循环
    # 否则软超时(SoftTimeLimitExceeded)无法被及时响应
    loop = asyncio.get_running_loop()
    raw_text = await loop.run_in_executor(None, parser.parse_file, file_path)
    
    if not raw_text or not raw_text.strip():
        logger.warning("⚠️ 文件内容为空: %s", file_path)
        return 0

    # === Step 2: 文本分块 ===
    # 切分器模式：recursive(默认) | naive | semantic（环境变量 RAG_SPLITTER 切换）
    # 2026-08-14：默认改用 recursive，避免朴素 500 字滑动窗口把多个主题揉进一个块，导致检索不准
    splitter_mode = os.environ.get("RAG_SPLITTER", "recursive").strip().lower()
    chunk_size = 500
    chunk_overlap = 50
    if splitter_mode == "semantic":
        from ai_rag.rag.splitters.semantic_chunker import SemanticChunker
        chunks: List[str] = SemanticChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_text(raw_text)
    elif splitter_mode == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        chunks: List[str] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        ).split_text(raw_text)
    else:  # naive：保持原有按字符滑动窗口行为
        chunks = [
            raw_text[i : i + chunk_size]
            for i in range(0, len(raw_text), chunk_size - chunk_overlap)
        ]
    logger.info("📦 分块完成: %d chunks", len(chunks))

    # === ✅ Step 2.5: 入库前去重 ===
    from src.ai_rag.utils.dedup import deduplicate_chunks

    chunk_dicts = [{"content": c} for c in chunks]
    unique_chunk_dicts = deduplicate_chunks(chunk_dicts, key_field="content")
    chunks = [d["content"] for d in unique_chunk_dicts]

    if not chunks:
        logger.warning("⚠️ 去重后无有效 chunk: %s", file_path)
        return 0

    # === Step 3: 向量化 + 写入 ===
    task_id = metadata.get("task_id", "unknown")
    # ✅ 加固2: 确定性 ID 保证幂等性（task_acks_late 重试安全）
    ids = [f"{task_id}-chunk-{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    from src.ai_rag.core.vector_store import get_vector_store

    # ✅ 加固3: 资源安全获取，支持上下文管理或显式关闭
    vector_store = await get_vector_store()
    try:
        added = await vector_store.add_documents(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )
    finally:
        # 如果 vector_store 持有连接池/客户端，确保释放
        # 若 get_vector_store() 返回的是全局单例且无需关闭，可移除 finally
        if hasattr(vector_store, 'close'):
            await vector_store.close()

    logger.info("✅ 入库完成: %d chunks -> %s", added, file_path)
    return added