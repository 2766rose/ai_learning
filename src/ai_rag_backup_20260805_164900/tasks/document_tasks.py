# \src\ai_rag\tasks\document_tasks.py
import os
import logging
from pathlib import Path

from celery.utils.log import get_task_logger
from src.ai_rag.tasks.celery_app import celery_app
from src.ai_rag.core.config import rag_config
from src.ai_rag.services.etl_service import ingest_document

logger = get_task_logger(__name__)


@celery_app.task(bind=True, name="ingest_document_task", max_retries=3)
def ingest_document_task(self, file_path: str, metadata: dict = None) -> dict:
    """异步 ETL 任务：解析 → 智能分块 → 向量化 → 写入 ChromaDB"""
    if metadata is None:
        metadata = {}

    task_id = metadata.get("task_id", self.request.id)

    logger.info(
        "🚀 document_ingest_started | task_id=%s | file=%s | metadata=%s",
        task_id, file_path, metadata,
    )

    # 1. 路径安全检查（防目录穿越）
    abs_path = os.path.abspath(file_path)
    allowed_base = os.path.abspath(str(rag_config.UPLOAD_DIR))
    safe_prefix = allowed_base.rstrip(os.sep) + os.sep

    if not (abs_path.startswith(safe_prefix) or abs_path == allowed_base):
        logger.error("path_access_denied | task_id=%s | path=%s", task_id, abs_path)
        raise ValueError(f"Path access denied: {abs_path}")

    if not os.path.exists(abs_path):
        logger.error("file_not_found | task_id=%s | path=%s", task_id, abs_path)
        raise FileNotFoundError(f"File not found: {abs_path}")

    try:
        chunk_count = ingest_document(
            file_path=abs_path,
            metadata=metadata,
        )
        logger.info(
            "✅ document_ingested | task_id=%s | chunks=%d | file=%s",
            task_id, chunk_count, abs_path,
        )
        return {
            "status": "completed",
            "task_id": task_id,
            "chunk_count": chunk_count,
            "filename": metadata.get("source", ""),
        }
    except Exception as exc:
        logger.exception(
            "❌ document_ingest_failed | task_id=%s | file=%s",
            task_id, abs_path,
        )
        raise exc