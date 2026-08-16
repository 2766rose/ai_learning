# src/ai_rag/tasks/document_tasks.py
import asyncio
import os
import logging
from pathlib import Path

from celery.utils.log import get_task_logger
from celery.exceptions import SoftTimeLimitExceeded, Retry
from ai_rag.tasks.celery_app import celery_app
from ai_rag.core.config import rag_config
from ai_rag.services.etl_service import ingest_document

logger = get_task_logger(__name__)


@celery_app.task(
    bind=True,
    name="ingest_document_task",
    max_retries=3,
    # ✅ 生产级加固：为单个任务设置独立的超时阈值
    # 如果不同文档大小差异极大，可在 .delay() 时通过 time_limit 参数动态覆盖
    soft_time_limit=300,
    time_limit=310,
)
def ingest_document_task(self, file_path: str, metadata: dict = None) -> dict:
    """异步 ETL 任务：解析 → 智能分块 → 向量化 → 写入 ChromaDB
    
    ⚠️ 幂等性要求：由于启用了 task_acks_late，Worker 崩溃可能导致任务重试。
    ingest_document 内部应保证重复执行不会产生重复数据（如基于 doc_id 去重）。
    """
    if metadata is None:
        metadata = {}

    task_id = metadata.get("task_id", self.request.id)

    logger.info(
        "🚀 document_ingest_started | task_id=%s | file=%s | retry=%d",
        task_id, file_path, self.request.retries,
    )

    # ==========================================
    # 1. 路径安全检查（防目录穿越攻击）
    # ==========================================
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
        # ✅ 在同步 Celery Worker 中正确驱动异步 ETL 管道
        # asyncio.run() 创建独立事件循环并在结束后自动清理，避免跨 task 状态污染
        chunk_count = asyncio.run(
            ingest_document(file_path=abs_path, metadata=metadata)
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

    except SoftTimeLimitExceeded:
        # 🛡️ 生产级加固：软超时优雅处理
        # 硬超时(time_limit)会直接杀进程无法捕获，软超时允许我们记录日志、清理资源
        logger.warning(
            "⏰ document_ingest_timeout | task_id=%s | file=%s | elapsed=300s",
            task_id, abs_path,
        )
        # TODO: 可在此处更新数据库状态为 TIMEOUT、删除临时文件、发送告警等
        # 重新抛出异常，让 Celery 将任务标记为 FAILURE
        raise

    except Exception as exc:
        logger.exception(
            "❌ document_ingest_failed | task_id=%s | file=%s | retry=%d",
            task_id, abs_path, self.request.retries,
        )
        # ✅ 指数退避重试：60s → 120s → 240s
        # 对于 OOM、网络抖动等瞬时故障有效；对于文件损坏等永久故障，max_retries 耗尽后自动失败
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))