# src/ai_rag/core/lifespan.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.ai_rag.core.config import rag_config
from src.ai_rag.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：预热核心单例，确保启动时所有基础设施就绪。"""
    configure_logging()
    rag_config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Loading RAG resources...")

    # ✅ 1. 预热 Embedding 服务单例
    # embedding_service 内部已封装 SentenceTransformer 加载与缓存，
    # 此处调用 embed_query 触发首次模型加载，避免首个请求延迟。
    from src.ai_rag.core.embeddings import embedding_service
    try:
        embedding_service.embed_query("warmup")
        logger.info("✅ Embedding model warmed up successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Embedding model warmup failed: {e}") from e

    # ✅ 2. 预热 VectorStore 单例
    # 通过统一入口初始化 ChromaDB，确保 Settings 全局一致、Collection 引用正确。
    from src.ai_rag.core.vector_store import get_vector_store
    try:
        store = get_vector_store()
        logger.info(
            "✅ ChromaDB ready | collection=%s | path=%s | count=%d",
            rag_config.CHROMA_COLLECTION_NAME,
            rag_config.CHROMA_PERSIST_DIR,
            store.count,
        )
    except Exception as e:
        raise RuntimeError(f"❌ ChromaDB initialization failed: {e}") from e

    # ✅ 3. 预热 RAGEngine（可选但推荐）
    # 确保 OpenAI client 和内部依赖在启动阶段完成初始化，
    # 而非等到第一个用户请求时才触发。
    try:
        from src.ai_rag.services.rag_engine import RAGEngine
        RAGEngine()
        logger.info("✅ RAGEngine initialized")
    except Exception as e:
        logger.warning("⚠️ RAGEngine pre-init skipped: %s", e)

    try:
        yield
    finally:
        # ✅ 资源释放说明：
        # - embedding_service: 进程级单例，随 Python 进程退出自动 GC
        # - VectorStore / ChromaDB PersistentClient: 进程级单例，SQLite WAL 由 OS 回收
        # - AsyncOpenAI client: httpx.AsyncClient 支持 __del__ 安全关闭
        logger.info("🛑 RAG resources released (process-level singletons auto-cleaned)")