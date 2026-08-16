# \src\ai_rag\services\embedding.py
from __future__ import annotations

import logging
import threading
from typing import List

from sentence_transformers import SentenceTransformer

from src.ai_rag.core.config import rag_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """BGE embedding service with thread-safe lazy loading."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or rag_config.EMBED_MODEL_PATH
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self.dimension: int = 0

    def _ensure_loaded(self) -> None:
        """线程安全的延迟加载（双重检查锁）"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("🔄 Loading BGE model: %s", self.model_path)
                    self._model = SentenceTransformer(self.model_path)

                    # ✅ 兼容 sentence-transformers v2/v3 API
                    if hasattr(self._model, "get_embedding_dimension"):
                        self.dimension = self._model.get_embedding_dimension()
                    elif hasattr(self._model, "get_sentence_embedding_dimension"):
                        self.dimension = self._model.get_sentence_embedding_dimension()
                    else:
                        raise RuntimeError(
                            f"Unsupported sentence-transformers version: "
                            f"no dimension getter on {type(self._model).__name__}"
                        )
                    logger.info("✅ BGE model loaded | dim=%d", self.dimension)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (no instruction prefix)."""
        if not texts:
            return []
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        logger.debug("Embedded %d texts → dim=%d", len(texts), self.dimension)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed query WITH BGE retrieval instruction prefix."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        instruction = "为这个句子生成表示以用于检索中文文档："
        return self.embed_texts([f"{instruction}{query}"])[0]


# ✅ 模块级单例：创建时不加载模型，首次 embed 时才触发
embedding_service = EmbeddingService()