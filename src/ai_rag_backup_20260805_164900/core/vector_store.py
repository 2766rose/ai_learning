# src/ai_rag/core/vector_store.py
from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ai_rag.core.config import rag_config
from src.ai_rag.core.embeddings import embedding_service

logger = logging.getLogger(__name__)

# ✅ 企业规范: Settings 固化为模块级常量，保证全局唯一对象引用
# ChromaDB v0.6+ SharedSystemClient 通过对象引用(而非值)判断一致性
_CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=rag_config.CHROMA_PERSIST_DIR,
)

_UPSERT_BATCH_SIZE = 500
_lock = threading.Lock()
_vector_store_instance: Optional["VectorStore"] = None


class VectorStore:
    """Thread-safe ChromaDB wrapper with enterprise-grade initialization."""

    def __init__(self) -> None:
        logger.info("📂 Initializing ChromaDB at: %s", rag_config.CHROMA_PERSIST_DIR)
        # ✅ 始终使用同一个 Settings 对象引用
        self._client = chromadb.PersistentClient(
            path=rag_config.CHROMA_PERSIST_DIR,
            settings=_CHROMA_SETTINGS,
        )

        collection_name = rag_config.CHROMA_COLLECTION_NAME
        expected_space = rag_config.CHROMA_VECTOR_SPACE
        existing = self._client.list_collections()

        if collection_name in [c.name for c in existing]:
            self._collection = self._client.get_collection(name=collection_name)
            current_metadata = self._collection.metadata or {}
            current_space = current_metadata.get("hnsw:space")

            if current_space != expected_space:
                raise ValueError(  # ✅ 企业规范: 配置漂移应快速失败，而非仅 warning
                    f"Collection '{collection_name}' space mismatch: "
                    f"db={current_space}, config={expected_space}. "
                    f"Please delete and recreate the collection."
                )
        else:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": expected_space},
            )

        logger.info(
            "✅ ChromaDB ready | collection=%s | space=%s | count=%d",
            collection_name, expected_space, self._collection.count(),
        )

    def _get_fresh_collection(self) -> chromadb.Collection:
        """✅ FIX: 获取最新的 Collection 引用，解决跨进程写入后 HNSW 段缓存失效问题。

        ChromaDB PersistentClient 不支持跨进程实时同步。
        get_or_create_collection 是轻量元数据重载操作，不会重建索引，
        但会刷新 Rust 底层的 HNSW segment reader 句柄。
        """
        return self._client.get_or_create_collection(
            name=rag_config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": rag_config.CHROMA_VECTOR_SPACE},
        )

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        """批量写入文档到向量库，自动分批 upsert。"""
        if not texts:
            return 0

        embeddings = embedding_service.embed_texts(texts)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{}] * len(texts)

        total_added = 0
        for start in range(0, len(texts), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            try:
                self._collection.upsert(
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )
                total_added += min(_UPSERT_BATCH_SIZE, len(texts) - start)
            except Exception as e:
                # ⚠️ 写入失败属于严重故障，保留完整 traceback 用于排查
                logger.exception("❌ Upsert batch [%d:%d] failed: %s", start, end, e)
                raise

        logger.info("📥 Upserted %d chunks → total=%d", total_added, self._collection.count())
        return total_added

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """语义检索，每次查询前刷新 Collection 引用以确保读到最新数据。"""
        k = top_k or rag_config.TOP_K
        try:
            query_embedding = embedding_service.embed_query(query)

            # ✅ FIX: 使用新鲜引用替代缓存的 self._collection
            collection = self._get_fresh_collection()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            hits: List[Dict] = []
            result_ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for i in range(len(result_ids)):
                hits.append({
                    "id": result_ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                })

            logger.info("🔍 Query='%s...' → %d hits (top_k=%d)", query[:30], len(hits), k)
            return hits
        except Exception as e:
            logger.error("❌ VectorStore.search failed: %s", e)
            return []

    @property
    def count(self) -> int:
        """返回当前 Collection 中的文档总数（使用新鲜引用）。"""
        # ✅ FIX: 与 search 保持一致，避免返回过期的 count
        return self._get_fresh_collection().count()


def get_vector_store() -> VectorStore:
    """✅ 企业规范: 双重检查锁(DCL)线程安全单例。"""
    global _vector_store_instance
    if _vector_store_instance is None:
        with _lock:
            if _vector_store_instance is None:
                _vector_store_instance = VectorStore()
    return _vector_store_instance


class _LazyVectorStore:
    """延迟初始化代理，避免 import 时立即连接 ChromaDB。"""

    def __getattr__(self, name: str):
        return getattr(get_vector_store(), name)


vector_store = _LazyVectorStore()