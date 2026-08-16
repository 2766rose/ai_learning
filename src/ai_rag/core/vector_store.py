# src\ai_rag\core\vector_store.py
from __future__ import annotations

import asyncio
import logging
import uuid
from functools import partial
from typing import Dict, List, Optional, TypedDict

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_rag.core.config import rag_config
from ai_rag.core.embeddings import embedding_service

logger = logging.getLogger(__name__)

_CHROMA_SETTINGS = ChromaSettings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=rag_config.CHROMA_PERSIST_DIR,
)

_UPSERT_BATCH_SIZE = 500
_GET_BATCH_SIZE = 500
_GET_MAX_LIMIT = 10000


class RawDocument(TypedDict):
    """BM25 预热等场景的统一文档契约，字段名与 BM25Engine.index() 对齐。"""
    content: str
    metadata: dict


# ✅ 异步单例锁
_async_lock = asyncio.Lock()
_vector_store_instance: Optional["VectorStore"] = None


class VectorStore:
    """Enterprise-grade ChromaDB 1.x wrapper with asyncio.to_thread bridging."""

    def __init__(self) -> None:
        # ✅ __init__ 仅做轻量赋值，IO 全部延迟到 initialize()
        logger.info("📂 Preparing ChromaDB 1.x PersistentClient at: %s", rag_config.CHROMA_PERSIST_DIR)
        self._sync_client = chromadb.PersistentClient(
            path=rag_config.CHROMA_PERSIST_DIR,
            settings=_CHROMA_SETTINGS,
        )
        self._collection: Optional[chromadb.Collection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """✅ 异步初始化入口，所有同步 IO 通过 run_in_executor 桥接。"""
        if self._initialized:
            return

        loop = asyncio.get_running_loop()
        collection_name = rag_config.CHROMA_COLLECTION_NAME
        expected_space = rag_config.CHROMA_VECTOR_SPACE

        # 1. 列出集合（同步 → 异步）
        existing = await loop.run_in_executor(None, self._sync_client.list_collections)
        existing_names = [c.name for c in existing] if existing else []

        if collection_name in existing_names:
            self._collection = await loop.run_in_executor(
                None, partial(self._sync_client.get_collection, name=collection_name)
            )
            current_metadata = self._collection.metadata or {}
            current_space = current_metadata.get("hnsw:space")

            if current_space != expected_space:
                raise ValueError(
                    f"Collection '{collection_name}' space mismatch: "
                    f"db={current_space}, config={expected_space}. "
                    f"Please delete and recreate the collection."
                )
        else:
            self._collection = await loop.run_in_executor(
                None,
                partial(
                    self._sync_client.create_collection,
                    name=collection_name,
                    metadata={"hnsw:space": expected_space},
                ),
            )

        count = await loop.run_in_executor(None, self._collection.count)
        logger.info(
            "✅ ChromaDB 1.x ready | collection=%s | space=%s | count=%d",
            collection_name, expected_space, count,
        )
        self._initialized = True

    async def _get_fresh_collection(self) -> chromadb.Collection:
        """异步刷新 Collection 引用，解决跨进程 HNSW 段缓存失效问题。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._sync_client.get_or_create_collection,
                name=rag_config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": rag_config.CHROMA_VECTOR_SPACE},
            ),
        )

    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> int:
        if not texts:
            return 0

        loop = asyncio.get_running_loop()

        # ✅ 修复原 TODO：同步 embedding 走线程池，避免阻塞事件循环
        embeddings = await loop.run_in_executor(None, embedding_service.embed_texts, texts)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        # ✅ 修复：ChromaDB 1.x 不允许空 dict，用占位符替代
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in texts]

        total_added = 0
        for start in range(0, len(texts), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE
            try:
                await loop.run_in_executor(
                    None,
                    partial(
                        self._collection.upsert,
                        documents=texts[start:end],
                        embeddings=embeddings[start:end],
                        metadatas=metadatas[start:end],
                        ids=ids[start:end],
                    ),
                )
                total_added += min(_UPSERT_BATCH_SIZE, len(texts) - start)
            except Exception as e:
                logger.exception("❌ Upsert batch [%d:%d] failed: %s", start, end, e)
                raise

        count = await loop.run_in_executor(None, self._collection.count)
        logger.info("📥 Upserted %d chunks → total=%d", total_added, count)
        return total_added

    async def get_all_documents(self, limit: int = _GET_MAX_LIMIT) -> List[RawDocument]:
        """
        获取集合中全部文档，供 BM25 预热等外部消费者使用。
        - 自动完成幂等初始化，调用方无需手动先调 initialize()
        - 分批拉取避免单次内存峰值 & ChromaDB 默认返回上限
        - 统一输出 content 字段，与 BM25Engine.index() 契约对齐
        """
        if not self._initialized:
            await self.initialize()

        collection = await self._get_fresh_collection()
        loop = asyncio.get_running_loop()

        all_docs: List[RawDocument] = []
        offset = 0

        while len(all_docs) < limit:
            result = await loop.run_in_executor(
                None,
                partial(
                    collection.get,
                    include=["documents", "metadatas"],
                    limit=_GET_BATCH_SIZE,
                    offset=offset,
                ),
            )

            docs = result.get("documents") or []
            if not docs:
                break

            ids = result.get("ids") or [""] * len(docs)
            metas = result.get("metadatas") or [{}] * len(docs)
            for doc, did, meta in zip(docs, ids, metas):
                all_docs.append({"content": doc, "id": did, "metadata": meta or {}})

            offset += len(docs)
            if len(docs) < _GET_BATCH_SIZE:
                break

        if len(all_docs) >= limit:
            logger.warning(
                "⚠️ get_all_documents reached limit=%d, BM25 index may be incomplete", limit
            )

        logger.info("📚 Loaded %d documents from ChromaDB for BM25 warm-up", len(all_docs))
        return all_docs

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        k = top_k or rag_config.TOP_K
        try:
            loop = asyncio.get_running_loop()

            # ✅ 修复原 TODO：同步 embed_query 走线程池
            query_embedding = await loop.run_in_executor(None, embedding_service.embed_query, query)

            collection = await self._get_fresh_collection()
            # ✅ 防越界：n_results 不能超过集合实际数量，否则 Chroma 抛 ValueError
            col_count = await loop.run_in_executor(None, collection.count)
            k = max(1, min(k, col_count))
            results = await loop.run_in_executor(
                None,
                partial(
                    collection.query,
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                ),
            )

            hits: List[Dict] = []
            result_ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for i in range(len(result_ids)):
                hits.append({
                    "id": result_ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                })

            logger.info("🔍 Query='%s...' → %d hits (top_k=%d)", query[:30], len(hits), k)
            return hits
        except Exception as e:
            logger.error("❌ VectorStore.search failed: %s", e, exc_info=True)
            return []

    async def get_count(self) -> int:
        """✅ 替代原 async property，符合 Python 异步规范。"""
        collection = await self._get_fresh_collection()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, collection.count)


async def get_vector_store() -> VectorStore:
    """✅ 异步 DCL 单例模式。"""
    global _vector_store_instance
    if _vector_store_instance is None:
        async with _async_lock:
            if _vector_store_instance is None:
                store = VectorStore()
                await store.initialize()
                _vector_store_instance = store
    return _vector_store_instance


class _LazyVectorStore:
    """延迟初始化代理 — 自动适配 async/sync 属性访问。"""

    def __getattr__(self, name: str):
        async def _proxy(*args, **kwargs):
            store = await get_vector_store()
            attr = getattr(store, name)
            if callable(attr):
                # ✅ 兼容 async 方法和普通同步方法
                if asyncio.iscoroutinefunction(attr):
                    return await attr(*args, **kwargs)
                return attr(*args, **kwargs)
            return attr
        return _proxy


vector_store = _LazyVectorStore()