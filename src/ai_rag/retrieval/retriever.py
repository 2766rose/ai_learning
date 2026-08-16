# src/ai_rag/retrieval/retriever.py
"""混合检索服务：BM25 + 向量检索 + RRF 融合
- 懒加载：首次检索时从向量库全量文档预热 BM25 索引
- 可选接入 BGE-Reranker 精排（见 reranker.py）
- 环境变量：RAG_RETRIEVAL=hybrid|vector（默认 hybrid）
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from ai_rag.retrieval.bm25_engine import BM25Engine
from ai_rag.retrieval.config import RetrievalConfig
from ai_rag.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class HybridRetrieverService:
    """进程级单例：混合检索入口"""

    def __init__(self) -> None:
        self._config = RetrievalConfig()
        self._bm25 = BM25Engine()
        self._warmed = False
        self._lock = threading.Lock()

    async def warmup(self, store=None) -> None:
        """从向量库全量文档预热 BM25 索引（幂等 + 自动感知入库变化）"""
        if store is None:
            from ai_rag.core.vector_store import get_vector_store
            store = await get_vector_store()
        count = await store.get_count()
        if self._warmed and self._bm25.doc_count == count:
            return
        with self._lock:
            if self._warmed and self._bm25.doc_count == count:
                return
            docs = await store.get_all_documents()
            self._bm25.index(docs)
            self._warmed = True
            logger.info("BM25 预热完成 | docs=%d", len(docs))

    async def refresh(self) -> None:
        """强制重建 BM25（文档入库后调用）"""
        self._warmed = False
        self._bm25 = BM25Engine()
        await self.warmup()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """混合检索 → (可选)精排 → 返回统一契约 [{id, document, metadata, score}]"""
        from ai_rag.core.vector_store import get_vector_store
        store = await get_vector_store()
        await self.warmup(store)

        hybrid = HybridRetriever(self._bm25, store, self._config)
        docs = await hybrid.retrieve(query)  # final_top_k=20

        # 向量相似度映射：用于格式化展示与阈值过滤（RRF 融合分数不是相似度）
        sim_map: Dict[str, float] = {}
        try:
            vec_hits = await store.search(query, top_k=self._config.final_top_k, where=where)
            for h in vec_hits:
                sim_map[str(h.get("id", ""))] = 1.0 - float(h.get("distance", 1.0))
        except Exception as e:
            logger.warning("向量相似度映射失败: %s", e)

        results: List[Dict[str, Any]] = []
        for d in docs[: self._config.final_top_k]:
            did = str(d.get("id", ""))
            results.append({
                "id": did,
                "document": d.get("content", ""),
                "metadata": d.get("metadata", {}) or {},
                "score": float(d.get("score", 0.0)),
                "similarity": sim_map.get(did, float(d.get("score", 0.0))),
            })

        if rerank and results:
            try:
                from ai_rag.retrieval.reranker import reranker
                results = await reranker.rerank(query, results, top_k=top_k)
                logger.info("Rerank 完成 | query=%s | top=%d", query[:30], len(results))
            except Exception as e:
                logger.warning("Reranker 不可用，跳过精排: %s", e)

        return results[:top_k]


hybrid_retriever_service = HybridRetrieverService()


def retrieval_mode() -> str:
    return os.environ.get("RAG_RETRIEVAL", "hybrid").strip().lower()