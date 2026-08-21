# hybrid_retriever.py
"""
Enterprise-grade Hybrid Retriever with RRF fusion and strict typing.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

# 【关键】统一契约导入，消除裸 Dict
from ai_rag.retrieval.types import RetrievedDoc

logger = logging.getLogger(__name__)

class HybridRetriever:
    """混合检索器：负责多路召回、格式标准化与 RRF 融合"""

    def __init__(self, bm25_engine, vector_store, config):
        self.bm25 = bm25_engine
        self.vector_store = vector_store
        self.config = config
        # RRF 常数 k，防止排名靠前的文档分数差异过大
        self._rrf_k = getattr(config, "rrf_k", 60)

    async def retrieve(self, query: str) -> List[RetrievedDoc]:
        """
        主检索入口：并发召回 → 标准化 → RRF 融合
        返回严格符合 RetrievedDoc 契约的结果列表
        """
        # 1. 并发获取两路原始结果（保持异步性能）
        bm25_raw = await self._safe_bm25_search(query)
        vector_raw = await self._safe_vector_search(query)

        # 2. 【关键】标准化为统一契约，隔离上游数据结构变化
        bm25_docs = self._normalize_bm25_results(bm25_raw)
        vector_docs = self._normalize_vector_results(vector_raw)

        # 3. RRF 融合并截断
        fused = self._rrf_fusion(
            [bm25_docs, vector_docs],
            top_k=getattr(self.config, "final_top_k", 10)
        )

        logger.info(
            f"Hybrid retrieve completed: query='{query[:30]}...', "
            f"bm25={len(bm25_docs)}, vector={len(vector_docs)}, fused={len(fused)}"
        )
        return fused

    # ------------------------------------------------------------------ #
    #                       安全召回层（防御性编程）                        #
    # ------------------------------------------------------------------ #
    async def _safe_bm25_search(self, query: str) -> list:
        try:
            return self.bm25.search(query, top_k=self.config.bm25_top_k)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}", exc_info=True)
            return []

    async def _safe_vector_search(self, query: str) -> list:
        try:
            return await self.vector_store.search(
                query=query, top_k=self.config.vector_top_k
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    #              格式标准化层（将异构数据转为 RetrievedDoc）               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_bm25_results(
        raw: List[Tuple[Dict[str, Any], float]]
    ) -> List[RetrievedDoc]:
        """BM25 原始格式: [(doc_dict, score), ...]"""
        normalized: List[RetrievedDoc] = []
        for doc, score in raw:
            normalized.append(RetrievedDoc(
                id=str(doc.get("id", "")),
                content=str(doc.get("content", doc.get("text", ""))),  # 兼容旧字段
                score=float(score),
                metadata=doc.get("metadata", {})
            ))
        return normalized

    @staticmethod
    def _normalize_vector_results(raw: List[Dict[str, Any]]) -> List[RetrievedDoc]:
        """Vector Store 原始格式: [{"id":..., "document":..., "distance":...}, ...]"""
        normalized: List[RetrievedDoc] = []
        for item in raw:
            distance = float(item.get("distance", 1.0))
            similarity = max(0.0, 1.0 - distance / 2.0)
            normalized.append(RetrievedDoc(
                id=str(item.get("id", "")),
                # ✅ 优先读 document，兼容 content/text
                content=str(item.get("document", item.get("content", item.get("text", "")))),
                score=similarity,
                metadata=item.get("metadata", {})
            ))
        return normalized

    # ------------------------------------------------------------------ #
    #                     RRF 融合算法（企业级实现）                        #
    # ------------------------------------------------------------------ #
    def _rrf_fusion(
        self,
        ranked_lists: List[List[RetrievedDoc]],
        top_k: int = 10
    ) -> List[RetrievedDoc]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009)
        score(d) = Σ 1/(k + rank_i(d))
        """
        if not ranked_lists:
            return []

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDoc] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc["id"]
                if not doc_id:
                    continue
                # 【安全】防止除零，k 值可配置
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
                # 保留首次出现的完整文档信息
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # 按 RRF 分数降序排列，截取 top_k
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # 【关键】用 RRF 分数覆盖原始分数，使下游消费者获得可解释的融合分
        result: List[RetrievedDoc] = []
        for doc_id in sorted_ids:
            merged_doc = dict(doc_map[doc_id])  # shallow copy
            merged_doc["score"] = round(rrf_scores[doc_id], 6)
            result.append(RetrievedDoc(**merged_doc))

        return result