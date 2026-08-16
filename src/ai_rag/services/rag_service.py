# src/ai_rag/services/rag_service.py
from __future__ import annotations

import logging
import os
from typing import Optional, List, Dict, Any

from ai_rag.core.vector_store import vector_store

logger = logging.getLogger(__name__)


def _similarity(h: Dict[str, Any]) -> float:
    """统一相似度：向量模式用 distance，混合/精排模式用 similarity/score"""
    if h.get("distance") is not None:
        return 1.0 - float(h["distance"])
    if h.get("similarity") is not None:
        return float(h["similarity"])
    return float(h.get("score", 0.0))

# 可调参数：根据实际召回质量调整，建议范围 0.3~0.6
# 2026-08-14：text2vec 对长混合块的相似度普遍在 0.35~0.55，0.5 会误杀相关块导致模型编造，
# 下调到 0.35 并让 RRF 排序兜底（排后面的低分结果自然被 top_k 截断）。
MIN_SIMILARITY = 0.35
# 2026-08-14：控制工具返回体量（3 块 × 300 字），避免检索结果过大导致上下文裁剪塌缩/循环
MAX_CHUNK_CHARS = 300
MAX_FORMATTED_CHUNKS = 3
NO_RESULT_MSG = "No relevant information found in knowledge base."


async def knowledge_search_handler(
    query: str,
    session_id: Optional[str] = None,
) -> str:
    """
    企业知识库检索工具（带诊断日志 + 安全兜底）

    Args:
        query: 用户问题或检索关键词
        session_id: 仅用于日志追踪，不参与检索过程
    """
    try:
        # 企业规范：知识库是全局共享资源，where=None
        # 检索模式：RAG_RETRIEVAL=hybrid(默认，BM25+向量+RRF) | vector(纯向量)
        mode = os.environ.get("RAG_RETRIEVAL", "hybrid").strip().lower()
        if mode == "vector":
            hits: List[Dict[str, Any]] = await vector_store.search(query=query, where=None)
        else:
            from ai_rag.retrieval.retriever import hybrid_retriever_service
            # 2026-08-14：BGE-Reranker 已本地化，默认开启精排（RAG_RERANK=off 可关闭）
            use_rerank = os.environ.get("RAG_RERANK", "on").strip().lower() == "on"
            hits = await hybrid_retriever_service.search(
                query=query, top_k=5, where=None, rerank=use_rerank
            )

        if not hits:
            logger.info("🔍 No raw hits returned | query='%s' | session=%s", query[:80], session_id)
            return NO_RESULT_MSG

        # 🔍 诊断日志：打印每条原始结果的相似度，用于排查阈值问题
        for i, h in enumerate(hits):
            sim = _similarity(h)
            passed = sim >= MIN_SIMILARITY
            logger.debug(
                "Raw hit %d | similarity=%.4f | passed=%s | preview='%s'",
                i, sim, passed, h.get("document", "")[:30].replace("\n", " ")
            )

        # 阈值过滤
        filtered_hits = [
            h for h in hits
            if _similarity(h) >= MIN_SIMILARITY
        ]

         # 🛡️ 安全兜底：若全部低于阈值，返回空字符串触发 System Prompt 兜底回复
        if not filtered_hits:
            logger.info(
                "ℹ️ All %d hits below threshold (%.2f), returning empty context | query='%s'",
                len(hits), MIN_SIMILARITY, query[:80]
            )
            return ""

        # 格式化输出（只保留前 MAX_FORMATTED_CHUNKS 块，控制体量）
        formatted_parts = []
        for i, h in enumerate(filtered_hits[:MAX_FORMATTED_CHUNKS], 1):
            sim = _similarity(h)
            text = h.get("document", "")[:MAX_CHUNK_CHARS]
            formatted_parts.append(f"[{i}] (similarity:{sim:.3f}) {text}")

        formatted = "\n\n".join(formatted_parts)

        logger.info(
            "✅ Returned %d chunks | query='%s' | session=%s | total_chars=%d",
            len(filtered_hits), query[:80], session_id, len(formatted),
        )
        return formatted

    except Exception as e:
        logger.exception("❌ knowledge_search failed | query='%s'", query[:80])
        # 企业规范：异常也返回结构化文本，不抛裸异常给 LLM
        return f"Knowledge search encountered an error: {type(e).__name__}. Please retry."