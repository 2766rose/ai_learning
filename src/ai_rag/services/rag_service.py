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

# Retrieval quality knobs (tuned 2026-08-17):
# - MIN_SIMILARITY: vector-sim floor. text2vec long mixed chunks score ~0.30-0.55;
#   0.35 dropped the 'work hours' chunk (0.333) -> lowered to 0.30.
# - RERANK_MIN_SCORE: rerank logit below this => no relevant info (anti-hallucination gate).
MIN_SIMILARITY = 0.30
RERANK_MIN_SCORE = 0.20
VEC_GATE_SCORE = 0.55  # 重排分低时的向量相似度附加门槛（双信号才拦截）
# 2026-08-14：控制工具返回体量（3 块 × 300 字），避免检索结果过大导致上下文裁剪塌缩/循环
MAX_CHUNK_CHARS = 450
MAX_FORMATTED_CHUNKS = 3
NO_RESULT_MSG = "No relevant information found in knowledge base."


async def knowledge_search_handler(
    query: str,
    session_id: Optional[str] = None,
    domain: str = "company",
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
        # 相关性门槛：重排分过低视为无相关信息（防幻觉抦底）
        if use_rerank and hits:
            _top_score = float(hits[0].get("score", 0.0))
            _top_sim = float(hits[0].get("similarity", 0.0))
            # 双信号门槛：仅当重排分和向量相似度都低时才视为无相关信息（避免误杀像保密这类重排分低但向量相似度高的真问题）
            if _top_score < RERANK_MIN_SCORE and _top_sim < VEC_GATE_SCORE:
                logger.info("No relevant info (rerank=%.4f vec=%.3f) | query=%s", _top_score, _top_sim, query[:60])
                return NO_RESULT_MSG

        # 相关性过滤：rerank 开启时用重排分（Cross-Encoder 更准），否则用向量相似度
        if use_rerank:
            filtered_hits = [h for h in hits if float(h.get("score", 0.0)) >= 0.0]
        else:
            filtered_hits = [
                h for h in hits
                if _similarity(h) >= MIN_SIMILARITY
            ]
        # 知识域过滤：按元数据 domain 分离企业与个人知识（默认 company）
        if domain and domain != "all":
            filtered_hits = [
                h for h in filtered_hits
                if (h.get("metadata") or {}).get("domain", "company") == domain
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