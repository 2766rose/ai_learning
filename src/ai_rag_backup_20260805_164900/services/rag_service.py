# src/ai_rag/services/rag_service.py
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from src.ai_rag.core.vector_store import vector_store

logger = logging.getLogger(__name__)

# ✅ 可调参数：根据实际召回质量调整，建议范围 0.3~0.6
MIN_SIMILARITY = 0.5
MAX_CHUNK_CHARS = 500
NO_RESULT_MSG = "No relevant information found in knowledge base."


async def knowledge_search_handler(
    query: str,
    session_id: Optional[str] = None,
) -> str:
    """
    企业知识库检索工具（带诊断日志 + 安全兜底）
    Args:
        query: 用户问题或检索关键词
        session_id: 仅用于日志追踪，不参与检索过滤
    """
    try:
        # ✅ 企业规范：知识库是全局共享资源，where=None
        hits: List[Dict[str, Any]] = vector_store.search(query=query, where=None)

        if not hits:
            logger.info("🔍 No raw hits returned | query='%s' | session=%s", query[:80], session_id)
            return NO_RESULT_MSG

        # 🔍 诊断日志：打印每条原始结果的相似度，用于排查阈值问题
        for i, h in enumerate(hits):
            sim = 1.0 - h.get("distance", 1.0)
            passed = sim >= MIN_SIMILARITY
            logger.info(
                "[DEBUG] Raw hit %d | similarity=%.4f | passed=%s | preview='%s'",
                i, sim, passed, h.get("text", "")[:60].replace("\n", " ")
            )

        # ✅ 阈值过滤
        filtered_hits = [
            h for h in hits
            if (1.0 - h.get("distance", 1.0)) >= MIN_SIMILARITY
        ]

        # 🛡️ 安全兜底：若全部被过滤，保留相似度最高的 Top-1 并标记低置信度
        low_confidence_fallback = False
        if not filtered_hits:
            logger.warning(
                "⚠️ All %d hits below threshold (%.2f) | falling back to top-1 | query='%s'",
                len(hits), MIN_SIMILARITY, query[:80]
            )
            # 按 distance 升序取第一条（即相似度最高）
            best_hit = min(hits, key=lambda x: x.get("distance", 1.0))
            filtered_hits = [best_hit]
            low_confidence_fallback = True

        # ✅ 格式化输出
        formatted_parts = []
        for i, h in enumerate(filtered_hits, 1):
            sim = 1.0 - h.get("distance", 1.0)
            text = h.get("text", "")[:MAX_CHUNK_CHARS]
            prefix = "[LOW_CONFIDENCE] " if low_confidence_fallback else ""
            formatted_parts.append(f"{prefix}[{i}] (similarity:{sim:.3f}) {text}")

        formatted = "\n\n".join(formatted_parts)

        logger.info(
            "✅ Returned %d chunks%s | query='%s' | session=%s | total_chars=%d",
            len(filtered_hits),
            " (fallback)" if low_confidence_fallback else "",
            query[:80],
            session_id,
            len(formatted),
        )
        return formatted

    except Exception as e:
        logger.exception("❌ knowledge_search failed | query='%s'", query[:80])
        # ✅ 企业规范：异常也返回结构化文本，不抛裸异常给 LLM
        return f"Knowledge search encountered an error: {type(e).__name__}. Please retry."