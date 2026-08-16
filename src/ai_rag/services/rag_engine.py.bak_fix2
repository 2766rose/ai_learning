# src/ai_rag/services/rag_engine.py
"""企业级 RAG 引擎 - 纯检索增强生成核心，零框架依赖"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI

from ai_rag.core.llm_client import get_async_openai_client
from ai_rag.core.config import rag_config
from ai_rag.core.vector_store import get_vector_store  # ✅ 改用统一入口

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """不可变检索结果"""
    documents: List[str]
    distances: List[float]
    formatted_context: str


class RAGEngine:
    """RAG 核心引擎：封装 检索 → 上下文构建 → LLM生成 全流程"""

    SYSTEM_PROMPT_TEMPLATE = (
        "你是企业知识库助手。请严格基于以下检索内容回答用户问题。\n"
        "如果检索内容与问题无关，请如实说明'未找到相关信息'，不要编造。\n\n"
        "=== 检索内容 ===\n{context}\n=== 结束 ==="
    )

    def __init__(self):  # ✅ 不再接收 collection 和 embed_model
        self._vs = get_vector_store()
        self._client: AsyncOpenAI = get_async_openai_client()

    async def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """向量检索 + 结果预格式化（委托给 VectorStore.search）"""
        top_k = top_k or rag_config.TOP_K
        try:
            hits = self._vs.search(query, top_k=top_k)  # ✅ 自动刷新引用

            docs = [h["content"] for h in hits]
            dists = [h["distance"] for h in hits]

            formatted = "\n\n".join(
                f"[{i}] (相似度:{1 - d:.3f}) {doc[:500]}"
                for i, (doc, d) in enumerate(zip(docs, dists), 1)
            ) if docs else ""

            return RetrievalResult(documents=docs, distances=dists, formatted_context=formatted)
        except Exception as e:
            logger.exception("RAGEngine.retrieve failed for query=%s", query[:80])
            return RetrievalResult(documents=[], distances=[], formatted_context=f"检索异常: {e}")

    async def generate(self, query: str, context: str) -> str:
        """基于上下文的 LLM 生成（无 Tool Call，单次调用）"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_TEMPLATE.format(context=context)},
            {"role": "user", "content": query},
        ]
        resp = await self._client.chat.completions.create(
            model=rag_config.OPENAI_MODEL, messages=messages,
            temperature=rag_config.LLM_TEMPERATURE, max_tokens=rag_config.LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""

    async def query(self, query: str) -> dict:
        """一站式 RAG 问答（retrieve + generate）"""
        retrieval = await self.retrieve(query)
        answer = await self.generate(query, retrieval.formatted_context)
        return {
            "ai_answer": answer,
            "retrieved_knowledge": retrieval.formatted_context,
        }