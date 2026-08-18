# src/ai_rag/agent/memory.py
import json
import time
import logging
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from openai import AsyncOpenAI

from ai_rag.core.config import rag_config
from ai_rag.core.embeddings import embedding_service

logger = logging.getLogger(__name__)

# 复用全局 LLM 客户端（懒加载：首次使用时才创建，避免缺少 API Key 时 import 崩溃）
_llm_client = None


def _get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        # Ollama 等本地服务不校验 key，空 key 时传 "ollama" 占位即可
        _llm_client = AsyncOpenAI(
            api_key=rag_config.OPENAI_API_KEY or "ollama",
            base_url=rag_config.OPENAI_BASE_URL,
            timeout=rag_config.LLM_TIMEOUT,
        )
    return _llm_client

# ✅ 优化1：ChromaDB Client 模块级单例，避免重复初始化
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_memory_collection() -> chromadb.Collection:
    """获取或创建长期记忆专用的 ChromaDB Collection（单例）"""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=rag_config.MEMORY_CHROMA_PERSIST_DIR
        )
        logger.info("🔗 [Memory] ChromaDB client initialized at %s", rag_config.MEMORY_CHROMA_PERSIST_DIR)

    # ✅ 优化2：使用配置中的 collection 名称，而非硬编码
    return _chroma_client.get_or_create_collection(
        name=rag_config.MEMORY_COLLECTION_NAME,
        metadata={"hnsw:space": rag_config.CHROMA_VECTOR_SPACE}
    )


async def extract_memories(conversation: List[Dict[str, Any]]) -> List[str]:
    """
    使用 LLM 从对话中提取值得长期记住的事实/偏好
    返回事实字符串列表，若无则返回空列表
    """
    if not conversation:
        return []

    extraction_prompt = """你是一个记忆提取器。请从以下对话中提取用户明确表达的、值得长期记住的个人事实、偏好或背景信息。

【提取规则】
1. 只提取确定性事实，不要推测
2. 每条事实用一句话简洁表述
3. 忽略与用户个人无关的通用知识
4. 如果没有值得记住的内容，返回空数组 []

【输出格式】严格返回 JSON 数组，如 ["事实1", "事实2"]

【对话内容】
{conversation}"""

    try:
        response = await _get_llm_client().chat.completions.create(
            model=rag_config.OPENAI_MODEL,
            temperature=0.0,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": extraction_prompt.format(
                    conversation=json.dumps(conversation, ensure_ascii=False)
                )
            }],
        )

        content = (response.choices[0].message.content or "").strip()
        # 兼容 LLM 可能包裹 ```json ... ``` 的情况
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        facts = json.loads(content)
        if not isinstance(facts, list):
            facts = []

        # 过滤非字符串项
        facts = [f for f in facts if isinstance(f, str) and f.strip()]

        logger.info("🧠 [Memory] Extracted %d facts from conversation.", len(facts))
        return facts

    except Exception as e:
        logger.error("❌ [Memory] Extraction failed: %s", e)
        return []


async def save_memories(user_id: str, facts: List[str]) -> None:
    """将提取到的事实保存到 ChromaDB 长期记忆库"""
    if not facts:
        return

    collection = _get_memory_collection()

    ids = [str(uuid.uuid4()) for _ in facts]
    metadatas = [{"user_id": user_id, "type": "fact"} for _ in facts]

    try:
        collection.add(
            ids=ids,
            documents=facts,
            metadatas=metadatas,
        )
        logger.info("💾 [Memory] Saved %d facts for user %s", len(facts), user_id)
    except Exception as e:
        logger.error("❌ [Memory] Save failed: %s", e)


async def retrieve_memories(
    user_id: str,
    query: str,
    top_k: int = 3,
    distance_threshold: float = 0.5  # ✅ 优化3：cosine distance 阈值，越大越不相关
) -> List[str]:
    """根据当前 query 检索用户的长期记忆，自动过滤低相关性结果"""
    collection = _get_memory_collection()

    try:
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[embedding_service.embed_query(query)],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "distances"]  # ✅ 同时返回距离用于过滤
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        # ✅ 按距离阈值过滤：cosine space 下 distance ∈ [0, 2]，越小越相关
        memories = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist <= distance_threshold:
                memories.append(doc)
            else:
                logger.debug("🚫 [Memory] Filtered out (distance=%.3f): %s", dist, doc[:50])

        logger.info("🔍 [Memory] Retrieved %d/%d memories for user %s", len(memories), top_k, user_id)
        return memories

    except Exception as e:
        logger.error("❌ [Memory] Retrieve failed: %s", e)
        return []
async def save_memories_with_dedup(user_id: str, facts: List[str]) -> None:
    """
    企业级记忆写入：先检索相似记忆，去重/更新后再写入。
    避免重复保存相同事实，同时支持事实修正（如"我在北京工作"→"我在上海工作"）。
    """
    if not facts:
        return

    collection = _get_memory_collection()

    for fact in facts:
        try:
            # 检索最相似的已有记忆
            _fact_emb = embedding_service.embed_query(fact)
            existing = collection.query(
                query_embeddings=[_fact_emb],
                n_results=1,
                where={"user_id": user_id},
                include=["distances", "documents"]
            )

            # 相似度阈值：cosine distance < 0.1 视为同一条记忆
            if (existing
                    and existing["distances"]
                    and existing["distances"][0]
                    and existing["distances"][0][0] < 0.1):

                old_id = existing["ids"][0][0]
                old_doc = existing["documents"][0][0]

                # 完全相同 → 跳过
                if old_doc == fact:
                    logger.debug("⏭️ [Memory] 记忆已存在，跳过: %s", fact)
                    continue

                # 高度相似但内容不同 → 更新（事实修正）
                collection.update(
                    ids=[old_id],
                    documents=[fact],
                    embeddings=[_fact_emb],
                    metadatas=[{"user_id": user_id, "type": "fact", "updated_at": time.time()}]
                )
                logger.info("🔄 [Memory] 更新记忆: '%s' → '%s'", old_doc[:50], fact[:50])
            else:
                # 无相似记忆 → 新增
                collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[fact],
                    embeddings=[_fact_emb],
                    metadatas=[{"user_id": user_id, "type": "fact", "created_at": time.time()}]
                )
                logger.info("💾 [Memory] 新增记忆: %s", fact)

        except Exception as e:
            logger.error("❌ [Memory] 单条记忆写入失败 | fact=%s | error=%s", fact[:50], e)

    logger.info("✅ [Memory] save_memories_with_dedup 完成 | user=%s | input_count=%d", user_id, len(facts))   