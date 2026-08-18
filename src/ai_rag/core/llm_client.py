# src/ai_rag/core/llm_client.py
from functools import lru_cache
from openai import AsyncOpenAI
from ai_rag.core.config import rag_config


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        # Ollama 等本地服务不校验 key，空 key 时传 "ollama" 占位即可（与 router.py 一致）
        api_key=rag_config.OPENAI_API_KEY or "ollama",
        base_url=rag_config.OPENAI_BASE_URL,
        timeout=rag_config.LLM_TIMEOUT,
    )