from functools import lru_cache
from openai import AsyncOpenAI
from src.ai_rag.core.config import rag_config


@lru_cache(maxsize=1)
def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=rag_config.OPENAI_API_KEY,
        base_url=rag_config.OPENAI_BASE_URL,
        timeout=rag_config.LLM_TIMEOUT,
    )