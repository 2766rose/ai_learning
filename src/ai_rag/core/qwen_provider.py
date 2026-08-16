#src\ai_rag\agent\qwen_provider.py
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Optional

from openai import APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ✅ 关键调整：复用你现有的客户端工厂，而非新建客户端
from ai_rag.core.llm_client import get_async_openai_client
from ai_rag.core.config import rag_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """RAG 生成结果的结构化契约（即使无 BaseLLM，此结构也不可省略）"""
    answer: str
    sources: List[str]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model_version: str


class QwenProvider:
    """Qwen-Plus RAG Provider - 适配现有 llm_client 架构"""

    def __init__(self):
        # ✅ 使用全局单例客户端，共享连接池
        self.client = get_async_openai_client()
        self.model = rag_config.OPENAI_MODEL
        self.temperature = rag_config.LLM_TEMPERATURE
        self.max_tokens = rag_config.LLM_MAX_TOKENS

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        reraise=True,
    )
    async def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        start_time = time.perf_counter()

        # 构建 RAG Prompt
        ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context)) if context else "无相关上下文"
        default_sys = (
            "你是一个严谨的知识助手。请仅根据提供的【参考上下文】回答用户问题。\n"
            "如果上下文中没有答案，请明确告知'未找到相关信息'，不要编造。\n"
            "回答时请在句末用 [序号] 标注引用来源。"
        )
        messages = [
            {"role": "system", "content": system_prompt or default_sys},
            {"role": "user", "content": f"【参考上下文】\n{ctx_text}\n\n【用户问题】\n{query}"},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
        except APIError as e:
            logger.error(f"Qwen API Error: {e}")
            raise

        # 解析响应
        latency_ms = (time.perf_counter() - start_time) * 1000
        choice = response.choices[0]
        usage = response.usage
        answer = choice.message.content or ""
        sources = list(set(re.findall(r"\[(\d+)\]", answer)))

        result = LLMResponse(
            answer=answer,
            sources=sources,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(latency_ms, 2),
            model_version=response.model or self.model,
        )

        logger.info(
            f"[Qwen] model={result.model_version} | "
            f"latency={result.latency_ms}ms | "
            f"tokens={result.prompt_tokens}+{result.completion_tokens}"
        )
        return result