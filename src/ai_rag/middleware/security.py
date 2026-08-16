# security.py
import os
import logging
from typing import Set

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from ai_rag.core.config import rag_config

logger = logging.getLogger(__name__)

API_KEY = rag_config.API_KEY
if not API_KEY:
    logger.warning("RAG_API_KEY not set: API is unauthenticated - do not expose to public network")


class SensitiveWordFilter:
    def __init__(self, words: Set[str]):
        self._trie: dict = {}
        for word in words:
            node = self._trie
            for char in word:
                node = node.setdefault(char, {})
            node["__end__"] = True

    def contains(self, text: str) -> str | None:
        for i in range(len(text)):
            node = self._trie
            j = i
            while j < len(text) and text[j] in node:
                node = node[text[j]]
                if "__end__" in node:
                    return text[i : j + 1]
                j += 1
        return None


_SENSITIVE_WORDS: Set[str] = set(
    os.getenv("RAG_SENSITIVE_WORDS", "暴力,赌博,毒品").split(",")
)
_word_filter = SensitiveWordFilter(_SENSITIVE_WORDS)


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # API Key 鉴权
        if API_KEY and request.url.path.startswith("/api/"):
            key = request.headers.get("X-API-Key", "")
            if key != API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})

        # JSON 请求体敏感词过滤
        content_type = request.headers.get("content-type", "")
        if request.method == "POST" and "application/json" in content_type:
            # ✅ await request.body() 会读取流并自动缓存到 request._body
            # BaseHTTPMiddleware 在 dispatch 返回后会将缓存的 body
            # 重新包装为 receive callable 传递给下游路由
            # 因此无需手动重建 Request 对象
            body = await request.body()
            text = body.decode("utf-8", errors="ignore")

            hit = _word_filter.contains(text)
            if hit:
                logger.warning(
                    "sensitive_word_blocked | path=%s | word=%s",
                    request.url.path,
                    hit,
                )
                return JSONResponse(status_code=400, content={"detail": f"Content contains sensitive word: {hit}"})

        return await call_next(request)