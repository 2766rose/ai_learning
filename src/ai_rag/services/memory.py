import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict

import redis.asyncio as aioredis
from ai_rag.core.config import rag_config

logger = logging.getLogger(__name__)


class ShortTermMemory(ABC):
    @abstractmethod
    async def load(self, session_id: str) -> List[Dict]: ...

    @abstractmethod
    async def save(self, session_id: str, messages: List[Dict]) -> None: ...


class RedisShortTermMemory(ShortTermMemory):
    def __init__(self):
        self._pool: aioredis.Redis | None = None
        self._lock = asyncio.Lock()

    async def _get_pool(self) -> aioredis.Redis:
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = aioredis.from_url(
                        rag_config.REDIS_URL, decode_responses=True
                    )
        return self._pool

    async def load(self, session_id: str) -> List[Dict]:
        pool = await self._get_pool()
        data = await pool.get(f"session:{session_id}")
        return json.loads(data) if data else []

    async def save(self, session_id: str, messages: List[Dict]) -> None:
        pool = await self._get_pool()
        await pool.setex(
            f"session:{session_id}", rag_config.SESSION_TTL, json.dumps(messages)
        )


class InMemoryShortTermMemory(ShortTermMemory):
    def __init__(self):
        self._store: Dict[str, List[Dict]] = {}

    async def load(self, session_id: str) -> List[Dict]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, messages: List[Dict]) -> None:
        self._store[session_id] = messages


_memory_instance: ShortTermMemory | None = None
_init_lock = asyncio.Lock()


async def get_short_term_memory() -> ShortTermMemory:
    global _memory_instance
    if _memory_instance is None:
        async with _init_lock:
            if _memory_instance is None:
                backend = rag_config.MEMORY_BACKEND.lower()
                if backend == "redis":
                    _memory_instance = RedisShortTermMemory()
                    logger.info("Memory backend: Redis")
                else:
                    _memory_instance = InMemoryShortTermMemory()
                    logger.info("Memory backend: InMemory")
    return _memory_instance