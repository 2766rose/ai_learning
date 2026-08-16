# src/ai_rag/core/semantic_cache.py
"""语义缓存（按向量相似度拦截重复/相似问题，减少重复推理）"""
import logging
import threading
import time

logger = logging.getLogger(__name__)


def _cosine(a, b):
    import numpy as np
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    d = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / d) if d else 0.0


class SemanticCache:
    """进程内语义缓存：embedding 相似度超过阈值即命中"""

    def __init__(self, threshold: float = 0.92, max_entries: int = 500, ttl: int = 3600):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl
        self._items = []  # [(embedding, answer, ts)]
        self._lock = threading.Lock()

    def get(self, query_embedding):
        """命中返回缓存的回答，否则 None"""
        with self._lock:
            now = time.time()
            self._items = [(e, a, t) for e, a, t in self._items if now - t < self.ttl]
            best, best_sim = None, self.threshold
            for emb, answer, _ts in self._items:
                sim = _cosine(emb, query_embedding)
                if sim > best_sim:
                    best_sim, best = sim, answer
            if best is not None:
                logger.info("语义缓存命中 | sim=%.4f", best_sim)
            return best

    def put(self, query_embedding, answer) -> None:
        if not answer:
            return
        with self._lock:
            self._items.append((list(query_embedding), answer, time.time()))
            if len(self._items) > self.max_entries:
                self._items = self._items[-self.max_entries:]
            logger.info("语义缓存写入 | entries=%d", len(self._items))

    def size(self) -> int:
        with self._lock:
            return len(self._items)


semantic_cache = SemanticCache()
