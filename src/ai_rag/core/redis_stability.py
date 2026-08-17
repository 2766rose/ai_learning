# -*- coding: utf-8 -*-
"""Redis 分布式版：语义缓存 / 限流 / 熔断（带内存降级，Redis 不可用时自动回退）"""
import json
import threading
import time
import uuid

import redis

from ai_rag.core.config import rag_config

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = redis.Redis.from_url(rag_config.REDIS_URL, decode_responses=True)
    return _client


class RedisRateLimiter:
    def __init__(self, limit=20, window=60):
        self.limit = limit
        self.window = window
        self._local = {}
        self._lock = threading.Lock()

    def allow(self, key):
        try:
            r = _get_client()
            rkey = "rl:" + str(key)
            n = r.incr(rkey)
            if n == 1:
                r.expire(rkey, self.window)
            if n > self.limit:
                return False, max(1, r.ttl(rkey))
            return True, 0
        except Exception:
            now = time.time()
            with self._lock:
                ts = self._local.setdefault(key, [])
                ts = [t for t in ts if now - t < self.window]
                if len(ts) >= self.limit:
                    return False, max(1, int(self.window - (now - ts[0])) + 1)
                ts.append(now)
                self._local[key] = ts
                return True, 0


class RedisCircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown=30):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._local_failures = 0
        self._local_opened = None

    def allow(self):
        try:
            r = _get_client()
            opened = r.get("cb:opened_at")
            if opened is not None:
                if time.time() - float(opened) >= self.cooldown:
                    r.delete("cb:opened_at")
                    return True
                return False
            return True
        except Exception:
            if self._local_opened is not None:
                if time.time() - self._local_opened >= self.cooldown:
                    self._local_opened = None
                    return True
                return False
            return True

    def record_success(self):
        try:
            r = _get_client()
            r.delete("cb:failures")
            r.delete("cb:opened_at")
        except Exception:
            self._local_failures = 0
            self._local_opened = None

    def record_failure(self):
        try:
            r = _get_client()
            n = r.incr("cb:failures")
            if n == 1:
                r.expire("cb:failures", self.cooldown)
            if n >= self.failure_threshold:
                r.set("cb:opened_at", time.time(), ex=self.cooldown)
        except Exception:
            self._local_failures += 1
            if self._local_failures >= self.failure_threshold:
                self._local_opened = time.time()


class RedisSemanticCache:
    def __init__(self, threshold=0.92, max_entries=500, ttl=3600):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl
        self._local = []
        self._lock = threading.Lock()

    @staticmethod
    def _cosine(a, b):
        import numpy as np
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        d = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / d) if d else 0.0

    def get(self, query_embedding):
        try:
            r = _get_client()
            best, best_sim = None, self.threshold
            for k in r.scan_iter("sc:*", count=200):
                data = r.hgetall(k)
                if not data or "emb" not in data:
                    continue
                emb = json.loads(data["emb"])
                sim = self._cosine(emb, query_embedding)
                if sim > best_sim:
                    best_sim, best = sim, data.get("answer")
            return best
        except Exception:
            with self._lock:
                now = time.time()
                self._local = [(e, a, t) for e, a, t in self._local if now - t < self.ttl]
                best, best_sim = None, self.threshold
                for emb, answer, _ts in self._local:
                    sim = self._cosine(emb, query_embedding)
                    if sim > best_sim:
                        best_sim, best = sim, answer
                return best

    def put(self, query_embedding, answer):
        if not answer:
            return
        try:
            r = _get_client()
            k = "sc:" + uuid.uuid4().hex
            r.hset(k, mapping={"emb": json.dumps(list(query_embedding)), "answer": answer})
            r.expire(k, self.ttl)
        except Exception:
            with self._lock:
                self._local.append((list(query_embedding), answer, time.time()))
                if len(self._local) > self.max_entries:
                    self._local = self._local[-self.max_entries:]

    def clear(self):
        """Flush all semantic cache entries (call after knowledge base updates)."""
        try:
            r = _get_client()
            for k in r.scan_iter("sc:*", count=200):
                r.delete(k)
        except Exception:
            with self._lock:
                self._local = []

    def size(self):
        try:
            return len(list(_get_client().scan_iter("sc:*", count=500)))
        except Exception:
            return len(self._local)
