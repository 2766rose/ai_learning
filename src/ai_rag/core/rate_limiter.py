# src/ai_rag/core/rate_limiter.py
"""用户级限流（滑动窗口，进程内实现）"""
import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, limit: int = 20, window: int = 60):
        self.limit = limit          # 窗口内允许次数
        self.window = window        # 窗口秒数
        self._hits = {}             # key -> [timestamps]
        self._lock = threading.Lock()

    def allow(self, key: str):
        """返回 (是否允许, 建议等待秒数)；超限返回 (False, retry_after)"""
        now = time.time()
        with self._lock:
            ts = self._hits.setdefault(key, [])
            ts = [t for t in ts if now - t < self.window]
            if len(ts) >= self.limit:
                retry = int(self.window - (now - ts[0])) + 1
                logger.warning("限流触发 | key=%s | retry=%ds", key, retry)
                return False, max(1, retry)
            ts.append(now)
            self._hits[key] = ts
            return True, 0


rate_limiter = RateLimiter()
