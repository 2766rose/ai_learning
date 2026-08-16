# src/ai_rag/core/circuit_breaker.py
"""熔断器（LLM 连续失败则快速失败，防止雪崩）"""
import logging
import threading
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown: int = 30):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is not None:
                if time.time() - self._opened_at >= self.cooldown:
                    self._opened_at = None  # 半开：放一个探针请求
                    logger.info("熔断器半开，放行探针")
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()
                logger.warning("熔断器打开 | 连续失败 %d 次，冷却 %ds", self._failures, self.cooldown)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and time.time() - self._opened_at < self.cooldown


llm_circuit_breaker = CircuitBreaker()
