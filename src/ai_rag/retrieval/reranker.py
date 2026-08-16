# src/ai_rag/retrieval/reranker.py
"""BGE-Reranker 交叉编码器精排（第3周）
- 使用 sentence-transformers CrossEncoder 加载 bge-reranker 系列模型
- 模型默认 BAAI/bge-reranker-base，可用环境变量 RAG_RERANKER_MODEL 指定
- 下载走 HF，国内可设 HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# 项目根目录（src/ai_rag/retrieval/reranker.py → 上溯4级 = 项目根）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_RERANKER = str(_PROJECT_ROOT / "models" / "bge-reranker-base")
DEFAULT_RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", _LOCAL_RERANKER)


class Reranker:
    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        self.model_name = model_name or DEFAULT_RERANKER_MODEL
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # 本地目录缺失时，走 HF 缓存/镜像（设置后再导入 sentence_transformers 才生效）
            os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "models" / "hf_cache"))
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import CrossEncoder
            logger.info("加载 Reranker 模型: %s (device=%s)", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
            logger.info("Reranker 加载完成")

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """对候选片段打分并返回 top_k（保留原 dict，score 字段覆盖为精排分）"""
        if not candidates:
            return []
        await asyncio.to_thread(self._ensure_loaded)
        pairs = [(query, c["document"]) for c in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        for c, s in ranked:
            c["score"] = float(s)
        return [c for c, _ in ranked][:top_k]


reranker = Reranker()