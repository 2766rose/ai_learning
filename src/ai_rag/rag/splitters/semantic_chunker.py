# src/ai_rag/rag/splitters/semantic_chunker.py
"""基于 Embedding 的语义切分器

原理（LangChain SemanticChunker 同款思路）：
1. 先把文本切成句子（中文按句号/感叹号/问号/分号/换行切，Markdown 表格行整体保留）
2. 用 Embedding 模型把每个句子向量化
3. 计算相邻句子的余弦相似度 → 距离
4. 以距离的百分位数作为断点阈值，语义突变处切块
5. 合并成不超过 chunk_size 的块
"""
from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional

import numpy as np

from ai_rag.rag.models import Document
from ai_rag.rag.splitters.base import BaseTextSplitter

logger = logging.getLogger(__name__)

_SENT_END = re.compile(r"([。！？!?；;])")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")


def _split_sentences(text: str) -> List[str]:
    """中文句子切分：按句末标点切，Markdown 表格行整体保留，超长无标点句按长度兜底"""
    sentences: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _TABLE_LINE.match(line):
            sentences.append(line)
            continue
        parts = _SENT_END.split(line)
        buf = ""
        for part in parts:
            if not part:
                continue
            buf += part
            if part in "。！？!?；;":
                if buf.strip():
                    sentences.append(buf.strip())
                buf = ""
            elif len(buf) >= 100:
                sentences.append(buf.strip())
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())
    return sentences


def _cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度"""
    import numpy as _np
    va, vb = _np.asarray(a), _np.asarray(b)
    denom = (_np.linalg.norm(va) * _np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(_np.dot(va, vb) / denom)


class SemanticChunker(BaseTextSplitter):
    """语义切分器：按相邻句子语义相似度的"断崖"处切块"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        breakpoint_percentile: float = 0.75,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap({chunk_overlap}) must be < chunk_size({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.breakpoint_percentile = breakpoint_percentile
        # 默认使用项目全局 embedding 服务（延迟导入，避免循环依赖）
        self._embedding_fn = embedding_fn

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self._embedding_fn is not None:
            return self._embedding_fn(texts)
        from ai_rag.core.embeddings import embedding_service
        return embedding_service.embed_texts(texts)

    def _find_boundaries(self, sentences: List[str]) -> set:
        """返回需要断开的句子下标集合（i 与 i-1 之间断开）"""
        if len(sentences) < 2:
            return set()
        embeddings = self._embed(sentences)
        distances = [
            1.0 - _cosine(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        if not distances:
            return set()
        threshold = float(np.percentile(distances, self.breakpoint_percentile * 100))
        logger.info(
            "SemanticChunker | sentences=%d | dist_range=(%.3f, %.3f) | threshold=%.3f",
            len(sentences), min(distances), max(distances), threshold,
        )
        return {i for i, d in enumerate(distances) if d > threshold}

    def split_text(self, text: str) -> List[str]:
        """文本 → 语义块列表"""
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        boundaries = self._find_boundaries(sentences)

        chunks: List[str] = []
        current = ""
        for i, sent in enumerate(sentences):
            # 语义断点处：若已积累内容则封块
            if i - 1 in boundaries and current:
                chunks.append(current)
                current = sent
                continue
            # 超长时按 chunk_size 兜底切分
            if current and len(current) + len(sent) + 1 > self.chunk_size:
                chunks.append(current)
                current = sent
                continue
            current = (current + " " + sent).strip() if current else sent
        if current:
            chunks.append(current)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """兼容 BaseTextSplitter 接口：List[Document] → List[Document]"""
        out: List[Document] = []
        for doc in documents:
            for chunk in self.split_text(doc.content):
                out.append(Document(
                    content=chunk,
                    metadata={**doc.metadata, "splitter": "semantic"},
                    doc_id=doc.doc_id,
                ))
        return out