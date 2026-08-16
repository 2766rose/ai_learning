# src/ai_rag/retrieval/bm25_engine.py
from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from ai_rag.retrieval.types import RetrievedDoc


class BM25Engine:
    """
    基于 BM25Okapi + jieba 的中文关键词检索引擎。
    支持全量重建与增量追加，线程安全。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.bm25: BM25Okapi | None = None
        self.documents: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    #  全量索引（启动时 / 强制刷新时调用）
    # ------------------------------------------------------------------
    def index(self, documents: List[Dict[str, str]]) -> None:
        """全量重建 BM25 索引。"""
        with self._lock:
            if not documents:
                self.bm25 = None
                self.documents = []
                return

            tokenized_docs = [list(jieba.cut(doc["content"])) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            self.documents = list(documents)  # 防御性拷贝

    # ------------------------------------------------------------------
    #  增量追加（ETL 完成后调用，避免全量重建开销）
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        增量追加文档到现有索引。
        rank_bm25 不支持原生增量，此处采用"合并后重建"策略；
        当文档量较大时应替换为 Elasticsearch / Meilisearch 等外部引擎。
        """
        if not documents:
            return

        with self._lock:
            merged = self.documents + documents
            self.index(merged)

    # ------------------------------------------------------------------
    #  检索
    # ------------------------------------------------------------------
    def search(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[RetrievedDoc, float]]:
        """返回 top_k 个 (RetrievedDoc, score) 元组，按分数降序。"""
        with self._lock:
            if self.bm25 is None or not self.documents:
                return []

            tokens = list(jieba.cut(query))
            scores = self.bm25.get_scores(tokens)

            # 过滤零分项，减少无效排序
            scored_indices = [
                i for i, s in enumerate(scores) if s > 0
            ]
            if not scored_indices:
                return []

            top_indices = sorted(
                scored_indices, key=lambda i: scores[i], reverse=True
            )[:top_k]

            results: List[Tuple[RetrievedDoc, float]] = []
            for idx in top_indices:
                doc_dict = self.documents[idx]
                retrieved = RetrievedDoc(
                    id=str(doc_dict.get("id", "")),
                    content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {}),
                    score=float(scores[idx]),
                )
                results.append((retrieved, float(scores[idx])))

            return results

    @property
    def doc_count(self) -> int:
        """当前已索引文档数。"""
        with self._lock:
            return len(self.documents)