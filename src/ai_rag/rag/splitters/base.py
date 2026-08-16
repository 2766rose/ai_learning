# src/ai_rag/rag/splitters/base.py
from abc import ABC, abstractmethod
from typing import List
from ai_rag.rag.models import Document

class BaseTextSplitter(ABC):
    """文本分割器抽象基类"""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档列表分割为更小的块"""
        ...