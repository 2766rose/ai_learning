from abc import ABC, abstractmethod
from typing import List
from ai_rag.rag.models import Document

class BaseLoader(ABC):
    """文档加载器抽象基类"""

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """从指定来源加载文档列表"""
        ...