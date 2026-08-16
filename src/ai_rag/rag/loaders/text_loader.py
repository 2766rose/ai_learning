from pathlib import Path
from typing import List
from ai_rag.rag.loaders.base import BaseLoader
from ai_rag.rag.models import Document

class TextLoader(BaseLoader):
    """纯文本文件加载器"""

    def load(self, source: str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        content = path.read_text(encoding="utf-8")
        return [
            Document(
                content=content,
                metadata={"source": str(path), "file_type": "txt"},
                doc_id=path.stem,
            )
        ]