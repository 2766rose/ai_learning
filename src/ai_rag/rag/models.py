from pydantic import BaseModel, Field
from typing import Dict, Any

class Document(BaseModel):
    """RAG 系统统一文档模型"""
    content: str = Field(description="文档文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    doc_id: str = Field(default="", description="文档唯一标识")