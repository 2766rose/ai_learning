"""RAG Agent 工具注册中心"""

from src.ai_rag.agent.tools.rag_search import rag_search_tool
from src.ai_rag.agent.tools.doc_upload import doc_upload_tool

TOOL_REGISTRY: dict[str, callable] = {
    "rag_search": rag_search_tool,
    "doc_upload": doc_upload_tool,
}

TOOL_SCHEMAS: list[dict] = [
    rag_search_tool.schema,
    doc_upload_tool.schema,
]

__all__ = ["TOOL_REGISTRY", "TOOL_SCHEMAS"]