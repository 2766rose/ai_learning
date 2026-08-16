# src/ai_rag/agent/tools/doc_upload.py
"""文档上传工具 - 供 Agent 触发文档入库"""
from langchain_core.tools import tool


@tool("doc_upload")
def doc_upload_tool(file_path: str) -> str:
    """将指定文件解析并写入向量数据库。当用户要求上传文件、导入文档时调用。"""
    # TODO: 接入你的文档解析 + 向量化 pipeline
    return f"[Doc Upload] file='{file_path}' — 暂未接入上传后端"