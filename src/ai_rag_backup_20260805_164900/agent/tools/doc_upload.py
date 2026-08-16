"""文档上传工具 - 供 Agent 触发文档入库"""


def doc_upload_tool(file_path: str) -> str:
    """将指定文件解析并写入向量数据库"""
    # TODO: 接入你的文档解析 + 向量化 pipeline
    return f"[Doc Upload] file='{file_path}' — 暂未接入上传后端"


doc_upload_tool.schema = {
    "type": "function",
    "function": {
        "name": "doc_upload",
        "description": "上传并解析文档，将其内容写入知识库",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "待上传文件的绝对路径"},
            },
            "required": ["file_path"],
        },
    },
}