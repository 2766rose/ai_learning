# tools.py
from ai_rag.tasks.document_tasks import process_document_task
from ai_rag.services.rag_service import knowledge_search_handler

TOOL_REGISTRY = {
    "knowledge_search": knowledge_search_handler,
    "parse_document": lambda **kw: process_document_task.delay(**kw).id,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "Retrieve relevant document snippets from enterprise knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_document",
            "description": "Submit async document parsing task, returns task_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["file_path", "filename"],
            },
        },
    },
]