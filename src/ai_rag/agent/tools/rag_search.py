# src/ai_rag/agent/tools/rag_search.py
"""RAG 检索工具 - 供 Agent 调用以获取企业知识库内容"""
import logging

from langchain_core.tools import tool

from ai_rag.services.rag_service import knowledge_search_handler

logger = logging.getLogger(__name__)


@tool("knowledge_search")
async def rag_search_tool(query: str, domain: str = "company") -> str:
    """
    从企业知识库中检索与用户问题相关的文档片段。
    当用户询问公司内部制度、产品参数、业务数据、历史文档时必须调用此工具。
    返回带相似度评分和来源编号的文本片段。
    """
    logger.info("[Tool:rag_search] Executing search | query='%s' | domain=%s", query[:80], domain)
    try:
        result = await knowledge_search_handler(query=query, domain=domain)
        logger.info("[Tool:rag_search] Search completed | result_len=%d", len(result))
        return result
    except Exception as e:
        logger.error("[Tool:rag_search] Search failed | error=%s", str(e))
        return f"检索失败: {str(e)}"