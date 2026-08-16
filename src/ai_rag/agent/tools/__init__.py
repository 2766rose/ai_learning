#src\ai_rag\agent\tools\_init_.py
"""
RAG Agent 工具注册中心 (LangGraph & OpenAI Tool Calling 适配版)
统一管理 Agent 可用工具，并自动生成 OpenAI API 所需的 JSON Schema。
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool, tool

# 导入具体工具实现
from ai_rag.agent.tools.doc_upload import doc_upload_tool
from ai_rag.agent.tools.memory_tool import save_user_memory
from ai_rag.agent.tools.rag_search import rag_search_tool
from ai_rag.agent.tools.weather_tool import get_weather

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 本地工具定义 (使用 @tool 装饰器)
# ==============================================================================

@tool
async def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间。当用户询问现在几点、今天日期、当前时间等实时信息时，必须调用此工具。"""
    try:
        dt = datetime.now(ZoneInfo(timezone))
        return json.dumps({
            "iso_timestamp": dt.isoformat(),
            "unix_epoch": int(dt.timestamp()),
            "timezone": timezone,
            "formatted": dt.strftime("%Y年%m月%d日 %H:%M:%S %Z")
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Invalid timezone: {timezone}, details: {str(e)}"}, ensure_ascii=False)


# ==============================================================================
# 2. 工具注册表 (Registry)
# ==============================================================================

# 原始工具列表 (可能包含未正确装饰的普通函数)
_RAW_TOOLS: List[Any] = [
    rag_search_tool,
    doc_upload_tool,
    get_current_time,
    save_user_memory,
    get_weather,
]

# ✅ 类型守卫：仅保留合法的 LangChain BaseTool 实例
TOOL_REGISTRY: List[BaseTool] = [
    t for t in _RAW_TOOLS if isinstance(t, BaseTool)
]

if len(TOOL_REGISTRY) != len(_RAW_TOOLS):
    logger.warning(
        "⚠️ 工具注册表过滤: 预期 %d 个工具，实际有效 %d 个。请检查是否遗漏 @tool 装饰器。",
        len(_RAW_TOOLS), len(TOOL_REGISTRY)
    )

# ✅ 字典映射 (供 Runner O(1) 复杂度快速分发执行使用)
TOOL_REGISTRY_MAP: Dict[str, BaseTool] = {t.name: t for t in TOOL_REGISTRY}


# ==============================================================================
# 3. OpenAI Tool Calling Schema 自动生成
# ==============================================================================

def _build_tool_schemas(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """将 LangChain @tool 对象转换为 OpenAI Chat Completions tools 参数格式"""
    schemas: List[Dict[str, Any]] = []
    for t in tools:
        try:
            # 优先使用 get_input_schema (LangChain >= 0.2)
            if hasattr(t, "get_input_schema"):
                input_schema = t.get_input_schema().model_json_schema()
            elif hasattr(t, "args_schema") and t.args_schema is not None:
                input_schema = t.args_schema.model_json_schema()
            else:
                logger.warning("工具 '%s' 无法提取 schema，跳过", t.name)
                continue

            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": input_schema,
                }
            })
        except Exception as e:
            logger.error("构建工具 '%s' schema 失败: %s", getattr(t, "name", "unknown"), e)
            continue

    if not schemas:
        logger.error("❌ 未生成任何有效工具 Schema，Agent 将无法调用工具！")

    return schemas


TOOL_SCHEMAS: List[Dict[str, Any]] = _build_tool_schemas(TOOL_REGISTRY)

__all__ = ["TOOL_REGISTRY", "TOOL_REGISTRY_MAP", "TOOL_SCHEMAS"]