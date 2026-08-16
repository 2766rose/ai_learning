"""记忆工具：供 LLM 主动调用保存用户长期记忆"""
import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from ai_rag.agent.memory import save_memories_with_dedup

logger = logging.getLogger(__name__)

@tool
async def save_user_memory(facts: list[str], config: RunnableConfig) -> str:
    """保存用户长期记忆（带去重）。当用户主动告知个人偏好、事实（如'我喜欢咖啡'、'我在北京工作'）时调用。仅保存稳定事实，忽略闲聊。"""
    # ✅ 1. 从 LangGraph 上下文中安全获取 user_id，绝不让 LLM 传参
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "❌ 错误：未找到当前用户身份，无法保存记忆。"
    
    try:
        # ✅ 2. 原生 async 函数直接 await，彻底避开 asyncio_0 线程池报错
        await save_memories_with_dedup(user_id=user_id, facts=facts)
        return f"✅ 已为用户 {user_id} 成功保存 {len(facts)} 条记忆"
    except Exception as e:
        logger.error(f"保存记忆失败 (user={user_id}): {e}", exc_info=True)
        return f"❌ 保存失败: {str(e)}"

# ⚠️ 注意：使用了 @tool 装饰器后，不再需要手动定义 schema 字典！
# LangChain 会自动根据类型注解 (list[str]) 和 docstring 生成完美的 Schema。