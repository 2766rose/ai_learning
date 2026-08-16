#src\ai_rag\agent\react_agent.py
import json
import os
import sys
import asyncio
from openai import AsyncOpenAI
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from ai_rag.core.config import rag_config

# 导入记忆模块
from ai_rag.agent.memory import save_memories_with_dedup, retrieve_memories

load_dotenv()

SYSTEM_PROMPT = """你是一个智能助手，具备以下能力：
1. 使用工具查询时间、检索公司内部知识库
2. 【长期记忆】你能记住用户的个人偏好、事实和背景信息
   - 当用户明确表达个人信息（如偏好、身份、经历）时，必须调用 save_user_memory 保存
   - 当用户询问与自己相关的过往信息时，必须先调用 recall_user_memory 检索
3. 如果没有相关记忆或知识，如实告知，不要编造
4. 不需要工具时，直接用自然语言回答"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的系统日期和时间。当用户询问'现在几点'、'今天日期'等时间相关问题时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在公司内部知识库中检索信息。当用户询问公司制度、项目细节或专业术语时调用。不要用于闲聊或个人记忆。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索的核心关键词或语义查询"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_memory",
            "description": "保存用户的个人事实、偏好或背景信息。当用户说'我喜欢...'、'我在...工作'、'我的...是...'等表达个人信息时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "要保存的用户个人事实，用一句话简洁表述"
                    }
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_user_memory",
            "description": "检索用户的长期记忆。当用户询问与自己相关的过往信息时调用，如'我平时喝什么'、'我之前说过什么'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索用户记忆的语义查询"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


async def execute_tool(tool_name: str, tool_args: dict, user_id: str) -> str:
    """执行工具并返回 Observation（异步版本）"""
    print(f"🛠️ [Action] 执行工具: {tool_name}, 参数: {tool_args}")

    try:
        if tool_name == "get_current_time":
            result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elif tool_name == "search_knowledge_base":
            query = tool_args.get("query", "")
            result = f"知识库检索结果：关于「{query}」，公司规定需在提交PR前完成Code Review。"

        elif tool_name == "save_user_memory":
            fact = tool_args.get("fact", "").strip()
            if not fact:
                result = "错误：fact 参数不能为空"
            else:
                await save_memories_with_dedup(user_id=user_id, facts=[fact])
                result = f"✅ 已成功记住：{fact}"

        elif tool_name == "recall_user_memory":
            query = tool_args.get("query", "").strip()
            memories = await retrieve_memories(user_id=user_id, query=query)
            if memories:
                result = "用户相关记忆：" + "；".join(memories)
            else:
                result = "未找到与该用户相关的记忆"

        else:
            result = f"错误：未知工具 {tool_name}"

    except Exception as e:
        result = f"工具执行异常: {type(e).__name__}: {str(e)}"

    print(f"👁️ [Observation] 工具返回: {result}\n")
    return result


class ReActAgent:
    def __init__(self, user_id: Optional[str] = None):
        """
        初始化 ReAct Agent
        :param user_id: 用户唯一标识，支持构造时传入或通过 set_user_id 动态绑定
        """
        self._user_id = user_id.strip() if user_id else None
        self.client = AsyncOpenAI(
            api_key=rag_config.OPENAI_API_KEY,
            base_url=rag_config.OPENAI_BASE_URL,
            timeout=rag_config.LLM_TIMEOUT,
        )
        self.max_iterations = 5
        # 使用 property 确保安全初始化消息历史
        self._reset_messages()

    @property
    def user_id(self) -> str:
        """安全获取 user_id，未设置时抛出明确异常防止数据污染"""
        if not self._user_id:
            raise ValueError(
                "❌ user_id 未设置！请在初始化时传入，或通过 set_user_id() 动态绑定。"
            )
        return self._user_id

    def set_user_id(self, user_id: str) -> None:
        """
        运行时动态更新 user_id
        💡 切换用户时自动重置对话历史，防止不同用户上下文串台
        """
        if not user_id or not isinstance(user_id, str):
            raise ValueError(f"❌ 无效的 user_id: {user_id!r}")
        self._user_id = user_id.strip()
        self._reset_messages()
        print(f"🔄 Agent 已切换至用户: {self._user_id}")

    def _reset_messages(self) -> None:
        """重置对话历史为初始状态"""
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def chat(self, user_input: str) -> str:
        # ✅ 在入口处统一校验 user_id，避免延迟到工具执行时报错
        current_user = self.user_id
        self.messages.append({"role": "user", "content": user_input})

        for i in range(self.max_iterations):
            print(f"--- 循环轮次 {i + 1} (user={current_user}) ---")

            response = await self.client.chat.completions.create(
                model=rag_config.OPENAI_MODEL,
                messages=self.messages,
                tools=tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                # 将 assistant 消息加入历史（过滤 None 值避免 API 报错）
                msg_dict = {k: v for k, v in assistant_message.model_dump().items() if v is not None}
                self.messages.append(msg_dict)

                # 逐个执行工具
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    observation = await execute_tool(func_name, func_args, current_user)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation
                    })
            else:
                # 最终回复
                msg_dict = {k: v for k, v in assistant_message.model_dump().items() if v is not None}
                self.messages.append(msg_dict)

                final_answer = assistant_message.content or ""
                print(f"💡 [Final Answer]: {final_answer}")
                return final_answer

        return "达到最大循环次数，Agent 停止执行。"


async def main():
    # ✅ P0: 支持命令行参数 / 交互式输入 / 默认测试账号三级降级
    cli_user_id = sys.argv[1] if len(sys.argv) > 1 else None

    if not cli_user_id:
        try:
            cli_user_id = input("🔑 请输入用户ID (直接回车使用默认测试账号): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            return

        if not cli_user_id:
            cli_user_id = "emp_test_default"

    agent = ReActAgent(user_id=cli_user_id)
    print(f"🤖 ReAct Agent 已启动 | 当前用户: {agent.user_id} | 输入 q 退出 | 输入 /switch <uid> 切换用户")

    while True:
        try:
            user_input = input("\n🧑 User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if user_input.lower() in ("q", "quit", "exit"):
            print("👋 Bye!")
            break
        if not user_input:
            continue

        # 💡 运行时切换用户命令
        if user_input.startswith("/switch "):
            new_uid = user_input.split(" ", 1)[1].strip()
            try:
                agent.set_user_id(new_uid)
            except ValueError as e:
                print(str(e))
            continue

        try:
            await agent.chat(user_input)
        except ValueError as e:
            print(str(e))
        except Exception as e:
            print(f"❌ 对话异常: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # ✅ 兜底捕获 Ctrl+C，防止 Windows PowerShell 下打印 CancelledError 堆栈
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 检测到中断信号，Agent 已安全退出。")