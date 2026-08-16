# src/ai_rag/agent/react_agent_langgraph.py
import os
import sys
import asyncio
import tiktoken
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from ai_rag.core.config import rag_config

from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
    trim_messages, RemoveMessage
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from ai_rag.agent.tools import TOOL_REGISTRY
from ai_rag.agent.memory import retrieve_memories

load_dotenv()


# ==========================================
# 0. 通用 Token 计数器
# ==========================================
def count_tokens_for_qwen(messages: list[BaseMessage]) -> int:
    """针对 Qwen 等兼容模型的通用 Token 计数器"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # ⚡【P2 修复】：保护 content=None 的情况
        total_chars = sum(len(str(m.content or "")) for m in messages)
        return max(1, int(total_chars / 1.5))

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        content_str = str(message.content or "")
        num_tokens += len(encoding.encode(content_str))
        if hasattr(message, 'tool_calls') and message.tool_calls:
            num_tokens += len(encoding.encode(str(message.tool_calls)))
    return num_tokens


# ==========================================
# 1. 定义全局工具
# ==========================================
@tool
async def recall_user_memory(query: str, config: RunnableConfig) -> str:
    """检索用户的长期记忆。当用户询问与自己相关的过往信息时调用。"""
    user_id = config["configurable"].get("user_id")
    if not user_id:
        return "❌ 错误：未找到当前用户身份，无法检索记忆。"

    memories = await retrieve_memories(user_id=user_id, query=query)
    return "用户相关记忆：" + "；".join(memories) if memories else "未找到与该用户相关的记忆"

# 仅合并外部工具集 + 本文件中独有的工具
TOOLS = TOOL_REGISTRY + [recall_user_memory]


# ==========================================
# 2. 定义全局状态
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ==========================================
# 3. 核心提示词
# ==========================================
def get_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是一个智能工业运维助手，具备长期记忆与知识库检索能力。
今天的日期是：{today}。

你的工作准则：
1. 当用户明确表达个人信息、偏好或背景时，必须调用 save_user_memory 保存
2. 当用户询问与自己相关的过往信息时，必须先调用 recall_user_memory 检索
3. 当用户询问公司制度、设备手册（如 X-9000）或专业知识时，调用 rag_search_tool 检索知识库
4. 当用户询问时间相关问题时，调用 get_current_time
5. 回答必须严谨，如果是设备故障问题，必须优先提示安全警告。"""


# ==========================================
# 4. LangGraph Agent 核心类
# ==========================================
class LangGraphAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=rag_config.OPENAI_MODEL,
            api_key=rag_config.OPENAI_API_KEY,
            base_url=rag_config.OPENAI_BASE_URL,
            temperature=rag_config.LLM_TEMPERATURE,
        )

        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        # ⚡【P1 修复】：消息清洗防御层，拦截畸形 ToolMessage
        def sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
            if not messages:
                return []

            valid_tool_call_ids = set()
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if isinstance(tc, dict) and "id" in tc:
                            valid_tool_call_ids.add(tc["id"])

            cleaned = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    if not msg.content and not msg.tool_call_id:
                        print(f"⚠️ [Sanitizer] 过滤无效 ToolMessage (空内容+无ID): {msg.id}")
                        continue
                    if msg.tool_call_id and msg.tool_call_id not in valid_tool_call_ids:
                        print(f"⚠️ [Sanitizer] 过滤孤立 ToolMessage: {msg.tool_call_id}")
                        continue
                cleaned.append(msg)

            return cleaned

        def call_model(state: AgentState):
            print("--- 🧠 [Node: Agent] 思考中 ---")
            history = list(state["messages"])

            trimmed_history = trim_messages(
                history,
                max_tokens=4000,
                token_counter=count_tokens_for_qwen,
                strategy="last",
                include_system=False,
                allow_partial=False,
                start_on="human",
            )

            trimmed_ids = {m.id for m in trimmed_history if m.id}
            messages_to_remove = [
                RemoveMessage(id=m.id) for m in history
                if m.id and m.id not in trimmed_ids
            ]

            if messages_to_remove:
                print(f"✂️ [Trimmer] 从 State 中永久移除 {len(messages_to_remove)} 条旧消息")

            system_msg = SystemMessage(content=get_system_prompt())
            messages_for_llm = [system_msg] + trimmed_history

            # ⚡ 送给 LLM 前清洗
            messages_for_llm = sanitize_messages(messages_for_llm)

            response = self.llm_with_tools.invoke(messages_for_llm)
            return {"messages": messages_to_remove + [response]}

        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "__end__"

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)

    async def chat(self, user_input: str, user_id: str, thread_id: str):
        inputs = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        async for event in self.graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            elif kind == "on_tool_start":
                print(f"\n🛠️ [Action] 调用工具: {event['name']}, 参数: {event['data'].get('input')}")
            elif kind == "on_tool_end":
                output = event['data'].get('output')
                print(f"\n👁️ [Observation] 工具返回: {str(output)[:100]}...\n")


# ==========================================
# 5. CLI 启动入口
# ==========================================
async def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else input("🔑 请输入用户ID: ").strip() or "emp_test"
    thread_id = f"thread_{user_id}"

    agent = LangGraphAgent()
    print(f"🤖 LangGraph Agent 已启动 | 当前用户: {user_id} | 会话ID: {thread_id}")

    while True:
        try:
            user_input = input("\n🧑 User: ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue

            await agent.chat(user_input, user_id=user_id, thread_id=thread_id)
            print()
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    asyncio.run(main())