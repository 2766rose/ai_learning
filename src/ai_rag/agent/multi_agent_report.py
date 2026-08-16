import json
import asyncio
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# ⚡ 核心复用：直接导入你已调通的单 Agent 类和工具
from ai_rag.agent.react_agent_langgraph import LangGraphAgent, get_system_prompt

# ==========================================
# 1. 定义协作专属 State
# ==========================================
class ReportState(TypedDict):
    messages: Annotated[list, add_messages]  # 用户原始对话
    research_findings: str                   # 研究员输出的纯文本事实（避免JSON解析坑）
    report_draft: str                        # 撰稿人生成的最终报告
    revision_count: int                      # 安全阀计数器


# ==========================================
# 2. 节点实现
# ==========================================
class ReportWorkflow:
    def __init__(self):
        # 复用现有的 MemorySaver，保证会话一致性
        self.memory = MemorySaver()
        
        # 实例化研究员（复用完整的 ReAct + 工具链）
        self.researcher_agent = LangGraphAgent()
        
        # 撰稿人使用独立的 LLM（不绑定任何工具）
        # 注意：这里直接从 researcher 中获取 llm 实例，避免重复初始化
        self.writer_llm = self.researcher_agent.llm 
        
        self.graph = self._build_graph()

    async def _researcher_node(self, state: ReportState, config: RunnableConfig):
        """研究员节点：调用现有 Agent 进行检索"""
        print("--- 🔍 [Node: Researcher] 正在检索知识库与记忆 ---")
        user_query = state["messages"][-1].content
        
        # 💡 关键技巧：通过 astream_events 捕获研究员的最终输出
        # 而不是解析中间态的 tool_calls，彻底避开 JSON 畸形问题
        final_content = []
        async for event in self.researcher_agent.graph.astream_events(
            {"messages": [HumanMessage(content=f"请仅检索并总结以下问题的客观事实，不要生成完整报告：{user_query}")]}, 
            config=config, 
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    final_content.append(chunk)
        
        findings = "".join(final_content).strip()
        
        # 如果研究员没返回有效内容，标记为空以触发重试
        if not findings or len(findings) < 10:
            findings = ""
            
        return {
            "research_findings": findings,
            "revision_count": state.get("revision_count", 0) + 1
        }

    def _quality_gate(self, state: ReportState) -> Literal["revise", "write"]:
        """质量门禁：简单规则判断，不消耗 Token"""
        findings = state.get("research_findings", "")
        count = state.get("revision_count", 0)
        
        # 安全阀：最多重试2次；且要求事实长度达标
        if count >= 2 or len(findings) >= 50:
            print(f"--- ✅ [Gate] 质量检查通过 (重试次数: {count}, 事实长度: {len(findings)}) ---")
            return "write"
        
        print(f"--- ⚠️ [Gate] 信息不足，触发重试 (当前次数: {count}) ---")
        return "revise"

    async def _writer_node(self, state: ReportState, config: RunnableConfig):
        """撰稿人节点：纯粹的内容生成，无工具绑定"""
        print("--- ✍️ [Node: Writer] 正在撰写行业分析报告 ---")
        
        prompt = f"""你是资深行业分析师。请严格基于以下【事实依据】撰写专业分析报告。
        
【事实依据】
{state['research_findings']}

【用户原始问题】
{state['messages'][-1].content}

【写作要求】
1. 使用 Markdown 格式，包含摘要、核心发现、风险提示三个部分
2. 严禁添加【事实依据】中未提及的任何信息
3. 语言风格：专业、客观、数据驱动"""

        response = await self.writer_llm.ainvoke([SystemMessage(content=prompt)])
        return {"report_draft": response.content}

    def _build_graph(self):
        workflow = StateGraph(ReportState)
        
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("writer", self._writer_node)
        
        workflow.add_edge(START, "researcher")
        workflow.add_conditional_edges("researcher", self._quality_gate, {
            "revise": "researcher",
            "write": "writer"
        })
        workflow.add_edge("writer", END)
        
        return workflow.compile(checkpointer=self.memory)

    async def generate_report(self, query: str, user_id: str, thread_id: str):
        """对外暴露的统一入口"""
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        inputs = {"messages": [HumanMessage(content=query)]}
        
        result = await self.graph.ainvoke(inputs, config=config)
        return result["report_draft"]


# ==========================================
# 3. CLI 测试入口
# ==========================================
async def main():
    workflow = ReportWorkflow()
    
    while True:
        query = input("\n📊 输入分析需求 (q退出): ").strip()
        if query.lower() == 'q': break
            
        report = await workflow.generate_report(
            query=query,
            user_id="emp_test",
            thread_id="thread_report_demo"
        )
        print("\n" + "="*50)
        print(report)
        print("="*50)

if __name__ == "__main__":
    asyncio.run(main())