"""
Agent Runner 单元测试
✅ Mock 模块级 llm_client 避免真实 API 调用
✅ 覆盖：无工具调用、单轮工具调用、多轮工具调用、最大迭代兜底、stream模式、未知工具、无效JSON参数
"""
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ✅ Mock knowledge_search_handler 避免触发真实检索
sys.modules.setdefault("ai_rag.services.rag_service", MagicMock())

from ai_rag.agent.runner import agent_run, _execute_tool, get_available_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_choice(content=None, tool_calls=None):
    """构造 OpenAI ChatCompletionMessage 模拟对象"""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    # model_dump 需要返回可序列化的 dict
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (tool_calls or [])
        ],
    }
    choice = MagicMock()
    choice.message = msg
    return choice


def _make_tool_call(call_id: str, name: str, arguments: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture
def mock_llm():
    """Mock runner 模块级的 llm_client"""
    with patch("ai_rag.agent.runner.llm_client") as m:
        m.chat.completions.create = AsyncMock()
        yield m


@pytest.fixture
def mock_knowledge_search():
    """Mock TOOL_REGISTRY 中的 knowledge_search 工具"""
    with patch.dict(
        "ai_rag.agent.runner.TOOL_REGISTRY",
        {"knowledge_search": AsyncMock(return_value="模拟检索结果")}
    ) as mock_registry:
        yield mock_registry["knowledge_search"]


# ---------------------------------------------------------------------------
# _execute_tool 测试
# ---------------------------------------------------------------------------

class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_knowledge_search_success(self, mock_knowledge_search):
        mock_knowledge_search.return_value = "检索结果片段"
        result = await _execute_tool("knowledge_search", {"query": "测试问题"})
        assert result == "检索结果片段"
        mock_knowledge_search.assert_awaited_once_with(query="测试问题")

    @pytest.mark.asyncio
    async def test_knowledge_search_empty_query(self, mock_knowledge_search):
        mock_knowledge_search.return_value = ""
        result = await _execute_tool("knowledge_search", {})
        assert result == ""
        mock_knowledge_search.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_json(self):
        result = await _execute_tool("nonexistent_tool", {"foo": "bar"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "nonexistent_tool" in parsed["error"]


# ---------------------------------------------------------------------------
# agent_run 测试
# ---------------------------------------------------------------------------

class TestAgentRunNoStream:
    """stream=False 场景（默认）"""

    @pytest.mark.asyncio
    async def test_direct_answer_no_tool_call(self, mock_llm):
        """LLM 直接回复，不调用任何工具"""
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[_make_choice(content="这是直接回答")]
        )
        answer, knowledge = await agent_run("sess-1", "你好")
        assert answer == "这是直接回答"
        assert knowledge == ""
        mock_llm.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_tool_call_then_answer(self, mock_llm, mock_knowledge_search):
        """一轮工具调用后 LLM 给出最终回答"""
        mock_knowledge_search.return_value = "知识库片段A"
        mock_llm.chat.completions.create.side_effect = [
            # 第1轮：LLM 请求调用工具
            MagicMock(choices=[_make_choice(
                tool_calls=[_make_tool_call("call_1", "knowledge_search", {"query": "测试"})]
            )]),
            # 第2轮：LLM 根据工具结果生成最终回答
            MagicMock(choices=[_make_choice(content="根据知识库[1]，答案是X")]),
        ]
        answer, knowledge = await agent_run("sess-2", "查一下测试")
        assert answer == "根据知识库[1]，答案是X"
        assert "知识库片段A" in knowledge
        assert mock_llm.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_accumulate_chunks(self, mock_llm, mock_knowledge_search):
        """多轮工具调用，retrieved_chunks 应累积"""
        mock_knowledge_search.side_effect = ["片段A", "片段B"]
        mock_llm.chat.completions.create.side_effect = [
            MagicMock(choices=[_make_choice(
                tool_calls=[_make_tool_call("c1", "knowledge_search", {"query": "q1"})]
            )]),
            MagicMock(choices=[_make_choice(
                tool_calls=[_make_tool_call("c2", "knowledge_search", {"query": "q2"})]
            )]),
            MagicMock(choices=[_make_choice(content="综合[1][2]回答")]),
        ]
        answer, knowledge = await agent_run("sess-3", "复杂问题")
        assert answer == "综合[1][2]回答"
        assert "片段A" in knowledge
        assert "片段B" in knowledge
        assert "---" in knowledge  # 分隔符
        assert mock_llm.chat.completions.create.await_count == 3

    @pytest.mark.asyncio
    async def test_max_iterations_fallback(self, mock_llm, mock_knowledge_search):
        """达到5轮上限应返回 fallback"""
        mock_knowledge_search.return_value = "some result"
        # 5轮都返回 tool_call，永远不给最终答案
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[_make_choice(
                tool_calls=[_make_tool_call("cx", "knowledge_search", {"query": "loop"})]
            )]
        )
        answer, knowledge = await agent_run("sess-4", "无限循环问题")
        assert "最大推理轮次" in answer
        assert knowledge == ""
        assert mock_llm.chat.completions.create.await_count == 5

    @pytest.mark.asyncio
    async def test_invalid_tool_arguments_json(self, mock_llm, mock_knowledge_search):
        """工具参数 JSON 解析失败时应优雅降级"""
        mock_knowledge_search.return_value = "降级结果"
        bad_tc = MagicMock()
        bad_tc.id = "bad_call"
        bad_tc.function.name = "knowledge_search"
        bad_tc.function.arguments = "{invalid json!!!"

        mock_llm.chat.completions.create.side_effect = [
            MagicMock(choices=[_make_choice(tool_calls=[bad_tc])]),
            MagicMock(choices=[_make_choice(content="降级回答")]),
        ]
        answer, knowledge = await agent_run("sess-5", "坏参数")
        assert answer == "降级回答"
        # 即使 JSON 坏了，knowledge_search 仍被调用（args={} → query=""）
        mock_knowledge_search.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_none_content_treated_as_empty(self, mock_llm):
        """LLM 返回 content=None 时应返回空字符串"""
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[_make_choice(content=None)]
        )
        answer, knowledge = await agent_run("sess-6", "test")
        assert answer == ""
        assert knowledge == ""


class TestAgentRunStream:
    """stream=True 场景"""

    @pytest.mark.asyncio
    async def test_stream_returns_async_generator(self, mock_llm):
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[_make_choice(content="流式回答")]
        )
        result = await agent_run("sess-7", "hello", stream=True)
        # 应返回 async generator，不是元组
        assert hasattr(result, "__aiter__")
        chunks = [chunk async for chunk in result]
        assert chunks == ["流式回答"]

    @pytest.mark.asyncio
    async def test_stream_max_iterations_fallback(self, mock_llm, mock_knowledge_search):
        mock_knowledge_search.return_value = "x"
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[_make_choice(
                tool_calls=[_make_tool_call("cx", "knowledge_search", {"query": "loop"})]
            )]
        )
        result = await agent_run("sess-8", "loop", stream=True)
        chunks = [chunk async for chunk in result]
        assert len(chunks) == 1
        assert "最大推理轮次" in chunks[0]


# ---------------------------------------------------------------------------
# get_available_tools 测试
# ---------------------------------------------------------------------------

class TestGetAvailableTools:
    def test_returns_list_of_dicts(self):
        tools = get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1
        assert all("name" in t and "description" in t for t in tools)

    def test_contains_knowledge_search(self):
        tools = get_available_tools()
        names = [t["name"] for t in tools]
        assert "knowledge_search" in names