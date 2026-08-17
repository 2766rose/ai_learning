# -*- coding: utf-8 -*-
"""Agent Runner unit tests (current code: Ollama /api/chat, dict tool calls)."""
from ai_rag.agent.runner import (
    get_available_tools,
    _normalize_tool_calls,
    _build_assistant_tool_message,
    _trim_history,
    _should_disable_thinking,
)


class TestNormalizeToolCalls:
    def test_none_returns_empty(self):
        assert _normalize_tool_calls(None) == []

    def test_dict_format(self):
        tcs = [{"id": "c1", "function": {"name": "knowledge_search", "arguments": {"query": "x"}}}]
        assert _normalize_tool_calls(tcs) == [{"id": "c1", "name": "knowledge_search", "arguments": {"query": "x"}}]

    def test_arguments_as_json_string(self):
        tcs = [{"id": "c2", "function": {"name": "get_weather", "arguments": '{"city": "sh"}'}}]
        out = _normalize_tool_calls(tcs)
        assert out[0]["arguments"] == {"city": "sh"}

    def test_invalid_json_arguments_default_empty(self):
        tcs = [{"id": "c3", "function": {"name": "x", "arguments": "{bad"}}]
        assert _normalize_tool_calls(tcs)[0]["arguments"] == {}


class TestBuildAssistantToolMessage:
    def test_structure(self):
        msg = _build_assistant_tool_message("", [{"id": "c1", "name": "knowledge_search", "arguments": {"q": "1"}}])
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["id"] == "c1"
        assert msg["tool_calls"][0]["function"]["name"] == "knowledge_search"

    def test_missing_id_fallback(self):
        msg = _build_assistant_tool_message("", [{"id": "", "name": "t", "arguments": {}}])
        assert msg["tool_calls"][0]["id"].startswith("call_")


class TestTrimHistory:
    def test_keeps_recent_within_budget(self):
        history = [{"role": "user", "content": "hello world " * 20}]
        assert len(_trim_history(history, budget=100000)) == 1

    def test_drops_old_when_over_budget(self):
        history = [{"role": "user", "content": "x" * 50} for _ in range(10)]
        out = _trim_history(history, budget=60)
        assert 0 < len(out) < 10
        assert out[-1] == history[-1]


class TestDisableThinking:
    def test_qwen3(self):
        assert _should_disable_thinking("qwen3:8b") is True

    def test_non_thinking(self):
        assert _should_disable_thinking("qwen2.5:7b") is False


class TestGetAvailableTools:
    def test_contains_knowledge_search(self):
        names = [t["name"] for t in get_available_tools()]
        assert "knowledge_search" in names
