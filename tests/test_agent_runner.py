# -*- coding: utf-8 -*-
"""Agent Runner unit tests (current code: Ollama /api/chat, dict tool calls)."""
from ai_rag.agent.runner import (
    SYSTEM_PROMPT,
    get_available_tools,
    _normalize_tool_calls,
    _build_assistant_tool_message,
    _trim_history,
    _should_disable_thinking,
    _is_realtime_tool_query,
    _ensure_weather_evidence,
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


class TestRealtimeToolRouting:
    def test_weather_clothing_query_skips_kb(self):
        assert _is_realtime_tool_query("北京现在适合穿什么衣服？") is True

    def test_weather_umbrella_query_skips_kb(self):
        assert _is_realtime_tool_query("上海今天需要带伞吗？") is True

    def test_time_query_skips_kb(self):
        assert _is_realtime_tool_query("现在几点了？") is True

    def test_company_policy_query_keeps_kb(self):
        assert _is_realtime_tool_query("公司高温天气考勤制度是什么？") is False

    def test_ordinary_query_keeps_kb(self):
        assert _is_realtime_tool_query("红楼梦的作者是谁？") is False

    def test_weather_prompt_requires_specific_metrics(self):
        assert "当前温度" in SYSTEM_PROMPT
        assert "体感温度" in SYSTEM_PROMPT
        assert "最高/最低温度" in SYSTEM_PROMPT


class TestWeatherEvidence:
    def test_adds_facts_when_answer_has_no_units(self):
        evidence = "北京当前天气：晴，气温 8°C（体感 6°C），今日 3°C ~ 15°C。"
        answer = "北京气温较低，建议穿外套。"
        result = _ensure_weather_evidence(answer, evidence)
        assert "8°C" in result
        assert "建议穿外套" in result

    def test_keeps_answer_when_metrics_already_present(self):
        evidence = "北京当前天气：晴，气温 8°C。"
        answer = "北京当前 8°C，建议穿外套。"
        assert _ensure_weather_evidence(answer, evidence) == answer
