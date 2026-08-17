# -*- coding: utf-8 -*-
"""Unit tests: context_trimmer (regression for dict tool_calls crash)."""
from ai_rag.utils.context_trimmer import trim_messages


class TestTrimMessagesDictToolCalls:
    def test_assistant_tool_calls_as_dicts_does_not_crash(self):
        """Regression: dict-based tool_calls must not crash (was AttributeError 'tool_calls')."""
        messages = [
            {"role": "system", "content": "S" * 50},
            {"role": "user", "content": "U" * 100},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "knowledge_search", "arguments": {"query": "x"}}}
            ]},
            {"role": "tool", "content": "R" * 50, "tool_call_id": "call_1"},
            {"role": "assistant", "content": "A" * 200},
        ]
        out = trim_messages(messages, max_tokens=120)
        assert isinstance(out, list)
        assert out

    def test_keeps_system_and_recent_when_over_budget(self):
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U" * 300},
            {"role": "assistant", "content": "A" * 300},
        ]
        out = trim_messages(messages, max_tokens=50)
        assert out[0]["role"] == "system"
        assert len(out) >= 2

    def test_within_budget_returns_all(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        out = trim_messages(messages, max_tokens=4000)
        assert len(out) == 2
