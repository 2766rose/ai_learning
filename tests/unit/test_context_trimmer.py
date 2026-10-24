# -*- coding: utf-8 -*-
"""Unit tests: context_trimmer (预算、工具调用结构和回退路径)。"""
from ai_rag.utils.context_trimmer import (
    _count_message_tokens,
    trim_messages,
)


def _messages():
    return [
        {"role": "system", "content": "S" * 50},
        {"role": "user", "content": "U" * 500},
        {"role": "assistant", "content": "A" * 500},
        {"role": "user", "content": "latest question " + "Q" * 300},
        {"role": "assistant", "content": "latest answer " + "R" * 300},
    ]


def test_within_budget_returns_all():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert trim_messages(messages, max_tokens=4000) == messages


def test_never_exceeds_budget_with_long_history():
    out = trim_messages(_messages(), max_tokens=120)
    assert _count_message_tokens(out) <= 120


def test_keeps_system_and_latest_user_when_over_budget():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "old answer " * 100},
        {"role": "user", "content": "latest question " * 50},
    ]
    out = trim_messages(messages, max_tokens=60)
    assert out[0]["role"] == "system"
    assert any(m.get("role") == "user" for m in out)
    assert _count_message_tokens(out) <= 60


def test_preserves_complete_tool_group():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "knowledge_search", "arguments": {"query": "x"}}}
        ]},
        {"role": "tool", "content": "R" * 50, "tool_call_id": "call_1"},
        {"role": "assistant", "content": "answer"},
    ]
    out = trim_messages(messages, max_tokens=1000)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]


def test_drops_orphan_tool_message():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "tool", "content": "orphan", "tool_call_id": "missing"},
    ]
    out = trim_messages(messages, max_tokens=1000)
    assert [m["role"] for m in out] == ["system"]


def test_dict_tool_calls_does_not_crash():
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
    assert _count_message_tokens(out) <= 120


def test_system_prompt_is_always_first():
    out = trim_messages(_messages(), max_tokens=80)
    assert out[0]["role"] == "system"
