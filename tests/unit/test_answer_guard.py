# -*- coding: utf-8 -*-
"""Unit tests: anti-hallucination guard (should_refuse)."""
from ai_rag.core.answer_guard import should_refuse


class TestShouldRefuse:
    def test_kb_content_grounds(self):
        assert should_refuse("the answer is 100 [1]", True, False) is False

    def test_other_tool_grounds(self):
        assert should_refuse("weather today is 18C", False, True) is False

    def test_already_refusing(self):
        assert should_refuse("\u62b1\u6b49\uff0c\u77e5\u8bc6\u5e93\u4e2d\u672a\u627e\u5230\u4e0e\u60a8\u95ee\u9898\u76f8\u5173\u7684\u4fe1\u606f\u3002", False, False) is False

    def test_empty_answer(self):
        assert should_refuse("", False, False) is False

    def test_no_digits_kept(self):
        assert should_refuse("hello there", False, False) is False

    def test_hallucination_detected(self):
        assert should_refuse("\u52a0\u73ed\u8d39\u6309150%\u652f\u4ed8", False, False) is True

    def test_short_hallucination_with_digit(self):
        assert should_refuse("\u653e1\u5929", False, False) is True
