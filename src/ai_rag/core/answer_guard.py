# -*- coding: utf-8 -*-
"""Anti-hallucination guard decision (pure function, unit-testable)."""
import re

REFUSAL_MESSAGE = "\u62b1\u6b49\uff0c\u77e5\u8bc6\u5e93\u4e2d\u672a\u627e\u5230\u4e0e\u60a8\u95ee\u9898\u76f8\u5173\u7684\u4fe1\u606f\u3002"


def should_refuse(answer: str, kb_had_content: bool, other_tool_content: bool) -> bool:
    """Return True when the answer is an ungrounded factual claim (digits, no grounding)."""
    if not answer:
        return False
    if kb_had_content or other_tool_content:
        return False
    if "\u672a\u627e\u5230" in answer or "\u6ca1\u6709\u627e\u5230" in answer:
        return False
    return re.search(r"\d", answer) is not None
