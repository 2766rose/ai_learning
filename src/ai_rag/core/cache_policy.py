# -*- coding: utf-8 -*-
"""Cache policy: whether an answer may enter the semantic cache (pure function)."""
import re

REFUSAL_MARKERS = ("\u672a\u627e\u5230", "\u6ca1\u6709\u627e\u5230")  # ??? / ????


def is_cacheable(answer: str) -> bool:
    """Only cache grounded answers (with [n] source citations), never refusals."""
    if not answer or len(answer) < 10:
        return False
    if any(m in answer for m in REFUSAL_MARKERS):
        return False
    return re.search(r"\[\d+\]", answer) is not None
