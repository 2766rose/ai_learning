# -*- coding: utf-8 -*-
"""Unit tests: cache policy (is_cacheable)."""
from ai_rag.core.cache_policy import is_cacheable


class TestIsCacheable:
    def test_empty_and_none(self):
        assert is_cacheable("") is False
        assert is_cacheable(None) is False

    def test_too_short(self):
        assert is_cacheable("abc") is False

    def test_refusal_never_cached(self):
        assert is_cacheable("\u62b1\u6b49\uff0c\u77e5\u8bc6\u5e93\u4e2d\u672a\u627e\u5230\u4e0e\u60a8\u95ee\u9898\u76f8\u5173\u7684\u4fe1\u606f\u3002") is False

    def test_no_citation_not_cacheable(self):
        assert is_cacheable("the answer is 12 percent.") is False

    def test_with_citation_cacheable(self):
        assert is_cacheable("the answer is 12 percent [1].") is True

    def test_multiple_citations(self):
        assert is_cacheable("rule A [1] and rule B [2].") is True

    def test_bare_number_is_not_citation(self):
        assert is_cacheable("answer 1, source 2.") is False
