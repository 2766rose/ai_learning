# tests/test_router.py
"""
Router 模块单元测试
✅ Mock 掉 LocalAgent 避免触发 Ollama 网络检查
✅ 覆盖路由决策、配置校验、客户端工厂分支
"""
import sys
from unittest.mock import MagicMock

# ✅ Mock 实际路径下的 LocalAgent，阻止 __init__ 中的 _verify_service 网络调用
sys.modules["ai_rag.core.local_agent"] = MagicMock()

import pytest
from ai_rag.agent.router import route, _load_rules, get_client, RouterRules


class TestRouteDecision:
    """route() 核心逻辑测试"""

    def test_cloud_keyword_priority_over_local(self):
        """同一 query 同时命中 cloud 和 local 时，cloud 优先"""
        assert route("请帮我分析并总结这份合同") == "cloud"

    def test_local_keyword_match(self):
        assert route("这段话是什么意思") == "local"

    def test_cloud_keyword_match(self):
        assert route("请做一下风险评估") == "cloud"

    def test_default_fallback_when_no_match(self):
        assert route("今天天气不错") == "cloud"

    def test_empty_query_uses_default(self):
        rules = _load_rules()
        assert route("") == rules.default

    def test_case_insensitive_matching(self):
        """验证 q_lower 逻辑不会误伤无匹配场景"""
        assert route("SUMMARY") == "cloud"  # 无匹配 → default


class TestLoadRules:
    """配置加载与校验测试"""

    def test_rules_loaded_successfully(self):
        rules = _load_rules()
        assert isinstance(rules, RouterRules)
        assert len(rules.local_keywords) > 0
        assert len(rules.cloud_keywords) > 0
        assert rules.default in ("local", "cloud")

    def test_rules_cache_works(self):
        rules1 = _load_rules()
        rules2 = _load_rules()
        assert rules1 is rules2


class TestGetClient:
    """get_client() 工厂函数分支测试"""

    def test_explicit_cloud_decision(self):
        client = get_client(decision="cloud")
        assert client is not None

    def test_explicit_local_decision(self):
        """传入 local 时应返回 Mock 后的 LocalAgent 实例"""
        client = get_client(decision="local")
        assert client is not None

    def test_query_triggers_route(self):
        client = get_client(query="这是什么意思")
        assert client is not None

    def test_no_args_returns_default(self):
        client = get_client()
        assert client is not None