# test_local_agent.py
"""
LocalAgent 单元测试
✅ 完全 Mock requests 层，不依赖真实 Ollama 服务
✅ 覆盖初始化校验、正常对话、流式解析、超时/网络异常、结构化错误码
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from ai_rag.core.local_agent import LocalAgent, LocalAgentError


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def mock_verify_service():
    """Mock 掉 __init__ 中的 _verify_service，避免每个测试都触发网络检查"""
    with patch.object(LocalAgent, "_verify_service"):
        yield


@pytest.fixture
def agent(mock_verify_service):
    """返回一个跳过服务校验的 LocalAgent 实例"""
    return LocalAgent(model_name="qwen3:8b")



def _make_stream_response(chunks: list[dict], status_code=200):
    """构造模拟 Ollama /api/generate 流式响应的 Mock 对象"""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.iter_lines.return_value = [
        json.dumps(chunk).encode("utf-8") for chunk in chunks
    ]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)
    return mock_resp


# ─── 初始化校验测试 ──────────────────────────────────────────

class TestInitVerification:
    """_verify_service 启动校验逻辑"""

    @patch("ai_rag.core.local_agent.requests.get")
    def test_init_success(self, mock_get):
        """Ollama 可用且模型存在 → 正常初始化"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "qwen3:8b"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        agent = LocalAgent(model_name="qwen3:8b")
        assert agent.model_name == "qwen3:8b"

    @patch("ai_rag.core.local_agent.requests.get")
    def test_init_model_not_found(self, mock_get):
        """模型未加载 → MODEL_NOT_FOUND"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(LocalAgentError, match="MODEL_NOT_FOUND"):
            LocalAgent(model_name="qwen3:8b")

    @patch("ai_rag.core.local_agent.requests.get")
    def test_init_service_unreachable(self, mock_get):
        """Ollama 未启动 → SERVICE_UNREACHABLE"""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError()

        with pytest.raises(LocalAgentError, match="SERVICE_UNREACHABLE"):
            LocalAgent(model_name="qwen3:8b")


# ─── chat() 正常流程测试 ────────────────────────────────────

class TestChatSuccess:
    """正常对话 & 流式响应解析"""

    @patch("ai_rag.core.local_agent.requests.post")
    def test_chat_returns_complete_response(self, mock_post, agent):
        """多 chunk 流式响应正确拼接"""
        chunks = [
            {"response": "你好", "done": False},
            {"response": "世界", "done": False},
            {"response": "", "done": True, "prompt_eval_count": 10, "eval_count": 5},
        ]
        mock_post.return_value = _make_stream_response(chunks)

        result = agent.chat("测试提示词")

        assert result["status"] == "success"
        assert result["response"] == "你好世界"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["latency_ms"] >= 0

    @patch("ai_rag.core.local_agent.requests.post")
    def test_chat_payload_contains_keep_alive(self, mock_post, agent):
        """验证请求体中包含 keep_alive 参数"""
        chunks = [{"response": "ok", "done": True, "prompt_eval_count": 1, "eval_count": 1}]
        mock_post.return_value = _make_stream_response(chunks)

        agent.chat("test")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["options"]["keep_alive"] == "-1"
        assert payload["stream"] is True


# ─── 异常处理测试 ────────────────────────────────────────────

class TestChatErrors:
    """超时、网络异常、未知异常的结构化错误码"""

    @patch("ai_rag.core.local_agent.requests.post")
    def test_timeout_raises_timeout_error(self, mock_post, agent):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()

        with pytest.raises(LocalAgentError) as exc_info:
            agent.chat("test")
        assert exc_info.value.error_code == "TIMEOUT"

    @patch("ai_rag.core.local_agent.requests.post")
    def test_network_error_raises_network_error(self, mock_post, agent):
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("connection refused")

        with pytest.raises(LocalAgentError) as exc_info:
            agent.chat("test")
        assert exc_info.value.error_code == "NETWORK_ERROR"

    @patch("ai_rag.core.local_agent.requests.post")
    def test_unknown_error_raises_unknown_error(self, mock_post, agent):
        mock_post.side_effect = RuntimeError("unexpected")

        with pytest.raises(LocalAgentError) as exc_info:
            agent.chat("test")
        assert exc_info.value.error_code == "UNKNOWN_ERROR"

    @patch("ai_rag.core.local_agent.requests.post")
    def test_http_error_raises_network_error(self, mock_post, agent):
        """HTTP 4xx/5xx 属于 RequestException 子类 → NETWORK_ERROR"""
        from requests.exceptions import HTTPError
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.side_effect = HTTPError(response=mock_resp)

        with pytest.raises(LocalAgentError) as exc_info:
            agent.chat("test")
        assert exc_info.value.error_code == "NETWORK_ERROR"