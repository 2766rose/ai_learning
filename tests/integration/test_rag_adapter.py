# tests/integration/test_rag_adapter.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from ai_rag.models.schemas import ChatRequest
from ai_rag.api.rag_router import _unified_chat, chat
from ai_rag.core.local_agent import LocalAgent
from fastapi import Request, HTTPException


@pytest.mark.asyncio
async def test_local_agent_returns_answer():
    """本地模型正常返回时，应直接使用本地结果"""
    mock_client = MagicMock(spec=LocalAgent)
    mock_client.chat.return_value = {"response": "本地回答"}

    messages = [MagicMock(role="user", content="你好")]
    result = await _unified_chat(mock_client, "sess-001", messages, stream=False)

    assert result == ("本地回答", "")
    mock_client.chat.assert_called_once()


@pytest.mark.asyncio
async def test_local_fallback_to_cloud():
    """本地模型抛异常时，chat 端点应自动降级到云端"""
    request = ChatRequest(
        session_id="sess-fb",
        messages=[{"role": "user", "content": "测试降级"}]
    )
    raw_request = MagicMock(spec=Request)

    local_mock = MagicMock(spec=LocalAgent)
    cloud_mock = MagicMock()  # 类型不重要，只要不是 LocalAgent 即可

    # ✅ 核心修正：直接 patch _unified_chat，精确控制两次调用的行为
    # 第一次调用（local）抛异常，第二次调用（cloud fallback）返回正常结果
    with patch("ai_rag.api.rag_router._unified_chat", side_effect=[
        RuntimeError("Ollama down"),       # 第一次：本地失败
        ("云端兜底回答", ""),               # 第二次：云端兜底成功
    ]):
        with patch("ai_rag.api.rag_router.get_client", side_effect=[local_mock, cloud_mock]):
            with patch("ai_rag.api.rag_router.route", return_value="local"):
                response = await chat(request, raw_request)

    assert response.ai_answer == "云端兜底回答"
    assert response.session_id == "sess-fb"


@pytest.mark.asyncio
async def test_no_user_message_raises_400():
    """messages 中无 user 消息时，chat 端点应返回 400"""
    request = ChatRequest(
        session_id="sess-empty",
        messages=[{"role": "assistant", "content": "只有助手消息"}]
    )
    raw_request = MagicMock(spec=Request)

    with pytest.raises(HTTPException) as exc_info:
        await chat(request, raw_request)

    assert exc_info.value.status_code == 400