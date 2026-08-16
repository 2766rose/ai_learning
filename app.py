import streamlit as st
import requests
import json
from config import rag_config

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式（企业级UI优化）====================
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    div[data-testid="stChatMessage"] {margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# ==================== 从统一配置读取（消除硬编码）====================
STREAM_ENDPOINT = f"{rag_config.BACKEND_API_URL.rstrip('/')}/api/v1/chat/rag/stream"
# 💡 流式传输耗时较长，动态超时增加基础缓冲
max_tokens = getattr(rag_config, "LLM_MAX_TOKENS", 2048)
REQUEST_TIMEOUT = max(120, int(max_tokens / 50) + 60)

# ==================== Session State 初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🏢 知识库问答")
    st.divider()

    source_filter = st.text_input(
        "📁 文档来源过滤",
        placeholder="留空则检索全部文档",
        help="输入文件名或来源标识，仅从指定文档中检索"
    )

    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"🤖 模型: {rag_config.OPENAI_MODEL}")
    st.caption("v1.0.0 | 内部使用")

# ==================== 主聊天区域 ====================
st.title("💬 智能知识问答")
st.caption("基于 RAG 的企业知识库检索增强生成系统")

# 空状态引导
if not st.session_state.messages:
    st.info("👋 您好！请在下方输入问题，或在左侧筛选特定文档来源。")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("citation"):
                with st.expander("📚 查看引用来源", expanded=False):
                    # 💡 使用 st.text 防止知识库原文 Markdown 污染 UI
                    st.text(msg["citation"])
            if msg.get("trace_id"):
                st.caption(f"🔗 Trace: `{msg['trace_id']}`")

# ==================== 用户输入 & 流式响应 ====================
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ✅ 核心修复：严格对齐后端 ChatRequest Schema
    # 后端通过 request.messages[-1].content 提取问题，可选过滤字段名为 source
    request_body = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # 💡 仅当用户实际输入了过滤条件时才传递，避免传 None 触发 422
    if source_filter and source_filter.strip():
        request_body["source"] = source_filter.strip()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        citation_data = None
        trace_id = None
        error_occurred = False

        try:
            response = requests.post(
                STREAM_ENDPOINT,
                json=request_body,
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = payload.get("type")

                if event_type == "citation":
                    citation_data = payload.get("data", "")
                elif event_type == "chunk":
                    full_answer += payload.get("data", "")
                    placeholder.markdown(full_answer + "▌")
                elif event_type == "done":
                    placeholder.markdown(full_answer)
                    trace_id = payload.get("trace_id")
                elif event_type == "error":
                    error_msg = payload.get("message", "未知错误")
                    placeholder.error(f"❌ 服务异常: {error_msg}")
                    error_occurred = True
                    break

        except requests.exceptions.ConnectionError:
            placeholder.error("❌ 无法连接到后端服务，请检查服务是否启动")
            error_occurred = True
        except requests.exceptions.Timeout:
            placeholder.error(f"❌ 请求超时（>{REQUEST_TIMEOUT}s），请稍后重试")
            error_occurred = True
        except Exception as e:
            placeholder.error(f"❌ 请求异常: {str(e)}")
            error_occurred = True

    if not error_occurred:
        assistant_msg = {
            "role": "assistant",
            "content": full_answer,
            "citation": citation_data,
            "trace_id": trace_id
        }
        st.session_state.messages.append(assistant_msg)