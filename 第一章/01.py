import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# 1. 页面配置
st.set_page_config(page_title="我的 AI 助手", page_icon="🤖",layout="wide",
                   initial_sidebar_state="expanded",menu_items={})

st.title("🤖 阿里云百炼 AI 助手")
st.logo("resources/cat.png")

# 2. 加载环境变量并初始化客户端
load_dotenv()
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
except Exception as e:
    st.error(f"客户端初始化失败: {e}")
    st.stop()

# 3. 初始化聊天记录（核心：使用 session_state 记住历史）
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# 4. 在网页上显示历史聊天记录
for message in st.session_state.messages:
    if message["role"] != "system":  # 不显示系统提示词
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 5. 创建聊天输入框（支持回车键发送！）
if prompt := st.chat_input("请输入你的问题，按回车发送..."):
    # 5.1 将用户输入添加到历史记录并在网页显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5.2 调用大模型获取回复
   # 5.2 调用大模型获取回复（流式输出）
    with st.chat_message("assistant"):
        try:
            # 【关键改动 1】开启流式模式 stream=True
            stream = client.chat.completions.create(
                model="qwen-plus", 
                messages=st.session_state.messages,
                stream=True,  # 开启流式输出
            )
            
            # 【关键改动 2】使用 st.write_stream 逐步渲染回复
            # st.write_stream 会自动处理流数据，并返回完整的回复文本
            answer = st.write_stream(stream)
            
            # 5.3 将完整的回复加入历史记录
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"调用出错: {e}")
    