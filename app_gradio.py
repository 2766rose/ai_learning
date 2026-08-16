import gradio as gr
import httpx
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/upload"
RAG_CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat/rag"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def upload_file(file):
    if file is None:
        return "❌ 请先选择一个文件"
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return f"❌ 不支持该格式: {file_ext}"
    try:
        with open(file.name, "rb") as f:
            files = {"file": (os.path.basename(file.name), f, "application/octet-stream")}
            response = httpx.post(UPLOAD_ENDPOINT, files=files, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return f"✅ 上传成功！任务ID: {data.get('task_id', '未知')}\n后台解析中..."
        else:
            return f"❌ 上传失败: {response.text}"
    except Exception as e:
        logger.error(f"上传异常: {e}")
        return f"❌ 发生错误: {str(e)}"

def chat_with_ai(message, history):
    if not message.strip():
        return history

    ai_reply = ""
    try:
        # 【关键修复】：后端要求 messages 字段，不是 query 字段！
        payload = {
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        response = httpx.post(RAG_CHAT_ENDPOINT, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # 后端返回的可能是 ai_answer, answer, 或 content
            ai_reply = (
                result.get("ai_answer") 
                or result.get("answer") 
                or result.get("content") 
                or result.get("response") 
                or "⚠️ 未获取到有效回答"
            )
        elif response.status_code == 422:
            logger.error(f"422 详情: {response.text}")
            ai_reply = f"❌ 后端参数错误(422)：请检查后端接口定义。当前发送: {payload}"
        else:
            ai_reply = f"❌ 服务器错误: {response.status_code}"
    except Exception as e:
        ai_reply = f"❌ 连接失败: {str(e)}"

    # 保持字典格式写入聊天框（兼容 Gradio 校验）
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ai_reply})
    
    return history

# --- 界面构建 ---
with gr.Blocks(title="星云科技 AI 助手") as demo:
    gr.Markdown("# ☁️ 星云科技 AI 智能助手")
    gr.Markdown("基于 RAG 架构的企业知识库问答系统")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传知识文档 (PDF/TXT/DOCX)")
            upload_btn = gr.Button("📤 上传并解析", variant="primary")
            upload_status = gr.Textbox(label="上传状态", lines=3, interactive=False)
        with gr.Column(scale=2):
            # 【关键】：不传 type 参数，让 UI 保持默认
            # 但我们在 chat_with_ai 中强制写入字典格式
            chatbot = gr.Chatbot(label="对话记录", height=400)
            msg_input = gr.Textbox(label="请输入问题", placeholder="例如：入职满10年有几天年假？")
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary", scale=2)
                clear_btn = gr.Button("清空对话", scale=1)

    upload_btn.click(upload_file, inputs=[file_input], outputs=[upload_status])
    send_btn.click(chat_with_ai, inputs=[msg_input, chatbot], outputs=[chatbot])
    msg_input.submit(chat_with_ai, inputs=[msg_input, chatbot], outputs=[chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)