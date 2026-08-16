# 1. 引入 LangChain 提供的 Ollama 接口模块
from langchain_ollama import OllamaLLM

# 2. 初始化模型，告诉它我们要连的是本地的 qwen3:8b
llm = OllamaLLM(model="qwen3:8b")

# 3. 发送一个简单的测试问题
response = llm.invoke("你好，请用一句话介绍你自己，并告诉我你支持处理图片吗？")

# 4. 打印出 Qwen 的回答
print("Qwen 的回答：", response)