from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

# 1. 连接我们刚刚建好的本地“图书馆”
# persist_directory 必须和建库时保持一致
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="my_first_library"
)

# 2. 初始化你的大模型（图书管理员）
llm = OllamaLLM(model="qwen3:8b")

# 3. 模拟提问
question = "RAG 技术是用来解决什么问题的？"

# 4. 让图书馆去检索最相关的文本块
# k=1 表示只找最相似的 1 个文本块
docs = vectorstore.similarity_search(question, k=1)

# 5. 打印看看，图书管理员找到了哪本书？
print("🔍 检索到的参考资料：")
print(docs[0].page_content)
print("-" * 30)

# 6. 把找到的资料和你的问题，拼成一段提示词（Prompt）喂给大模型
prompt = f"""请根据以下参考资料回答问题。如果资料中没有答案，请直接说不知道。

参考资料：
{docs[0].page_content}

问题：{question}

回答："""

# 7. 让大模型生成最终答案
answer = llm.invoke(prompt)
print("🤖 Qwen 的回答：")
print(answer)