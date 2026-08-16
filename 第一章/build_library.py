from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. 模拟一份“公司内部资料”
documents = [
    Document(page_content="Qwen3-8B 是一款由通义实验室开发的开源大语言模型，具有极高的性价比。", metadata={"source": "公司技术白皮书"}),
    Document(page_content="企业在使用大模型时，通常会采用 RAG 技术来解决大模型的幻觉问题。", metadata={"source": "AI应用开发指南"}),
    Document(page_content="LangChain 是目前最流行的 AI 应用开发框架，支持多种向量数据库。", metadata={"source": "AI应用开发指南"}),
]

# 2. 初始化文本切分器（Chunking）
# chunk_size=20 表示每 20 个字符切一刀，chunk_overlap=5 表示相邻两块之间有 5 个字符的重叠，防止语义被切断
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
split_docs = text_splitter.split_documents(documents)

# 3. 将切分后的文档存入本地的向量数据库（持久化）
# persist_directory 指定了数据保存在电脑硬盘上的哪个文件夹
vectorstore = Chroma.from_documents(
    documents=split_docs,
    collection_name="my_first_library",
    persist_directory="./chroma_db"  # 数据将保存在当前目录的 chroma_db 文件夹中
)

print("🎉 知识库构建成功！")
print(f"📚 原始文档数: {len(documents)}")
print(f"🔪 切分后的文本块数: {len(split_docs)}")