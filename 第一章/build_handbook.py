from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 1. 加载刚才创建的测试文档
loader = TextLoader("./employee_handbook.txt", encoding="utf-8")
documents = loader.load()
print(f"📄 成功加载《星云科技员工手册》，共 {len(documents)} 页")

# 2. 智能分块（企业级推荐参数）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 每块 300 字符，保证语义完整
    chunk_overlap=50     # 重叠 50 字符，防止上下文断裂
)
split_docs = text_splitter.split_documents(documents)
print(f"🔪 成功切分为 {len(split_docs)} 个文本块")

# 3. 存入向量数据库
vectorstore = Chroma.from_documents(
    documents=split_docs,
    collection_name="handbook_library",
    persist_directory="./chroma_handbook"  # 专属文件夹
)

print("🎉 员工手册知识库构建成功！")