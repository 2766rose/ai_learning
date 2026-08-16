# check.py - 仅用于验证知识库是否能被检索到
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔄 正在加载文档...")
try:
    docs = TextLoader("handbook.txt", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)
    print(f"✅ 文档切分完成，共 {len(chunks)} 个片段")
except Exception as e:
    print(f"❌ 文档加载失败: {e}")
    exit()

print("🔄 正在加载 Embedding 模型并建立索引（首次可能较慢）...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    db = FAISS.from_documents(chunks, embeddings)
    print("✅ 向量索引建立完成")
except Exception as e:
    print(f"❌ 模型加载或索引失败: {e}")
    exit()

print("🔄 正在执行检索测试...")
query = "入职满10年有几天年假"
results = db.similarity_search(query, k=1)

if results and results[0].page_content.strip():
    print(f"🎉 检索成功！找到内容: {results[0].page_content}")
else:
    print("❌ 检索失败！知识库中没有匹配内容")