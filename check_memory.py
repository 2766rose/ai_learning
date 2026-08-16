import chromadb
from ai_rag.core.config import rag_config

client = chromadb.PersistentClient(path=rag_config.MEMORY_CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(name=rag_config.MEMORY_COLLECTION_NAME)

results = collection.get(where={"user_id": "emp_zhangsan_001"}, include=["documents", "metadatas"])
print(f"📊 用户 emp_zhangsan_001 共有 {len(results['ids'])} 条记忆:")
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"  - {doc} | meta={meta}")
