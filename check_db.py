# check_db.py
import chromadb
from chromadb.config import Settings

print("正在尝试连接本地 ChromaDB 数据库...")

try:
    # 核心修改：使用 PersistentClient 直接读取本地文件夹
    # 路径 './chroma_db' 必须和 docker-compose.yml 里 volumes 配置的宿主机路径一致
    client = chromadb.PersistentClient(path="./chroma_db")

    # 替换成你代码里实际使用的集合名称
    collection_name = "company_knowledge" 
    
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    
    print(f"✅ 数据库连接成功！")
    print(f"📊 集合 '{collection_name}' 中当前共有 {count} 个文档片段。")
    
    if count > 0:
        # 随便 peek 一条看看内容
        results = collection.peek() 
        print("\n--- 最新存入的内容预览 ---")
        # 只打印前100个字符，避免刷屏
        print(results['documents'][0][:100] + "...") 

except Exception as e:
    print(f"❌ 查询失败: {e}")
    print("💡 提示：请检查 './chroma_db' 路径是否正确，以及集合名称是否匹配。")