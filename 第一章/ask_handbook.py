# 1. 导入绝对安全的基础组件
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 2. 初始化大模型
llm = OllamaLLM(model="qwen3:8b", temperature=0) 

# 3. 构建检索器
vectorstore = Chroma(persist_directory="./chroma_handbook", collection_name="handbook_library")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 提取所有文档用于 BM25
# 使用 .get() 方法获取数据库中存储的所有文档
all_data = vectorstore.get()
docs = [Document(page_content=content, metadata=metadata) for content, metadata in zip(all_data['documents'], all_data['metadatas'])]
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# 4. 【核心改变】手写混合检索函数，彻底抛弃 EnsembleRetriever
def ensemble_retrieve(query: str):
    # 让两个检索器分别去找
    bm25_docs = bm25_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    
    # 简单粗暴地合并并去重
    seen_contents = set()
    unique_docs = []
    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
            
    return unique_docs[:5]

# 5. 定义企业级防幻觉 Prompt
prompt = ChatPromptTemplate.from_template("""
你是一个严谨的星云科技 HR 助手。
【最高指令】：只能根据【参考资料】回答，严禁编造任何事实或数字！
【兜底策略】：如果资料中没有答案，请直接回答：“抱歉，员工手册中未找到相关规定，请咨询行政部。”

参考资料：
{context}

员工问题：{input}

HR 助手回答：
""")

# 6. 格式化检索结果
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. 构建 LCEL 检索问答链（使用我们手写的函数）
rag_chain = (
    {"context": ensemble_retrieve | format_docs, "input": RunnablePassthrough()}

    | prompt
    | llm
    | StrOutputParser()
)

# 8. 提问并打印结果
if __name__ == "__main__":
    question = "我去上海出差，住宿标准是多少？如果迟到了怎么扣钱？"
    
    print("🔍 混合检索找到的参考资料：")
    for i, doc in enumerate(ensemble_retrieve(question)):
        print(f"[{i+1}] {doc.page_content}")
    print("-" * 40)
    
    print("🤖 Qwen 的回答：")
    print(rag_chain.invoke(question))