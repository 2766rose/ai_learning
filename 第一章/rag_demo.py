# --- 这是一个纯模拟的 RAG 流程，无需任何外部库 ---

# 1. 数据层 (Data Layer): 我们的“知识库”
# 想象这是你上传的文档被切分成的几个段落
knowledge_base = [
    "RAG技术通过检索外部知识库来增强大模型的生成能力。",
    "向量数据库是RAG系统的核心组件，用于存储和检索文本向量。",
    "大语言模型（LLM）是RAG流程中的生成器，负责根据检索到的信息生成回答。",
    "Hugging Face是一个开源的AI模型和数据集社区。",
    "Python的sentence-transformers库可以将文本转换为向量。"
]

print("✅ 知识库已加载！")
print(f"当前知识库包含 {len(knowledge_base)} 个段落。\n")

# 2. 检索层 (Retrieval Layer): 模拟“查找”过程
def simple_retrieve(question, documents, top_k=2):
    """
    一个简单的检索函数，通过计算问题与文档的关键词重合度来排序。
    """
    # 将问题和文档都转换成小写，并按空格/标点分割成词语集合
    question_words = set(question.lower().replace('，', ' ').replace('。', ' ').split())
    
    scored_docs = []
    for doc in documents:
        doc_words = set(doc.lower().replace('，', ' ').replace('。', ' ').split())
        # 计算重合的关键词数量作为分数
        score = len(question_words.intersection(doc_words))
        scored_docs.append((doc, score))
    
    # 按分数从高到低排序，并取前 top_k 个
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

# 3. 生成层 (Generation Layer): 模拟“组装答案”过程
def rag_pipeline(question):
    print(f"🙋‍♂️ 用户提问: {question}")
    
    # --- 第一步：检索 ---
    # 从知识库中找出最相关的段落
    relevant_docs = simple_retrieve(question, knowledge_base)
    
    print("🔍 检索到的参考资料:")
    for i, doc in enumerate(relevant_docs):
        print(f"  - 资料 {i+1}: {doc}")
    
    # --- 第二步：增强 (Augmentation) ---
    # 把问题和找到的资料拼在一起，形成一个新的、信息更丰富的提示词
    context = "\n".join(relevant_docs)
    prompt = f"""
    请扮演一个严谨的AI助手，仅根据以下参考资料回答问题。
    
    【参考资料】
    {context}
    
    【问题】
    {question}
    
    【回答】
    """
    
    print("\n🤖 最终发给大模型的提示词 (Prompt):")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print("\n💡 你看，RAG的核心就是把检索到的资料（context）和问题（question）打包在一起，再交给大模型。")
    print("这样大模型就能基于事实回答，而不是瞎编了！")

# --- 开始测试 ---
rag_pipeline("RAG系统里用什么来存储信息？")