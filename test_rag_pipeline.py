# test_rag_pipeline.py
import time
import asyncio
import logging
from typing import List, Dict, Any

from ai_rag.core.qwen_provider import QwenProvider, LLMResponse
from ai_rag.retrieval import HybridRetriever, RetrievalConfig, BM25Engine
from ai_rag.core.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def warm_up_bm25(bm25_engine: BM25Engine, vector_store: VectorStore) -> None:
    logger.info("🔄 正在从 ChromaDB 加载语料到 BM25 内存索引...")
    try:
        # ✅ 先确保 VectorStore 已完成异步初始化
        await vector_store.initialize()

        # ✅ 使用公开 API，不再反射内部属性
        documents = await vector_store.get_all_documents()

        if not documents:
            logger.warning("⚠️ ChromaDB 中无数据，BM25 跳过加载")
            return

        bm25_engine.index(documents)
        logger.info(f"✅ BM25 索引预热完成: {len(documents)} chunks")
    except Exception as e:
        logger.error(f"❌ BM25 预热失败: {e}", exc_info=True)

#  新增：纯检索性能基准测试函数 
async def benchmark_retrieval(retriever: HybridRetriever, query: str) -> dict:
    """测量纯检索耗时（不含LLM）"""
    start = time.perf_counter()
    results = await retriever.retrieve(query=query)
    latency_ms = (time.perf_counter() - start) * 1000
    
    bm25_hits = 0
    for r in results:
        if isinstance(r, dict):
            score = r.get("bm25_score", 0) or r.get("score", 0)
        else:
            score = getattr(r, "bm25_score", 0)
        if score and score > 0:
            bm25_hits += 1
            
    return {
        "latency_ms": round(latency_ms, 2),
        "total_hits": len(results),
        "bm25_hits": bm25_hits
    }

async def main():
    query = "Qwen-Plus 有什么特点？回答时需要注意什么？"

    # ================= 1. 初始化检索组件 =================
    logger.info("⚙️ 正在初始化混合检索组件...")
    config = RetrievalConfig()
    bm25 = BM25Engine()
    vector_store = VectorStore()

    await warm_up_bm25(bm25, vector_store)

    retriever = HybridRetriever(
        bm25_engine=bm25,
        vector_store=vector_store,
        config=config
    )
    logger.info("✅ HybridRetriever 初始化完成")

    
    logger.info("⏱️ 开始纯检索性能基线测试...")
    metrics = await benchmark_retrieval(retriever, query)
    logger.info(f"📊 纯检索基线: 耗时={metrics['latency_ms']}ms | 总召回={metrics['total_hits']} | BM25命中={metrics['bm25_hits']}")


    # ================= 2. 执行混合检索 =================
    logger.info(f"🔍 开始混合检索 | Query: {query}")
    retrieval_results = await retriever.retrieve(query=query)

    # ✅ 修复2: 兼容 dict 和 RetrievedDoc 两种返回格式
    context_docs: List[str] = []
    for r in retrieval_results:
        if isinstance(r, dict):
            content = r.get("content") or r.get("document") or r.get("text", "")
        else:
            content = getattr(r, "content", "") or getattr(r, "document", "")
        if content:
            context_docs.append(content)

    logger.info(f"📚 检索完成 | 获取到 {len(context_docs)} 条上下文")

    if context_docs:
        for i, doc in enumerate(context_docs):
            preview = doc[:300].replace("\n", " ")
            logger.info(f"  DOC[{i}]: {preview}...")
    else:
        logger.warning("⚠️ 未检索到任何相关文档，将仅依赖 LLM 内部知识回答")

    # ✅ 修复3: 兼容 dict 格式的 BM25 召回检测
    bm25_hits = 0
    for r in retrieval_results:
        if isinstance(r, dict):
            score = r.get("bm25_score", 0) or r.get("score", 0)
        else:
            score = getattr(r, "bm25_score", 0)
        if score and score > 0:
            bm25_hits += 1

    if len(retrieval_results) > 0 and bm25_hits == 0:
        logger.warning("⚠️ BM25 召回数为 0，请检查 BM25Engine 是否已正确加载语料/索引")
    else:
        logger.info(f"📊 检索统计: BM25 命中 {bm25_hits} 条, 总融合结果 {len(retrieval_results)} 条")

    # ================= 3. RAG 生成 =================
    provider = QwenProvider()
    logger.info("🚀 开始 RAG 生成...")
    result: LLMResponse = await provider.generate(query=query, context=context_docs)

    # ================= 4. 输出与断言 =================
    print("\n" + "=" * 60)
    print(f"📝 Answer:\n{result.answer}")
    print(f"\n🔗 Sources: {result.sources}")
    print(f"⏱️  Latency: {result.latency_ms} ms")
    print(f"🔢 Tokens: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
    print(f"🤖 Model: {result.model_version}")
    print("=" * 60)

    assert isinstance(result, LLMResponse), "返回值必须是 LLMResponse 类型"
    assert len(result.answer) > 0, "Answer 不能为空"
    assert result.latency_ms > 0, "Latency 必须大于 0"
    logger.info("✅ 全链路断言通过！RAG Pipeline 集成测试成功。")


if __name__ == "__main__":
    asyncio.run(main())