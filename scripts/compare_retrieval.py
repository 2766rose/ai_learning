# scripts/compare_retrieval.py
"""第3周实验：纯向量 vs 混合(BM25+向量+RRF) vs 混合+BGE-Reranker 检索质量对比
用法：python scripts/compare_retrieval.py
"""
import os, sys, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_rag.core.embeddings import embedding_service
from ai_rag.parsers.pdf_parser import PDFParser
from ai_rag.retrieval.bm25_engine import BM25Engine
from ai_rag.retrieval.hybrid_retriever import HybridRetriever
from ai_rag.retrieval.config import RetrievalConfig
from ai_rag.retrieval.reranker import reranker

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
DOCS = [
    ("员工手册", r"D:\ai_learning\handbook.txt", "text"),
    ("报销制度PDF", r"D:\ai_learning\scripts\test_pdfs\费用报销管理制度.pdf", "pdf"),
    ("简历PDF", r"D:\ai_learning\data\uploads\1be0c8da-8ce2-4c8b-b81c-c059c0284bbd.pdf", "pdf"),
]
QUESTIONS = [
    ("员工的试用期是多长时间？", ["3个月试用期"]),
    ("转正考核需要提交什么材料？", ["试用期工作总结"]),
    ("入职满一年不满十年有多少天带薪年假？", ["5天"]),
    ("年假需要提前几天在系统申请？", ["3个工作日"]),
    ("病假需要提供什么？", ["医院证明"]),
    ("每月发薪日是几号？", ["10日"]),
    ("公积金缴纳比例是多少？", ["12%"]),
    ("每日餐补多少钱？", ["30元"]),
    ("高铁出差的报销标准是什么？", ["二等座"]),
    ("一线城市住宿每晚标准是多少？", ["500"]),
    ("二线城市住宿每晚标准是多少？", ["350"]),
    ("市内交通每天上限多少？", ["200"]),
    ("招待费单次超过多少需要VP审批？", ["1000"]),
    ("出差发票需要在多少天内提交？", ["30天"]),
    ("迟到超过两小时怎么处理？", ["旷工"]),
]


class EphemeralVectorStore:
    """把临时 Chroma 集合包装成 HybridRetriever 需要的 async search 接口"""

    def __init__(self, col, count):
        self._col = col
        self._count = count

    async def search(self, query, top_k=20, where=None):
        k = max(1, min(top_k, self._count))
        qe = embedding_service.embed_query(query)
        res = self._col.query(query_embeddings=[qe], n_results=k, include=["documents", "metadatas", "distances"])
        hits = []
        ids = (res.get("ids") or [[]])[0]
        for i in range(len(ids)):
            hits.append({
                "id": ids[i],
                "document": (res["documents"][0])[i],
                "metadata": (res["metadatas"][0])[i],
                "distance": (res["distances"][0])[i],
            })
        return hits


def load_docs():
    out = []
    for name, path, kind in DOCS:
        if kind == "text":
            with open(path, "r", encoding="utf-8") as f:
                out.append((name, f.read()))
        else:
            out.append((name, PDFParser().parse_file(path)))
    return out


def build_corpus(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    col = client.create_collection("cmp", metadata={"hnsw:space": "cosine"})
    chunks = []
    for name, text in docs:
        for c in splitter.split_text(text):
            chunks.append({"content": c, "source": name})
    ids = [f"d{i}" for i in range(len(chunks))]
    emb = embedding_service.embed_texts([c["content"] for c in chunks])
    col.add(
        documents=[c["content"] for c in chunks],
        embeddings=emb,
        metadatas=[{"source": c["source"]} for c in chunks],
        ids=ids,
    )
    bm25 = BM25Engine()
    bm25.index([
        {"content": c["content"], "id": ids[i], "metadata": {"source": c["source"]}}
        for i, c in enumerate(chunks)
    ])
    return col, bm25, len(chunks)


def is_hit(doc_text, golds):
    return any(g in doc_text for g in golds)


async def eval_mode(retriever_fn):
    """retriever_fn(query) -> 异步返回 [(doc_text, score)] 前 TOP_K 个"""
    hits1 = 0
    hits5 = 0
    mrr = 0.0
    for q, golds in QUESTIONS:
        ranked = await retriever_fn(q)
        rank = None
        for i, (doc_text, _score) in enumerate(ranked[:TOP_K]):
            if is_hit(doc_text, golds):
                rank = i + 1
                break
        if rank == 1:
            hits1 += 1
        if rank is not None:
            hits5 += 1
            mrr += 1.0 / rank
    n = len(QUESTIONS)
    return {"hit@1": hits1 / n, "hit@5": hits5 / n, "mrr": mrr / n}


async def main():
    t0 = time.time()
    print("构建语料（员工手册 + 报销制度PDF + 简历PDF）...", flush=True)
    docs = load_docs()
    col, bm25, count = build_corpus(docs)
    print(f"chunks={count}", flush=True)
    vs = EphemeralVectorStore(col, count)
    cfg = RetrievalConfig()

    # 1) 纯向量
    async def vector_fn(q):
        qe = embedding_service.embed_query(q)
        res = col.query(query_embeddings=[qe], n_results=TOP_K, include=["documents", "distances"])
        return [(d, 1.0 - dist) for d, dist in zip(res["documents"][0], res["distances"][0])]

    # 2) 混合（BM25 + 向量 + RRF）
    async def hybrid_fn(q):
        hr = HybridRetriever(bm25, vs, cfg)
        got = await hr.retrieve(q)
        return [(d["content"], float(d["score"])) for d in got[:TOP_K]]

    # 3) 混合 + BGE-Reranker（对 top20 精排取前5）
    async def hybrid_rerank_fn(q):
        hr = HybridRetriever(bm25, vs, cfg)
        got = await hr.retrieve(q)
        cands = [
            {"id": d["id"], "document": d["content"], "metadata": d.get("metadata", {}), "score": float(d["score"])}
            for d in got[: cfg.final_top_k]
        ]
        top = await reranker.rerank(q, cands, top_k=TOP_K)
        return [(c["document"], float(c["score"])) for c in top]

    results = {}
    for name, fn in [("vector", vector_fn), ("hybrid", hybrid_fn), ("hybrid+rerank", hybrid_rerank_fn)]:
        m = await eval_mode(fn)
        results[name] = m
        print(f"[{name}] Hit@1={m['hit@1']:.2f} Hit@5={m['hit@5']:.2f} MRR={m['mrr']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    lines = []
    lines.append("# 第3周实验：纯向量 vs 混合检索 vs 混合+Reranker 检索质量对比\n")
    lines.append(f"- 语料：{', '.join(n for n, _, _ in DOCS)}（chunks={count}）")
    lines.append(f"- 参数：chunk_size={CHUNK_SIZE}, top_k={TOP_K}, RRF k={cfg.rrf_k}, 精排候选={cfg.final_top_k}")
    lines.append(f"- 测试问题数：{len(QUESTIONS)}，命中判定：检索块包含答案关键词\n")
    lines.append("| 检索方式 | Hit@1 | Hit@5 | MRR |")
    lines.append("| --- | --- | --- | --- |")
    for name in ("vector", "hybrid", "hybrid+rerank"):
        m = results[name]
        lines.append(f"| {name} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['mrr']:.3f} |")
    report = "\n".join(lines)
    out = r"D:\ai_learning\docs\week3_retrieval_comparison.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n报告已保存:", out)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
