# scripts/compare_splitters.py
"""RecursiveCharacterTextSplitter vs SemanticChunker 检索效果对比
用法（在项目虚拟环境中）:
    python scripts/compare_splitters.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_rag.core.embeddings import embedding_service
from ai_rag.parsers.pdf_parser import PDFParser
from ai_rag.rag.splitters.semantic_chunker import SemanticChunker

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

def load_docs():
    texts = []
    for name, path, kind in DOCS:
        if kind == "text":
            with open(path, "r", encoding="utf-8") as f:
                texts.append((name, f.read()))
        else:
            texts.append((name, PDFParser().parse_file(path)))
    return texts

def build_collection(splitter_name, splitter, docs):
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    col = client.create_collection(f"test_{splitter_name}", metadata={"hnsw:space": "cosine"})
    all_chunks, all_metas = [], []
    for name, text in docs:
        chunks = splitter.split_text(text)
        for c in chunks:
            all_chunks.append(c)
            all_metas.append({"source": name})
    emb = embedding_service.embed_texts(all_chunks)
    ids = [f"{splitter_name}-{i}" for i in range(len(all_chunks))]
    col.add(documents=all_chunks, embeddings=emb, metadatas=all_metas, ids=ids)
    return col, all_chunks

def evaluate(col, all_chunks):
    hits_at_1 = 0
    hits_at_5 = 0
    mrr_sum = 0.0
    for q, golds in QUESTIONS:
        qe = embedding_service.embed_query(q)
        res = col.query(query_embeddings=[qe], n_results=TOP_K, include=["documents", "distances"])
        docs = (res.get("documents") or [[]])[0]
        rank = None
        for i, d in enumerate(docs):
            if any(g in d for g in golds):
                rank = i + 1
                break
        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_5 += 1
            mrr_sum += 1.0 / rank
    n = len(QUESTIONS)
    return {
        "questions": n,
        "hit@1": hits_at_1 / n,
        "hit@5": hits_at_5 / n,
        "mrr": mrr_sum / n,
    }

def main():
    print("加载文档...")
    docs = load_docs()
    print(f"语料: {[n for n,_ in docs]}")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    semantic = SemanticChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    results = {}
    for name, splitter in [("recursive", recursive), ("semantic", semantic)]:
        t0 = time.time()
        col, chunks = build_collection(name, splitter, docs)
        metrics = evaluate(col, chunks)
        metrics["chunks"] = len(chunks)
        results[name] = metrics
        print(f"\n[{name}] chunks={len(chunks)} 耗时={time.time()-t0:.1f}s")
        print(f"  Hit@1={metrics['hit@1']:.2f}  Hit@5={metrics['hit@5']:.2f}  MRR={metrics['mrr']:.3f}")

    # 汇总输出
    lines = []
    lines.append("# Recursive vs Semantic 切分器检索对比\n")
    lines.append(f"- 语料：{', '.join(n for n,_ in docs)}")
    lines.append(f"- 参数：chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, top_k={TOP_K}")
    lines.append(f"- 测试问题数：{len(QUESTIONS)}\n")
    lines.append("| 切分器 | 块数 | Hit@1 | Hit@5 | MRR |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in ("recursive", "semantic"):
        m = results[name]
        lines.append(f"| {name} | {m['chunks']} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['mrr']:.3f} |")
    report = "\n".join(lines)
    out = r"D:\ai_learning\docs\week2_splitter_comparison.md"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n报告已保存:", out)
    print(report)

if __name__ == "__main__":
    main()
