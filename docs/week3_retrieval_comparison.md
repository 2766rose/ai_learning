# 第3周实验：纯向量 vs 混合检索 vs 混合+Reranker 检索质量对比

## 实验设置
- 语料：员工手册 + 费用报销管理制度PDF（含复杂表格）+ 简历PDF
- 切分方式：RecursiveCharacterTextSplitter 与 SemanticChunker（均 chunk_size=500/overlap=50）
- 测试问题：15 个，命中判定 = 检索块包含答案关键词
- 指标：Hit@1 / Hit@5 / MRR（top_k=5）

## 结果（Recursive 切分，9 块）
| 检索方式 | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- |
| vector（纯向量） | 0.60 | 1.00 | 0.767 |
| hybrid（BM25+向量+RRF） | 0.93 | 1.00 | 0.967 |
| hybrid + BGE-Reranker | 0.93 | 1.00 | 0.967 |

## 结果（Semantic 切分，31 块，噪声更大）
| 检索方式 | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- |
| vector（纯向量） | 0.73 | 0.87 | 0.800 |
| hybrid（BM25+向量+RRF） | 0.87 | 1.00 | 0.933 |
| hybrid + BGE-Reranker | 0.93 | 1.00 | 0.967 |

## 结论
1. 混合检索（BM25 + 向量 + RRF）相比纯向量提升显著：9 块语料 Hit@1 0.60→0.93、MRR 0.767→0.967；31 块语料 Hit@1 0.73→0.87。
2. BGE-Reranker 在候选集小（9 块，混合已把正确块排第 1）时无额外收益；候选集增大（31 块）后，精排把 Hit@1 从 0.87 提到 0.93、MRR 0.933→0.967。
3. 生产建议：知识库块数越多、噪声越大，rerank 越值得开启；块数很少时仅混合检索即可。
4. 已接入生产链路：knowledge_search 默认走混合检索（RAG_RETRIEVAL=hybrid），RAG_RERANK=on 时叠加精排（模型：models/bge-reranker-base，本地加载）。
