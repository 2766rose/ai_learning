# 第2周实验：Recursive vs Semantic 切分器检索对比

## 实验设置
- 语料：员工手册 + 费用报销管理制度PDF（含复杂表格）+ 简历PDF
- 参数：chunk_size=500, overlap=50, top_k=5
- 测试问题数：15，命中判定 = 检索块包含答案关键词

## 结果
| 切分器 | 块数 | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| recursive | 9 | 0.60 | 1.00 | 0.767 |
| semantic | 31 | 0.73 | 0.87 | 0.800 |

## 结论
1. SemanticChunker 的首位命中与 MRR 更高（Hit@1 0.73 vs 0.60，MRR 0.800 vs 0.767），因为块更贴合语义单元。
2. 语义切分块数更多（31 vs 9），导致 top-5 召回略降（0.87 vs 1.00）；可通过调低 breakpoint_percentile（默认 0.75）减少切分点。
3. 配套成果：PDF 解析器升级（pymupdf 表格→Markdown，合并单元格前向填充），入库流程支持 RAG_SPLITTER=recursive|semantic 切换。
