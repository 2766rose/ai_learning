# 第 8-11 周总结报告（2026-08-15）

## 第 8 周：LoRA 微调（低成本训练行业专属模型）
- 环境：CUDA torch 2.13.0+cu126、RTX 4060 8GB、bitsandbytes 4bit QLoRA
- 模型：qwen2.5-3b-instruct；LoRA r=8，可训练参数 15M（0.48%）
- 数据：员工手册制度问答 35 → 146 条（qwen3 同义问法 + 章节抽取，质量抽查全对）
- 训练：3 epoch，train loss 2.26→0.64，约 4 分钟
- 结论：数据量是微调效果分水岭（35 条 1/4 正确 → 146 条 3/4 正确）

## 第 9 周：量化 + RAGAS 评估
- 量化：LoRA 合并 → GGUF q8_0（INT8，3.06GB）→ Ollama 模型 qwen2.5-3b-ft
- RAGAS 评估（20 题，评判=qwen2.5:7b）：
  | 系统 | answer_correctness | faithfulness |
  | --- | --- | --- |
  | 微调 3B（直接回答） | 0.870 | - |
  | qwen3:8b + RAG | 0.798 | 0.813 |
  | 微调 3B + RAG | 0.760 | 0.762 |
- 结论：小模型微调直接答最优；大模型配 RAG 更稳；小模型硬塞 RAG 反而变差
- 最终配置：RAG_OPENAI_MODEL=qwen3:8b（RAG 场景）

## 第 10 周：Langfuse 可观测性（白盒化）
- 部署：Langfuse 云端免费版，密钥在 .env（LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL）
- 接入：请求级 trace（rag-chat/rag-chat-stream）+ 每次 LLM 调用的 generation（llm_call，含 Prompt/模型/耗时/Token）
- 安全：失败静默降级，绝不影响业务
- 验证：控制台可见完整链路，能定位"哪次调用慢"

## 第 11 周：语义缓存 + 限流 + 熔断
- 语义缓存（core/semantic_cache.py）：向量相似度阈值 0.92、500 条、TTL 1h；重复问题 10s→0.06s
- 用户限流（core/rate_limiter.py）：20 次/分钟，超限 429（实测第 4 次 429）
- 熔断（core/circuit_breaker.py）：连续失败 3 次 → 冷却 30s 快速拒绝（实测第 4 次秒回"熔断中"）
- 说明：均为进程内内存版；多进程/多机需升级 Redis 版

## 资产清单
- LoRA：models/qwen2.5-3b-lora；合并模型：models/qwen2.5-3b-merged
- Ollama：qwen2.5-3b-ft（q8_0）；数据集：data/staff_qa_dataset.json（146 条）
- 可观测：core/observability.py；缓存/限流/熔断：core/{semantic_cache,rate_limiter,circuit_breaker}.py
- 备份：work/ 下 *.bak / *.week10 / *.week11
