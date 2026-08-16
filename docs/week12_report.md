# 第 12 周总结报告（2026-08-15）

## 第 12 周：标准化交付与自动化发布（Docker + GitHub Actions）

- **Dockerfile**：多阶段构建（builder 装依赖 → slim 运行镜像），Python 3.12，入口 `ai_rag.main:app`
- **docker-compose.yml**：API 服务 + 数据目录挂载（`./data/chroma_db`）
- **.dockerignore / .gitignore**：排除模型、数据、密钥（`.env` 不入库）
- **GitHub Actions**（`.github/workflows/ci.yml`）两个任务：
  - `check`：安装依赖 + 语法检查（compileall）
  - `docker`：构建镜像
- **实测结果**：CI 全绿 ✅（run #3，commit `b07e271`，2026-08-15）

### 踩坑记录（都解决了）
1. `requirements.txt` 曾有一行挤了两个包（`structlog pymupdf`）→ 改为一行一个包
2. Dockerfile 曾写死阿里云镜像源，GitHub 美国服务器连不上 → 改用官方 PyPI
3. `.env`（含密钥）曾误提交 → git 重写历史删除 + 轮换全部密钥

### 说明
CI 自动化已跑通；容器内完整运行（embedding 模型、Ollama 走宿主机）留到后续部署阶段。

---

## 12 周总览（全部完成 ✅）

| 周次 | 主题 | 核心产出 |
| --- | --- | --- |
| 1 | Celery + Redis 异步入库 | 上传秒回、后台处理 |
| 2 | 非结构化数据解析 | PDF 复杂表格 → Markdown；分块对比实验 |
| 3 | 混合检索 + Reranker | BM25+向量+RRF + BGE-Reranker 精排 |
| 4 | ReAct + Function Calling | Agent + 4 个工具 |
| 5 | LangGraph 工作流 | 条件分支 + 循环重试 |
| 6 | 长期记忆 + 多 Agent | 研究员/撰稿人协作 |
| 7 | Ollama 本地部署 | qwen3:8b + FastAPI 流式接入 |
| 8 | LoRA 微调 | qwen2.5-3b 行业专属微调（146 条，loss 2.26→0.64）|
| 9 | 量化 + RAGAS | GGUF q8_0；评估微调 0.870 最优 |
| 10 | Langfuse 可观测性 | trace + llm_call 全链路追踪 |
| 11 | 缓存 + 限流 + 熔断 | 语义缓存、20次/分限流、熔断快速拒绝 |
| 12 | Docker + CI/CD | Dockerfile/compose + GitHub Actions 全绿 |