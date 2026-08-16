# ===== 构建阶段 =====
FROM python:3.12 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# ===== 运行阶段 =====
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 poppler-utils && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/root/.local/bin:$PATH
COPY --from=builder /root/.local /root/.local
COPY . .
EXPOSE 8000
# 第12周修正：模块名应为 ai_rag.main（原 app.main 不存在）
CMD ["uvicorn", "ai_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
