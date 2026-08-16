# ===== 构建阶段 =====
FROM python:3.12 AS builder
WORKDIR /app
COPY requirements.txt .
# 第12周修复：GitHub Actions 在美国服务器，用官方 PyPI（阿里云镜像在美国不可靠）
RUN pip install --no-cache-dir --user -r requirements.txt

# ===== 运行阶段 =====
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 poppler-utils && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src
COPY --from=builder /root/.local /root/.local
COPY . .
EXPOSE 8000
CMD ["uvicorn", "ai_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
