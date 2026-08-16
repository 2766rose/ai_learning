# -*- coding: utf-8 -*-
"""Langfuse 连通性验证（读 .env 里的密钥，适配 langfuse 4.x）"""
import os, sys
from dotenv import load_dotenv
load_dotenv(r"D:\ai_learning\.env")   # 从 .env 读 LANGFUSE_* 密钥

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
    if not os.environ.get(k):
        raise SystemExit(f"缺少 {k}，请检查 .env")

from langfuse import Langfuse
lf = Langfuse()

with lf.start_as_current_observation(
    name="week10-connectivity-test",
    as_type="generation",
    model="qwen3:8b",
    input={"msg": "hello langfuse"},
    output="这是一条测试数据",
    usage_details={"input": 10, "output": 20, "total": 30},
):
    pass

lf.flush()
print("已上报，请到控制台查看 week10-connectivity-test")
print("控制台:", os.environ["LANGFUSE_BASE_URL"])
