# src/ai_rag/agent/router.py
"""
Agent Router - 模型路由策略引擎 (Enterprise Edition)
规范说明：
1. 禁止运行时创建目录/写入默认配置
2. 配置缺失或格式错误时快速失败，拒绝静默降级
3. 使用 Pydantic 进行强类型校验
4. 决策日志结构化埋点，为后续 LLM-as-Judge 预留数据集
"""
import logging
import time
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

from ai_rag.core.config import rag_config
from ai_rag.core.llm_client import get_async_openai_client

logger = logging.getLogger(__name__)

RouteDecision = Literal["local", "cloud"]
_RULES_CACHE: Optional["RouterRules"] = None
_RULES_PATH = Path(__file__).parent.parent / "config" / "router_rules.yaml"


class RouterRules(BaseModel):
    """路由规则 Schema - 强制类型校验，杜绝裸字典访问"""
    local_keywords: list[str] = Field(default_factory=list, description="触发本地模型的关键词")
    cloud_keywords: list[str] = Field(default_factory=list, description="触发云端模型的关键词")
    default: Literal["local", "cloud"] = Field(default="cloud", description="未匹配时的默认路由")


def _load_rules() -> RouterRules:
    """
    加载并校验路由规则
    ❌ 无兜底、无自动建目录、无运行时写盘
    ✅ 配置异常直接抛异常，由上层健康检查/启动流程捕获
    """
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE

    # 1. 文件存在性检查
    if not _RULES_PATH.exists():
        raise FileNotFoundError(
            f"❌ Router rules config missing: {_RULES_PATH}\n"
            "Please ensure the config file is correctly mounted or placed in the expected path."
        )

    # 2. 读取 + Pydantic 强校验
    try:
        with open(_RULES_PATH, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        
        if raw_data is None:
            raise ValueError("Router rules config file is empty")
            
        _RULES_CACHE = RouterRules(**raw_data)
        
    except ValidationError as e:
        raise ValueError(f"❌ Router rules schema validation failed:\n{e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"❌ Router rules YAML parse error: {e}") from e

    logger.info("✅ Router rules loaded & validated | local=%d | cloud=%d | default=%s",
                len(_RULES_CACHE.local_keywords),
                len(_RULES_CACHE.cloud_keywords),
                _RULES_CACHE.default)
    return _RULES_CACHE


def route(query: str) -> RouteDecision:
    """
    关键词规则路由（V1）
    接口签名稳定，未来升级为 LLM-as-Judge 时仅需修改内部实现
    """
    start = time.monotonic()
    rules = _load_rules()
    q_lower = query.lower()

    decision: RouteDecision = rules.default
    reason = "default"

    # 优先级：cloud > local > default
    for kw in rules.local_keywords:
        if kw in q_lower:
            decision, reason = "local", f"local_keyword:{kw}"
            break

    for kw in rules.cloud_keywords:
        if kw in q_lower:
            decision, reason = "cloud", f"cloud_keyword:{kw}"
            break

    latency_ms = int((time.monotonic() - start) * 1000)
    
    # 🔑 结构化埋点：未来训练/评估 Judge 模型的黄金数据集
    logger.info(
        "🔀 ROUTE_DECISION | query='%s' | decision=%s | reason=%s | latency_ms=%d",
        query[:80], decision, reason, latency_ms
    )
    return decision


def get_client(decision: Optional[RouteDecision] = None, query: str = "") -> AsyncOpenAI:
    """
    统一客户端工厂（企业级：纯 AsyncOpenAI 协议）
    - Local(Ollama) 与 Cloud 返回相同类型，消除双轨制
    - 传入 decision：直接使用指定客户端
    - 传入 query：自动路由后返回对应客户端
    - 都不传：返回默认云端客户端
    """
    if decision is None and query:
        decision = route(query)
    elif decision is None:
        decision = _load_rules().default

    if decision == "local":
        logger.debug("📦 Returning Local Ollama AsyncOpenAI client | model=%s", rag_config.LOCAL_MODEL)
        return AsyncOpenAI(
            base_url=rag_config.LOCAL_BASE_URL,
            api_key="ollama",  # Ollama 不校验 key，传任意非空字符串即可
            timeout=rag_config.LLM_TIMEOUT,
        )

    logger.debug("☁️ Returning Cloud OpenAI client | model=%s", rag_config.OPENAI_MODEL)
    return get_async_openai_client()