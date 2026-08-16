# src/ai_rag/core/observability.py
"""Langfuse 可观测性（进程级单例；失败静默降级，绝不影响业务）"""
import logging
from contextlib import contextmanager

from dotenv import load_dotenv

from ai_rag.core.config import PROJECT_ROOT
load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger(__name__)

langfuse = None
AVAILABLE = False
try:
    from langfuse import Langfuse
    langfuse = Langfuse()
    AVAILABLE = True
    logger.info("Langfuse 可观测性已启用")
except Exception as e:  # pragma: no cover
    logger.warning("Langfuse 不可用，追踪关闭: %s", e)


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs):
    """请求级观测（用 with 包裹，子观测通过上下文自动嵌套）。
    用法:
        with observe("rag-chat", as_type="agent", input={...}):
            ...业务...
    """
    if not AVAILABLE:
        yield None
        return
    try:
        cm = langfuse.start_as_current_observation(
            name=name, as_type=as_type, end_on_exit=True, **kwargs
        )
    except Exception as e:
        logger.warning("Langfuse start 失败: %s", e)
        yield None
        return
    with cm as obs:
        yield obs


def start_observation(name: str, as_type: str = "span", **kwargs):
    """手动开启一个子观测（挂在当前上下文之下）。返回 obs 或 None。"""
    if not AVAILABLE:
        return None
    try:
        return langfuse.start_observation(name=name, as_type=as_type, **kwargs)
    except Exception as e:
        logger.warning("Langfuse start_observation 失败: %s", e)
        return None


def end_observation(obs) -> None:
    """安全结束观测。"""
    if obs is None:
        return
    try:
        obs.end()
    except Exception as e:
        logger.warning("Langfuse end 失败: %s", e)


def safe_usage_update(obs, output=None, done_obj=None) -> None:
    """把 Ollama 返回的 token 用量写入观测；任何异常都吞掉。"""
    if obs is None:
        return
    try:
        usage = None
        if done_obj:
            p = done_obj.get("prompt_eval_count")
            c = done_obj.get("eval_count")
            usage = {"input": p or 0, "output": c or 0, "total": (p or 0) + (c or 0)}
        obs.update(output=output, usage_details=usage)
    except Exception as e:
        logger.warning("Langfuse update 失败: %s", e)


def safe_update_output(obs, output=None) -> None:
    """安全写入输出（无用量）。"""
    if obs is None:
        return
    try:
        obs.update(output=output)
    except Exception as e:
        logger.warning("Langfuse update 失败: %s", e)
