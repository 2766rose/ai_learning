# logging_config.py
import sys
import logging
import structlog
from celery.signals import setup_logging


@setup_logging.connect
def _on_setup_logging(**kwargs):
    """Celery 信号入口：绝不让 logging 配置异常影响任务执行"""
    try:
        configure_logging()
    except Exception as e:
        print(f"[LOGGING CONFIG ERROR] {e}", file=sys.stderr, flush=True)


def configure_logging(json_output: bool = False) -> None:
    """
    配置 structlog + 标准 logging 的混合日志系统。
    兼容 Celery Worker 的日志体系，不破坏其已有 handler。
    """
    # structlog 处理器链
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        renderer = structlog.dev.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # ✅ 防御性替换 formatter，跳过任何不支持的 handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        try:
            handler.setFormatter(formatter)
        except Exception:
            pass

    # 非 Celery 环境（如直接运行脚本）才添加默认 handler
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.INFO)