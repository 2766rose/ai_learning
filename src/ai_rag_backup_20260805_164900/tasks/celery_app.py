# celery_app.py
from celery import Celery
from celery.signals import setup_logging, worker_process_init, worker_ready
from kombu import Queue, Exchange

from src.ai_rag.core.config import rag_config
from src.ai_rag.core.logging_config import configure_logging


@worker_ready.connect
def on_worker_ready(**kwargs):
    print("=" * 60, flush=True)
    print("🟢 CELERY WORKER READY - 模块加载成功", flush=True)
    print(f"🔗 Worker Broker: {rag_config.CELERY_BROKER_URL}", flush=True)
    print(f"📋 Registered Tasks: {list(celery_app.tasks.keys())}", flush=True)
    print("=" * 60, flush=True)
    try:
        from src.ai_rag.services.etl_service import ingest_document
        print("✅ etl_service 导入成功", flush=True)
    except Exception as e:
        print(f"❌ etl_service 导入失败: {e}", flush=True)
        import traceback
        traceback.print_exc()


@setup_logging.connect
def on_setup_logging(**kwargs):
    configure_logging(json_output=True)


@worker_process_init.connect
def on_worker_process_init(**kwargs):
    configure_logging(json_output=True)


celery_app = Celery(
    "ai_worker",
    broker=rag_config.CELERY_BROKER_URL,
    backend=rag_config.CELERY_RESULT_BACKEND,
    include=["src.ai_rag.tasks.document_tasks"],
)

# ✅ 移除 broker_transport_options={'confirm_publish': True}
# Redis transport 不支持 AMQP 的 publisher confirms，设置后会被静默忽略
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_publish_retry=False,
    task_log_level="DEBUG",
)

celery_app.conf.task_queues = (
    Queue('celery', Exchange('celery'), routing_key='celery'),
)
celery_app.conf.task_default_queue = 'celery'
celery_app.conf.task_default_exchange = 'celery'
celery_app.conf.task_default_routing_key = 'celery'