# celery_app.py
import os
from celery import Celery
from celery.signals import setup_logging, worker_process_init, worker_ready
from kombu import Queue, Exchange

from ai_rag.core.config import rag_config
from ai_rag.core.logging_config import configure_logging

# ==========================================
# 1. Celery 实例初始化
# ==========================================
celery_app = Celery(
    "ai_worker",
    broker=rag_config.CELERY_BROKER_URL,
    backend=rag_config.CELERY_RESULT_BACKEND,
    include=[
        "ai_rag.tasks.document_tasks",
        # 未来第二阶段 Agent 任务可在此扩展
        # "ai_rag.tasks.agent_tasks", 
    ],
)

# ==========================================
# 2. 核心配置 (生产级加固)
# ==========================================
celery_app.conf.update(
    # --- 序列化与基础设置 ---
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_publish_retry=False,
    task_log_level="DEBUG",
    
    # 🛡️ --- 生产级加固：超时控制 (防止 PDF 解析卡死导致僵尸任务) ---
    # 软超时：300秒后抛出 SoftTimeLimitExceeded，任务可捕获异常并清理资源
    task_soft_time_limit=300,  
    # 硬超时：310秒后强制杀死 Worker 进程，防止死锁
    task_time_limit=310,       
    
    # 🧹 --- 生产级加固：结果管理 (防止 Redis 内存爆满) ---
    # 任务结果在 Redis 中保留 24 小时 (86400秒) 后自动清理
    result_expires=86400,      
    
    # 🔒 --- 生产级加固：容错与重试 (防止 Worker 崩溃导致任务丢失) ---
    # 晚期 ACK：任务执行完成后才发送 ACK。如果 Worker 中途被 kill，任务会重新回到队列
    task_acks_late=True,       
    # 预取限制：配合 task_acks_late 使用，确保任务在 Worker 之间均匀分配，避免单个 Worker 囤积过多任务
    worker_prefetch_multiplier=1, 
)

# ==========================================
# 3. 队列路由配置
# ==========================================
celery_app.conf.task_queues = (
    Queue('celery', Exchange('celery'), routing_key='celery'),
    # 未来可扩展：将 CPU 密集型 (解析) 与 IO 密集型 (向量化/API) 拆分到不同队列
    # Queue('parsing', Exchange('parsing'), routing_key='parsing'),
    # Queue('embedding', Exchange('embedding'), routing_key='embedding'),
)
celery_app.conf.task_default_queue = 'celery'
celery_app.conf.task_default_exchange = 'celery'
celery_app.conf.task_default_routing_key = 'celery'

# ==========================================
# 4. 信号处理 (生命周期管理)
# ==========================================
@worker_ready.connect
def on_worker_ready(**kwargs):
    print("=" * 60, flush=True)
    print("🟢 CELERY WORKER READY - 模块加载成功", flush=True)
    print(f"🔗 Worker Broker: {rag_config.CELERY_BROKER_URL}", flush=True)
    print(f"📋 Registered Tasks: {list(celery_app.tasks.keys())}", flush=True)
    print("=" * 60, flush=True)
    try:
        from ai_rag.services.etl_service import ingest_document
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