import redis
from src.ai_rag.core.config import rag_config

r = redis.from_url(rag_config.CELERY_BROKER_URL)
keys = r.keys("*")
print(f"🔑 Redis DB {rag_config.CELERY_BROKER_URL.split('/')[-1]} 中所有 key:")
for k in sorted(keys):
    ktype = r.type(k)
    length = r.llen(k) if ktype == b"list" else "N/A"
    print(f"  {k.decode()} (type={ktype.decode()}, len={length})")