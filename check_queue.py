# check_queue.py
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# 检查所有可能的队列名
for key in ['celery', 'default', 'ingest_document_task']:
    length = r.llen(key)
    print(f"List '{key}': {length} items")

# 再次列出所有 key
print("\nAll keys in DB 0:")
for k in r.scan_iter():
    t = r.type(k)
    print(f"  {k.decode()} (type={t.decode()})")