# -*- coding: utf-8 -*-
"""熔断验收：每次用新问题（避免缓存），先退出 Ollama 再跑"""
import json, time, urllib.request, urllib.error, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for i in range(5):
    q = f"公司第{i+1}条内部管理制度是什么？"   # 新问题，必走 LLM
    payload = {"messages":[{"role":"user","content":q}],"user_id":"breaker-test"}
    req = urllib.request.Request("http://127.0.0.1:8000/api/rag/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        d = json.loads(resp.read().decode("utf-8"))
        print(f"第{i+1}次: 200 [{time.time()-t0:.2f}s]  {(d.get('ai_answer') or '')[:30]}")
    except urllib.error.HTTPError as e:
        print(f"第{i+1}次: {e.code} [{time.time()-t0:.2f}s]")
    time.sleep(0.3)
