# -*- coding: utf-8 -*-
"""限流验收：连发 4 次请求，看第 4 次是否 429"""
import json, time, urllib.request, urllib.error, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
Q = "公积金缴纳比例是多少？"
for i in range(4):
    payload = {"messages":[{"role":"user","content":Q}],"user_id":"rl-test"}
    req = urllib.request.Request("http://127.0.0.1:8000/api/rag/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        d = json.loads(resp.read().decode("utf-8"))
        print(f"第{i+1}次: {resp.status}  [{time.time()-t0:.2f}s]  { (d.get('ai_answer') or '')[:20]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8","replace")[:80]
        print(f"第{i+1}次: {e.code}  [{time.time()-t0:.2f}s]  {body}")
    time.sleep(0.2)
