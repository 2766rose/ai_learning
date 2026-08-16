# -*- coding: utf-8 -*-
"""熔断验收：每次用新问题（避免缓存），先退出 Ollama 再跑"""
import json, time, urllib.request, urllib.error, sys
def _api_headers():
    import os
    _k = ""
    try:
        for _l in open(r"D:\ai_learning\.env", "r", encoding="utf-8"):
            _l = _l.strip()
            if _l.startswith("RAG_API_KEY="):
                _k = _l.split("=", 1)[1].strip()
                break
    except Exception:
        pass
    _h = {"Content-Type": "application/json"}
    if _k:
        _h["X-API-Key"] = _k
    return _h

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for i in range(5):
    q = f"公司第{i+1}条内部管理制度是什么？"   # 新问题，必走 LLM
    payload = {"messages":[{"role":"user","content":q}],"user_id":"breaker-test"}
    req = urllib.request.Request("http://127.0.0.1:8000/api/rag/chat", data=json.dumps(payload).encode("utf-8"), headers=_api_headers())
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        d = json.loads(resp.read().decode("utf-8"))
        print(f"第{i+1}次: 200 [{time.time()-t0:.2f}s]  {(d.get('ai_answer') or '')[:30]}")
    except urllib.error.HTTPError as e:
        print(f"第{i+1}次: {e.code} [{time.time()-t0:.2f}s]")
    time.sleep(0.3)
