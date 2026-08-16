# -*- coding: utf-8 -*-
"""带耗时显示的测试脚本（验证语义缓存）"""
import json, time, urllib.request, sys
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
def ask(q):
    payload = {"messages":[{"role":"user","content":q}],"user_id":"langfuse-test"}
    req = urllib.request.Request("http://127.0.0.1:8000/api/rag/chat", data=json.dumps(payload).encode("utf-8"), headers=_api_headers())
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    d = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    print("Q:", q)
    print("A:", (d.get("ai_answer") or "")[:60], f"  [{dt:.2f}s]")
    print()

ask("公积金缴纳比例是多少？")
ask("高铁出差的报销标准是什么？")
