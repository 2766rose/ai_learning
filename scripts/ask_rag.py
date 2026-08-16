# -*- coding: utf-8 -*-
"""带耗时显示的测试脚本（验证语义缓存）"""
import json, time, urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def ask(q):
    payload = {"messages":[{"role":"user","content":q}],"user_id":"langfuse-test"}
    req = urllib.request.Request("http://127.0.0.1:8000/api/rag/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    d = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    print("Q:", q)
    print("A:", (d.get("ai_answer") or "")[:60], f"  [{dt:.2f}s]")
    print()

ask("公积金缴纳比例是多少？")
ask("高铁出差的报销标准是什么？")
