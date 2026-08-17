# -*- coding: utf-8 -*-
"""RAGAS 评估 —— 微调模型 vs RAG
指标：
- answer_correctness：回答与参考答案的语义一致度（LLM 评判）
- faithfulness：回答是否忠于检索上下文（仅 RAG 场景）
评判 LLM：本地 Ollama qwen3:8b（OpenAI 兼容接口）
用法: python eval_ragas.py
"""
import json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_V1 = "http://127.0.0.1:11434/v1"
FT_MODEL = "qwen2.5-3b-ft:latest"
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE", "qwen2.5:7b")
RAG_API = "http://localhost:8000/api/rag/chat"
EVAL_FILE = r"D:\ai_learning\data\staff_qa_eval_50.json"

# ---------- 1. 取回答 ----------
def ollama_chat(model, q, max_tokens=200):
    payload = {"model": model, "messages": [{"role": "user", "content": q}], "stream": False, "options": {"temperature": 0.1, "num_predict": max_tokens}}
    req = urllib.request.Request(OLLAMA_V1.replace("/v1", "/api/chat"), data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=600)
    return (json.loads(resp.read().decode("utf-8")).get("message") or {}).get("content", "").strip()

def rag_chat(q):
    payload = {"messages": [{"role": "user", "content": q}], "user_id": "ragas-eval", "stream": False}
    req = urllib.request.Request(RAG_API, data=json.dumps(payload).encode("utf-8"), headers=_api_headers())
    last = None
    for _try in range(3):
        try:
            resp = urllib.request.urlopen(req, timeout=600)
            d = json.loads(resp.read().decode("utf-8"))
            return d.get("ai_answer", "").strip(), d.get("retrieved_knowledge", "") or ""
        except Exception as e:
            last = e
            print("  [eval] RAG call retry %d: %s" % (_try + 1, type(e).__name__))
            time.sleep(5)
    raise last


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


data = json.load(open(EVAL_FILE, encoding="utf-8"))[:int(os.environ.get("EVAL_COUNT", "50"))]
print(f"评估样本数: {len(data)}")

ft_records, rag_records = [], []
for i, ex in enumerate(data):
    q, ref = ex["instruction"], ex["output"]
    if os.environ.get("RAGAS_SKIP_FT") != "1":
        ft = ollama_chat(FT_MODEL, q)
        ft_records.append({"question": q, "answer": ft, "reference": ref})
    try:
        rag, ctx = rag_chat(q)
        rag_records.append({"question": q, "answer": rag, "reference": ref, "contexts": [ctx] if ctx else []})
    except Exception as e:
        print(f"RAG 不可用（跳过 RAG 对比）: {e}")
        break
    if (i + 1) % 5 == 0:
        print(f"  已取 {i+1}/{len(data)}")

print("微调模型回答完成:", len(ft_records), "| RAG回答完成:", len(rag_records))
with open(r"D:\ai_learning\data\eval_results.json", "w", encoding="utf-8") as _f:
    json.dump(rag_records, _f, ensure_ascii=False, indent=2)
print("详细结果已落盘: D:\\ai_learning\\data\\eval_results.json")

# ---------- 0. ragas 0.2.15 兼容桩（新版 langchain-community 移除了 vertexai，评估用不到） ----------
import sys, types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vmod = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI:
        def __init__(self, *a, **k):
            raise NotImplementedError("VertexAI stub")
    _vmod.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vmod

# ---------- 2. RAGAS 评估 ----------
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = OLLAMA_V1
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

# ---------- 自定义正确性评判（原生 /api/chat，Ollama 与 ragas 的 JSON 模式不兼容） ----------
import re
def native_chat(q, model=None, max_tokens=300):
    model = model or JUDGE_MODEL
    payload = {"model": model, "messages": [{"role": "user", "content": q}], "stream": False, "think": False, "options": {"temperature": 0, "num_predict": max_tokens}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=180)
    return (json.loads(resp.read().decode("utf-8")).get("message") or {}).get("content", "")

def judge_correctness(question, answer, reference):
    prompt = (
        "你是严谨的评估员。根据参考答案判断模型回答的事实一致性。\n"
        "参考答案：" + reference + "\n"
        "模型回答：" + (answer or "")[:400] + "\n"
        "只输出一个 0 到 1 之间的小数（1.0 表示完全一致，0.0 表示完全不符）："
    )
    out = native_chat(prompt)
    m = re.search(r"1\.0|0\.\d+|[01]\b", out)
    try:
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0

def avg_correctness(records):
    if not records:
        return float("nan")
    s = 0.0
    for r in records:
        s += judge_correctness(r["question"], r["answer"], r["reference"])
    return s / len(records)
from ragas.metrics import answer_correctness, faithfulness

llm = ChatOpenAI(model=JUDGE_MODEL, base_url=OLLAMA_V1, api_key="ollama", temperature=0)

if ft_records:
    ds_ft = Dataset.from_list(ft_records)
    ft_correctness = avg_correctness(ft_records)
    print("\n===== 微调模型 (qwen2.5-3b-ft) =====")
    print(f"answer_correctness(自定义评判): {ft_correctness:.3f}")

if rag_records:
    ds_rag = Dataset.from_list(rag_records)
    r_rag = evaluate(ds_rag, metrics=[faithfulness], llm=llm, run_config=RunConfig(max_workers=2, timeout=300))
    rag_correctness = avg_correctness(rag_records)
    print("\n===== RAG (qwen3:8b + 混合检索) =====")
    print(f"answer_correctness(自定义评判): {rag_correctness:.3f}")
    print("faithfulness(ragas):", r_rag)
