import requests
import json

# ✅ 改为流式接口路径
URL = "http://localhost:8000/api/rag/chat/stream"

payload = {
    "messages": [{"role": "user", "content": "什么是RAG?"}],
    "stream": True,"user_id": "test-user-001",
}

resp = requests.post(URL, json=payload, stream=True)

if resp.status_code != 200:
    print(f"❌ 请求失败 [{resp.status_code}]: {resp.text}")
else:
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("type") == "chunk":
                print(data["data"], end="", flush=True)
            elif data.get("type") == "error":
                print(f"\n❌ SSE Error: {data['message']}")
            elif data.get("type") == "done":
                print("\n✅ 流式响应结束")