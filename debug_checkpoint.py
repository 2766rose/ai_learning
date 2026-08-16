# debug_checkpoint.py
import asyncio
from langgraph.checkpoint.memory import MemorySaver

async def main():
    saver = MemorySaver()
    config = {"configurable": {"thread_id": "thread_emp_zhangsan_001"}}
    
    # MemorySaver.get 是同步方法，但为安全起见用 run_in_executor
    loop = asyncio.get_event_loop()
    checkpoint = await loop.run_in_executor(None, saver.get, config)
    
    if checkpoint is None:
        print("❌ 未找到该 thread_id 的 checkpoint")
        return
    
    messages = checkpoint.get("channel_values", {}).get("messages", [])
    print(f"📦 Checkpoint 中共有 {len(messages)} 条消息:\n")
    
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        msg_module = type(msg).__module__
        content = repr(getattr(msg, 'content', '<NO CONTENT ATTR>'))[:100]
        msg_id = getattr(msg, 'id', '<NO ID>')
        
        # 标记非标准类型
        from langchain_core.messages import BaseMessage
        is_standard = isinstance(msg, BaseMessage)
        flag = "✅" if is_standard else "🔴 非BaseMessage!"
        
        print(f"  {flag} [{i}] {msg_module}.{msg_type}")
        print(f"       id={msg_id}")
        print(f"       content={content}")
        print()

if __name__ == "__main__":
    asyncio.run(main())