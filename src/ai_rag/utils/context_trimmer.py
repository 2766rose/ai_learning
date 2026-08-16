# context_trimmer.py
import json
import tiktoken

# 预加载编码器，避免每次调用都重新初始化（提升性能）
# qwen-plus / qwen-max 等模型兼容 cl100k_base 编码
ENCODER = tiktoken.get_encoding("cl100k_base")

def _count_message_tokens(messages):
    """
    精确计算 messages 列表的 Token 数。
    注：API 实际计费还会加上每条消息约 3-4 个固定 overhead token，
    这里我们采用保守策略，每条消息额外 +4 以确保绝不超限。
    """
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # 消息元数据开销
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(ENCODER.encode(value))
            elif key == "tool_calls" and value:
                # tool_calls 是对象列表，需要序列化后计算
                num_tokens += len(ENCODER.encode(json.dumps([tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in value])))
        num_tokens += 2  # role 字段开销
    return num_tokens

def trim_messages(messages, max_tokens=4000):
    """
    Token 级智能裁剪器：确保不破坏 Function Calling 结构，且严格控制在 Token 预算内。
    
    :param messages: 完整的 messages 列表
    :param max_tokens: 允许保留的最大非 System Token 数 (默认 4000)
    :return: 裁剪后的 messages 列表
    """
    # 1. 提取并保护 System Prompt
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]
    
    # 计算 System Prompt 占用的 Token，剩余预算留给历史消息
    system_tokens = _count_message_tokens(system_messages)
    remaining_budget = max_tokens - system_tokens
    
    if remaining_budget <= 0:
        # 2026-08-14 修复：绝不能把对话塌缩成只剩 system，否则 Agent 会因丢失上下文反复调工具
        recent = non_system_messages[-4:]
        print(f"⚠️ System Prompt 已占用 {system_tokens} tokens，超出预算 {max_tokens}！保底保留最近 {len(recent)} 条历史。")
        return system_messages + recent

    # 如果历史消息未超 Token 预算，直接原样返回
    history_tokens = _count_message_tokens(non_system_messages)
    if history_tokens <= remaining_budget:
        return messages

    print(f"⚠️ 历史消息 ({history_tokens} tokens) 超出预算 ({remaining_budget} tokens)，开始 Token 级智能裁剪...")

    # 2. 从后往前（由近及远）挑选，直到预算耗尽
    kept_messages = []
    kept_tool_call_ids = set()
    current_used_tokens = 0

    for msg in reversed(non_system_messages):
        # 计算当前这条消息的 Token 数
        msg_tokens = _count_message_tokens([msg])
        
        # 预算检查：如果加上这条就超了，立刻停止
        if current_used_tokens + msg_tokens > remaining_budget:
            print(f"   ✂️ Token 预算耗尽，停止添加更多历史消息。")
            break
            
        role = msg.get("role")
        
        # 情况 A: Tool 返回结果
        if role == "tool":
            kept_messages.insert(0, msg)
            kept_tool_call_ids.add(msg.get("tool_call_id"))
            current_used_tokens += msg_tokens
            continue
            
        # 情况 B: Assistant 发起的工具调用
        if role == "assistant" and msg.get("tool_calls"):
            call_ids_in_this_msg = {tc.id for tc in msg.tool_calls}
            # 铁律：如果对应的 tool 结果不在保留列表中，这个 assistant 也不能要
            if not call_ids_in_this_msg.issubset(kept_tool_call_ids):
                print(f"   ✂️ 丢弃不完整的 Assistant 工具调用 (Token: {msg_tokens})")
                continue
                
            kept_messages.insert(0, msg)
            current_used_tokens += msg_tokens
            continue
            
        # 情况 C: 普通 User / Assistant 文本
        kept_messages.insert(0, msg)
        current_used_tokens += msg_tokens

    # 3. 重新组装（若全部被裁掉，保底保留最近 4 条，避免对话塌缩）
    if not kept_messages and non_system_messages:
        kept_messages = non_system_messages[-4:]
    final_messages = system_messages + kept_messages
    final_tokens = _count_message_tokens(final_messages)
    
    print(f"✅ Token 级裁剪完成！保留 {len(kept_messages)} 条历史，总计 {final_tokens} tokens (预算: {max_tokens})")
    return final_messages