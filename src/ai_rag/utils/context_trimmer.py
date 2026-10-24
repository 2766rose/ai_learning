# context_trimmer.py
import json
import logging

import tiktoken

logger = logging.getLogger(__name__)

# 使用 cl100k_base 做近似估算，并额外保留消息 overhead。
ENCODER = tiktoken.get_encoding("cl100k_base")


def _as_message_dict(message):
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return {}


def _count_message_tokens(messages):
    """近似计算消息总 Token，包含 role 和工具调用开销。"""
    total = 0
    for raw in messages:
        message = _as_message_dict(raw)
        total += 4  # 消息元数据
        for key, value in message.items():
            if isinstance(value, str):
                total += len(ENCODER.encode(value))
            elif key == "tool_calls" and value:
                normalized = [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc
                    for tc in value
                ]
                total += len(ENCODER.encode(json.dumps(normalized, ensure_ascii=False)))
        total += 2  # role 开销
    return total


def _tool_call_ids(message):
    ids = set()
    for tc in message.get("tool_calls") or []:
        if isinstance(tc, dict):
            call_id = tc.get("id")
        else:
            call_id = getattr(tc, "id", None)
        if call_id:
            ids.add(str(call_id))
    return ids


def _group_messages(messages):
    """
    将消息切成原子组。
    assistant tool_calls 和它后面的 tool 结果必须一起保留。
    孤立或不完整的 tool 消息直接丢弃，避免生成非法消息序列。
    """
    groups = []
    i = 0
    while i < len(messages):
        message = messages[i]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            call_ids = _tool_call_ids(message)
            group = [message]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            result_ids = {
                str(m.get("tool_call_id"))
                for m in group[1:]
                if m.get("tool_call_id")
            }
            if call_ids and not call_ids.issubset(result_ids):
                logger.warning("丢弃不完整的工具调用组: missing=%s", call_ids - result_ids)
            else:
                groups.append(group)
            i = j
            continue
        if role == "tool":
            logger.warning("丢弃孤立的 tool 消息: tool_call_id=%s", message.get("tool_call_id"))
            i += 1
            continue
        groups.append([message])
        i += 1
    return groups


def _truncate_text_to_tokens(text, max_tokens):
    if max_tokens <= 0:
        return ""
    tokens = ENCODER.encode(str(text))
    if len(tokens) <= max_tokens:
        return str(text)
    return ENCODER.decode(tokens[:max_tokens])


def _fallback_latest_user(messages, remaining_budget):
    """预算极小时，至少保留最近一条 user 消息，并截断其内容。"""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        candidate = {"role": "user", "content": ""}
        overhead = _count_message_tokens([candidate])
        content_budget = max(0, remaining_budget - overhead)
        candidate["content"] = _truncate_text_to_tokens(
            message.get("content", ""), content_budget
        )
        return [candidate]
    return []


def trim_messages(messages, max_tokens=4000):
    """
    按总 Token 预算裁剪消息。

    max_tokens 表示 System Prompt + 历史消息的近似总预算。
    优先保留最近消息；工具调用组不可拆分；最终必须尽量不超预算。
    """
    if not messages:
        return []

    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]
    system_tokens = _count_message_tokens(system_messages)
    remaining_budget = max_tokens - system_tokens

    if remaining_budget <= 0:
        fallback = _fallback_latest_user(non_system_messages, 0)
        logger.warning(
            "System Prompt 已占用 %d tokens，预算 %d，保留最近 user=%d 条",
            system_tokens, max_tokens, len(fallback),
        )
        return system_messages + fallback

    groups = _group_messages(non_system_messages)
    if not groups:
        return system_messages

    last_user_group = None
    for idx in range(len(groups) - 1, -1, -1):
        if any(m.get("role") == "user" for m in groups[idx]):
            last_user_group = idx
            break

    kept_groups = []
    used = 0
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx]
        cost = _count_message_tokens(group)
        if used + cost <= remaining_budget:
            kept_groups.insert(0, group)
            used += cost
            continue
        if idx == last_user_group:
            fallback = _fallback_latest_user(non_system_messages, remaining_budget)
            kept_groups = [fallback] if fallback else []
            break

    if not kept_groups:
        fallback = _fallback_latest_user(non_system_messages, remaining_budget)
        kept_groups = [fallback] if fallback else []

    final_messages = system_messages + [m for group in kept_groups for m in group]

    # 最终硬校验：优先丢弃最旧的普通消息，避免再次出现 2667 > 2048。
    while _count_message_tokens(final_messages) > max_tokens and len(kept_groups) > 1:
        if any(m.get("role") == "user" for m in kept_groups[0]):
            break
        kept_groups.pop(0)
        final_messages = system_messages + [m for group in kept_groups for m in group]

    if _count_message_tokens(final_messages) > max_tokens:
        fallback = _fallback_latest_user(non_system_messages, remaining_budget)
        final_messages = system_messages + fallback

    if _count_message_tokens(final_messages) > max_tokens:
        logger.warning(
            "裁剪后仍超出预算: final=%d budget=%d（通常是 System Prompt 本身过大）",
            _count_message_tokens(final_messages), max_tokens,
        )

    logger.info(
        "Token 裁剪完成: kept_groups=%d final=%d budget=%d",
        len(kept_groups), _count_message_tokens(final_messages), max_tokens,
    )
    return final_messages
