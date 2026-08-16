# src/ai_rag/agent/runner.py
"""
Agent 核心调度模块 (Runner)
负责管理 LLM 对话循环、工具调用分发、长期记忆注入及上下文裁剪。

v2（2026-08-14）：
- LLM 调用由 OpenAI 兼容接口(/v1/chat/completions) 切换为 Ollama 原生 /api/chat
- 通过顶层参数 think=false 关闭 qwen3 思考模式，显著降低"首字延迟"
  （本机 Ollama 版本的 /v1 接口会忽略 think 参数，因此不再使用 OpenAI SDK）
"""

import json
import logging
from typing import AsyncGenerator, Union, List, Dict, Any, Tuple, Optional

import httpx

from ai_rag.core.config import rag_config
from ai_rag.agent.tools import TOOL_REGISTRY_MAP, TOOL_SCHEMAS
from ai_rag.agent.memory import retrieve_memories
from ai_rag.utils.context_trimmer import trim_messages
from ai_rag.core.observability import observe, start_observation, end_observation, safe_usage_update, safe_update_output
from ai_rag.core.circuit_breaker import llm_circuit_breaker

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ollama 原生接口配置
# -----------------------------------------------------------------------------
OLLAMA_BASE_URL = rag_config.OPENAI_BASE_URL.rstrip("/")
if OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL[: -len("/v1")]

# 带思考能力的模型默认会先输出推理过程；这里通过 Ollama 顶层参数显式关闭
_THINKING_MODEL_KEYWORDS = ("qwen3", "qwq", "deepseek-r1", "reason")


def _should_disable_thinking(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(k in name for k in _THINKING_MODEL_KEYWORDS)


MAX_AGENT_ITERATIONS = 5  # 最大工具调用轮次，防止死循环
MEMORY_INJECTION_MARKER = "📌 以下为检索到的用户记忆"

SYSTEM_PROMPT = f"""你是企业知识库助手，严格遵守以下规则：

【工具选择原则】
按以下优先级判断，命中即执行，不再继续向下匹配：
1. 时间/日期/当前时刻 → get_current_time
2. 天气/气温/降雨/风力等 → get_weather
3. 企业内部知识/文档/业务规范/产品手册 → knowledge_search
4. 当用户自我介绍、告知姓名/偏好/职业/身份等个人稳定信息时，必须调用 save_user_memory 保存 → save_user_memory
5. 以上均不适用 → 直接基于对话上下文回答，不调用任何工具

【组合调用】
- 允许一轮对话中组合调用多个工具
- 但同一类问题不要重复调用相同工具

【用户长期记忆】
⚠️ 仅当下方出现「{MEMORY_INJECTION_MARKER}」标记时，才表示存在有效记忆。
- 有记忆时：自然融入回答，不要提及"根据您的记忆"等元描述；仅在用户主动提及或询问相关个人事实（如名字、偏好）时才使用记忆，日常问候或无关话题时不要主动提及用户名字。
- 无记忆时（未出现上述标记）：如用户询问个人偏好/禁忌/身份等信息，必须如实告知"目前没有找到您的相关记录"，严禁编造、猜测或使用示例数据。

【回答规范】
1. 回答要简洁，只回答用户问到的具体内容，不要罗列检索到的全部信息；仅依据工具返回的内容作答，禁止编造或推测。
2. 引用知识库内容时使用来源编号标注，如 [1][2]。
3. 仅当 knowledge_search 返回空结果且无其他适用工具时，才回复："抱歉，知识库中未找到与您问题相关的信息。"
4. 不要向用户提及工具名称、检索过程等内部实现细节。
5. 金额、天数、比例、日期等所有具体数字必须逐字来自检索内容，严禁自行编造或推算；检索内容未覆盖的细节，明确回答"知识库中未找到相关信息"。"""

# 全局 HTTP 客户端（复用连接池）
_http_client = httpx.AsyncClient(
    base_url=OLLAMA_BASE_URL,
    timeout=rag_config.LLM_TIMEOUT,
)


# -----------------------------------------------------------------------------
# Ollama 请求/响应辅助函数
# -----------------------------------------------------------------------------
def _build_ollama_payload(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """构造 Ollama 原生 /api/chat 请求体（含关闭思考的顶层 think 参数）"""
    payload: Dict[str, Any] = {
        "model": rag_config.OPENAI_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": rag_config.LLM_TEMPERATURE,
            "num_predict": rag_config.LLM_MAX_TOKENS,
        },
    }
    if tools:
        payload["tools"] = tools
    if _should_disable_thinking(rag_config.OPENAI_MODEL):
        # 关键：Ollama 原生接口的顶层参数 think=false 才能真正关闭思考
        payload["think"] = False
    return payload


def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """把 Ollama 返回的工具调用规范化为 [{name, arguments, id}] 结构"""
    normalized: List[Dict[str, Any]] = []
    if not tool_calls:
        return normalized
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "") or ""
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        normalized.append({
            "id": tc.get("id", "") or "",
            "name": name,
            "arguments": args,
        })
    return normalized


def _merge_tool_calls(
    target: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
) -> None:
    """按 function.index 合并流式到达的工具调用分片（幂等）"""
    for tc in incoming:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) or {}
        idx = fn.get("index", 0)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = 0
        while len(target) <= idx:
            target.append({})
        entry = target[idx]
        if not entry:
            entry.update({
                "id": tc.get("id", "") or "",
                "function": {
                    "name": fn.get("name", "") or "",
                    "arguments": fn.get("arguments", {}) or {},
                },
            })
            target[idx] = entry
        else:
            if tc.get("id"):
                entry["id"] = tc.get("id")
            if fn.get("name"):
                entry["function"]["name"] = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, dict) and args:
                entry["function"]["arguments"] = args


def _build_assistant_tool_message(
    collected_content: str,
    normalized_tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构造下一轮请求所需的 assistant 消息（Ollama 工具调用格式）"""
    return {
        "role": "assistant",
        "content": collected_content or None,
        "tool_calls": [
            {
                "id": tc["id"] or f"call_{i}",
                "function": {
                    "index": i,
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            }
            for i, tc in enumerate(normalized_tool_calls)
        ],
    }


# -----------------------------------------------------------------------------
# 内部辅助函数
# -----------------------------------------------------------------------------
async def _execute_tool(tool_name: str, arguments: Dict[str, Any], user_id: Optional[str] = None) -> str:
    """动态分发并执行工具调用"""
    tool_func = TOOL_REGISTRY_MAP.get(tool_name)

    if tool_func is None:
        logger.warning("[Tool] 未知工具调用: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

    try:
        _tool_cfg = {"configurable": {"user_id": user_id}} if user_id else None
        result = await tool_func.ainvoke(arguments, config=_tool_cfg)
        result_str = str(result)
        logger.info(
            "[Tool] 执行成功 | tool=%s | args=%s | result_len=%d",
            tool_name, str(arguments)[:100], len(result_str),
        )
        return result_str
    except Exception as e:
        logger.exception("[Tool] 执行异常 | tool=%s | args=%s", tool_name, str(arguments))
        return json.dumps({"error": f"Tool execution failed: {str(e)}"}, ensure_ascii=False)


def _build_initial_messages(system_content: str, user_message: str) -> List[Dict[str, Any]]:
    """
    构建初始消息列表。
    思考模式已通过 Ollama 顶层参数 think=false 关闭，无需再追加 /no_think 文本
    （该文本在旧实现里只是普通文字，无法真正关闭思考）。
    """
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]


# -----------------------------------------------------------------------------
# 核心调度逻辑
# -----------------------------------------------------------------------------
async def _stream_tool_loop(
    messages: List[Dict[str, Any]],
    session_id: str,
    user_id: Optional[str] = None,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> AsyncGenerator[str, None]:
    """
    全流式 Tool Calling 循环（基于 Ollama 原生 /api/chat）。
    - 所有轮次均使用 stream=True，实现真正的逐 Token SSE 推送
    - 支持流式 Tool Calls 到达（Ollama 在生成完成后一次性下发 tool_calls）
    """
    for iteration in range(max_iterations):
        collected_content = ""
        collected_tool_calls: List[Dict[str, Any]] = []

        if not llm_circuit_breaker.allow():
            yield "⚠️ 服务繁忙（熔断中），请稍后重试。"
            return
        obs = start_observation("llm_call", "generation", model=rag_config.OPENAI_MODEL, input=messages)
        try:
            async with _http_client.stream(
                "POST",
                "/api/chat",
                json=_build_ollama_payload(messages, tools=TOOL_SCHEMAS, stream=True),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message", {}) or {}

                    # 1. 实时输出正文内容
                    delta_content = msg.get("content") or ""
                    if delta_content:
                        collected_content += delta_content
                        yield delta_content

                    # 2. 收集工具调用（可能分片到达）
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        _merge_tool_calls(collected_tool_calls, tool_calls)

                    if obj.get("done"):
                        llm_circuit_breaker.record_success()
                        safe_usage_update(obs, output=collected_content, done_obj=obj)
                        break
        except httpx.HTTPError as e:
            llm_circuit_breaker.record_failure()
            logger.error("[Stream] LLM 调用失败 | session=%s | error=%s", session_id, e)
            yield f"⚠️ AI 服务暂时不可用，请稍后重试。({type(e).__name__})"
            return
        finally:
            end_observation(obs)

        normalized_tool_calls = _normalize_tool_calls(collected_tool_calls)

        # 3. 需要执行工具 → 追加 assistant 消息、执行工具、进入下一轮
        if normalized_tool_calls:
            messages.append(_build_assistant_tool_message(collected_content, normalized_tool_calls))

            for tc in normalized_tool_calls:
                logger.info(
                    "[Stream] 工具调用 | session=%s | iter=%d | tool=%s",
                    session_id, iteration, tc["name"],
                )
                result = await _execute_tool(tc["name"], tc["arguments"], user_id=user_id)
                messages.append({
                    "role": "tool",
                    "content": result,
                })

            messages = trim_messages(messages, max_tokens=rag_config.LLM_MAX_TOKENS)
            continue

        # 4. 无工具调用 → 流式输出已完成，正常退出
        logger.info(
            "💬 [Stream] 最终回答完成 | session=%s | iter=%d | len=%d",
            session_id, iteration, len(collected_content),
        )
        return

    logger.warning("⚠️ [Stream] 达到最大推理轮次 | session=%s", session_id)
    yield "⚠️ 达到最大推理轮次，请简化问题重试。"


async def agent_run(
    user_id: str,
    session_id: str,
    user_message: str,
    collection: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    stream: bool = False,
) -> Union[Tuple[str, str], AsyncGenerator[str, None]]:
    """
    Agent 核心入口。处理记忆注入、LLM 交互及工具调度。

    Args:
        user_id: 用户长期身份ID（用于跨会话记忆检索与写入）。
        session_id: 会话唯一标识（用于日志追踪与短期上下文）。
        user_message: 用户当前输入。
        collection: (预留) 向量数据库集合对象。
        embed_model: (预留) 嵌入模型对象。
        stream: 是否启用流式输出。

    Returns:
        非流式: Tuple[最终回答文本, 检索到的知识文本]
        流式: AsyncGenerator[str, None]
    """
    logger.info(
        "🤖 Agent Run 开始 | user=%s | session=%s | model=%s | stream=%s | msg=%s",
        user_id, session_id, rag_config.OPENAI_MODEL, stream, user_message[:80],
    )

    # 1. 检索长期记忆并动态拼接 System Prompt
    system_content = SYSTEM_PROMPT
    try:
        memories = await retrieve_memories(user_id=user_id, query=user_message, top_k=3)
        if memories:
            memory_str = "\n".join(f"- {m}" for m in memories)
            system_content += f"\n\n{MEMORY_INJECTION_MARKER}\n{memory_str}"
            logger.info("🧠 [Memory] 注入 %d 条记忆 | user=%s | session=%s", len(memories), user_id, session_id)
        else:
            logger.debug("🧠 [Memory] 无相关记忆 | user=%s | session=%s", user_id, session_id)
    except Exception as e:
        logger.exception("⚠️ [Memory] 检索失败，降级使用基础 Prompt | user=%s | session=%s", user_id, session_id)

    # 2. 构建初始消息
    # 预检索：不依赖模型工具调用，先把知识库结果注入上下文
    try:
        _ctx = await _execute_tool("knowledge_search", {"query": user_message})
        if _ctx and _ctx.strip() and "检索失败" not in _ctx:
            if _ctx == "No relevant information found in knowledge base.":
                system_content += "\n\n【知识库检索结果】知识库中未找到与用户问题相关的信息。"
            else:
                system_content += ("\n\n【知识库检索结果】以下内容可能包含与用户问题相关的信息：\n" + _ctx + "\n（若上述内容与问题无关，请忽略；若相关，请严格依据其回答并标注来源编号）")
            logger.info("[RAG] 知识库预检索注入 | len=%d | user=%s", len(_ctx), user_id)
    except Exception as e:
        logger.warning("[RAG] 知识库预检索失败 | user=%s | error=%s", user_id, e)

    messages = _build_initial_messages(system_content, user_message)

    # 3. 路由至流式或非流式处理
    if stream:
        return _stream_tool_loop(messages, session_id, user_id=user_id)

    # ====== 非流式模式 ======
    retrieved_chunks: List[str] = []

    for iteration in range(MAX_AGENT_ITERATIONS):
        tool_calls = None
        collected_content = ""
        if not llm_circuit_breaker.allow():
            return "⚠️ 服务繁忙（熔断中），请稍后重试。", ""
        obs = start_observation("llm_call", "generation", model=rag_config.OPENAI_MODEL, input=messages)
        try:
            async with _http_client.stream(
                "POST",
                "/api/chat",
                json=_build_ollama_payload(messages, tools=TOOL_SCHEMAS, stream=False),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message", {}) or {}
                    collected_content += msg.get("content") or ""
                    if msg.get("tool_calls"):
                        tool_calls = msg.get("tool_calls")
                    if obj.get("done"):
                        llm_circuit_breaker.record_success()
                        safe_usage_update(obs, output=collected_content, done_obj=obj)
                        break
        except httpx.HTTPError as e:
            llm_circuit_breaker.record_failure()
            logger.error("[Agent] LLM 调用失败 | session=%s | error=%s", session_id, e)
            return f"⚠️ AI 服务暂时不可用，请稍后重试。({type(e).__name__})", ""
        finally:
            end_observation(obs)

        normalized_tool_calls = _normalize_tool_calls(tool_calls)

        if not normalized_tool_calls:
            final_content = collected_content.strip()
            logger.info(
                "💬 [Agent] 最终回答 | session=%s | iter=%d | len=%d",
                session_id, iteration, len(final_content),
            )
            retrieved_knowledge = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""
            return final_content, retrieved_knowledge

        messages.append(_build_assistant_tool_message(collected_content, normalized_tool_calls))

        for tc in normalized_tool_calls:
            logger.info(
                "[Agent] 工具调用 | session=%s | iter=%d | tool=%s",
                session_id, iteration, tc["name"],
            )
            result = await _execute_tool(tc["name"], tc["arguments"], user_id=user_id)

            if tc["name"] == "knowledge_search" and result not in retrieved_chunks:
                retrieved_chunks.append(result)

            messages.append({
                "role": "tool",
                "content": result,
            })

        messages = trim_messages(messages, max_tokens=rag_config.LLM_MAX_TOKENS)

    logger.warning("⚠️ [Agent] 达到最大推理轮次 | session=%s", session_id)
    return "⚠️ 达到最大推理轮次，请简化问题重试。", ""


def get_available_tools() -> List[Dict[str, str]]:
    """获取当前注册的所有可用工具名称及描述"""
    return [
        {"name": t["function"]["name"], "description": t["function"]["description"]}
        for t in TOOL_SCHEMAS
    ]