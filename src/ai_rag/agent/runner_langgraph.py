# src/ai_rag/agent/runner_langgraph.py
"""LangGraph 版 Agent 编排（与自研 runner.py 对照实现）

实验设计：只替换「调度循环」这一层 —— LLM 调用、提示词、工具实现全部复用
ai_rag.agent.runner，保证与自研版**公平对比**（变量只有编排方式）。

用法：.env 里设 RAG_AGENT_BACKEND=langgraph 即可切换到本实现。
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TypedDict, Union

from langgraph.graph import END, START, StateGraph

from ai_rag.agent import runner as base
from ai_rag.agent.tools import TOOL_SCHEMAS
from ai_rag.core.answer_guard import REFUSAL_MESSAGE, should_refuse
from ai_rag.core.circuit_breaker import llm_circuit_breaker
from ai_rag.core.config import rag_config
from ai_rag.core.observability import end_observation, safe_usage_update, start_observation
from ai_rag.services.rag_service import NO_RESULT_MSG
from ai_rag.utils.context_trimmer import trim_messages

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态（对照自研版里的局部变量）"""
    messages: List[Dict[str, Any]]
    iterations: int
    retrieved_chunks: List[str]
    kb_had_content: bool
    final_content: str
    user_id: str


async def _llm_node(state: AgentState) -> AgentState:
    """节点 1：调用 LLM（产出最终回答，或产出工具调用）"""
    messages = list(state.get("messages") or [])

    if not llm_circuit_breaker.allow():
        return {"final_content": "服务繁忙（熔断中），请稍后重试。"}

    collected = ""
    raw_tool_calls = None
    obs = start_observation("llm_call", "generation", model=rag_config.OPENAI_MODEL, input=messages)
    try:
        async with base._http_client.stream(
            "POST",
            "/api/chat",
            json=base._build_ollama_payload(messages, tools=TOOL_SCHEMAS, stream=False),
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
                collected += msg.get("content") or ""
                if msg.get("tool_calls"):
                    raw_tool_calls = msg.get("tool_calls")
                if obj.get("done"):
                    llm_circuit_breaker.record_success()
                    safe_usage_update(obs, output=collected, done_obj=obj)
                    break
    except Exception as e:
        llm_circuit_breaker.record_failure()
        logger.error("[LangGraph] LLM 调用失败 | error=%s", e)
        return {"final_content": f"AI 服务暂时不可用，请稍后重试。({type(e).__name__})"}
    finally:
        end_observation(obs)

    tool_calls = base._normalize_tool_calls(raw_tool_calls)

    if not tool_calls:
        return {"final_content": collected.strip(), "messages": messages}

    return {
        "messages": messages + [base._build_assistant_tool_message(collected, tool_calls)],
        "final_content": "",
    }


def _route_after_llm(state: AgentState) -> str:
    """条件边：LLM 之后往哪走"""
    messages = state.get("messages") or []
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
        return "tools"
    return END


async def _tools_node(state: AgentState) -> AgentState:
    """节点 2：执行工具（把结果按 tool 角色回灌进消息）"""
    messages = list(state.get("messages") or [])
    assistant_msg = messages[-1]
    raw_calls = assistant_msg.get("tool_calls") or []

    calls: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw_calls):
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""), "arguments": args})

    retrieved = list(state.get("retrieved_chunks") or [])
    kb_had = bool(state.get("kb_had_content"))
    user_id = state.get("user_id") or "anonymous"

    for call in calls:
        result = await base._execute_tool(call["name"], call["arguments"], user_id=user_id)
        if call["name"] == "knowledge_search":
            if result and result != NO_RESULT_MSG:
                kb_had = True
            if result not in retrieved:
                retrieved.append(result)
        messages.append({"role": "tool", "content": result, "tool_call_id": call["id"]})

    messages = trim_messages(messages, max_tokens=rag_config.LLM_MAX_TOKENS)

    return {
        "messages": messages,
        "iterations": int(state.get("iterations") or 0) + 1,
        "retrieved_chunks": retrieved,
        "kb_had_content": kb_had,
    }


def _route_after_tools(state: AgentState) -> str:
    """条件边：工具执行完是否继续循环（迭代上限兜底）"""
    if int(state.get("iterations") or 0) >= rag_config.MAX_AGENT_ITERATIONS:
        return END
    return "agent"


def build_graph():
    """编译图：agent → (tools → agent)* → END"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _llm_node)
    graph.add_node("tools", _tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_llm, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", END: END})
    return graph.compile()


_GRAPH = build_graph()


async def agent_run(
    user_id: str,
    session_id: str,
    user_message: str,
    collection: Optional[Any] = None,
    embed_model: Optional[Any] = None,
    stream: bool = False,
    history: Optional[List[Dict[str, Any]]] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> Union[Tuple[str, str], AsyncGenerator[str, None]]:
    """对外入口：与自研版 agent_run 保持同样的签名与返回"""
    logger.info(
        "[LangGraph] Agent Run 开始 | user=%s | session=%s | model=%s | stream=%s | msg=%s",
        user_id, session_id, rag_config.OPENAI_MODEL, stream, user_message[:80],
    )

    # ---- 1. 记忆注入 ----
    system_content = base.SYSTEM_PROMPT
    try:
        from ai_rag.agent.memory import retrieve_memories
        memories = await retrieve_memories(user_id=user_id, query=user_message, top_k=3)
        if memories:
            system_content += "\n\n" + base.MEMORY_INJECTION_MARKER + "\n" + "\n".join(f"- {m}" for m in memories)
    except Exception as e:
        logger.warning("[LangGraph] 记忆检索失败 | error=%s", e)

    # ---- 2. 预检索注入（先公司域，无结果回退个人域）----
    retrieved_chunks: List[str] = []
    kb_had_content = False
    try:
        from ai_rag.services.rag_service import knowledge_search_handler
        _ctx = await knowledge_search_handler(query=user_message, domain="company")
        if _ctx and _ctx.strip() and _ctx != NO_RESULT_MSG:
            system_content += "\n\n【企业知识域检索结果】\n" + _ctx
            kb_had_content = True
            retrieved_chunks.append(_ctx)
        else:
            _pctx = await knowledge_search_handler(query=user_message, domain="personal")
            if _pctx and _pctx.strip() and _pctx != NO_RESULT_MSG:
                system_content += "\n\n【个人学习域检索结果】\n" + _pctx
                kb_had_content = True
                retrieved_chunks.append(_pctx)
    except Exception as e:
        logger.warning("[LangGraph] 预检索失败 | error=%s", e)

    # ---- 3. 组装消息（含历史裁剪）----
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
    if history:
        recent = [h for h in history[-base.MAX_HISTORY_MESSAGES:]
                  if isinstance(h, dict) and h.get("role") in ("user", "assistant", "tool")]
        for h in base._trim_history(recent):
            messages.append({"role": h["role"], "content": str(h.get("content", ""))})
    messages.append({"role": "user", "content": user_message})

    # ---- 4. 跑图 ----
    final_state = await _GRAPH.ainvoke({
        "messages": messages,
        "iterations": 0,
        "retrieved_chunks": retrieved_chunks,
        "kb_had_content": kb_had_content,
        "user_id": user_id,
    })

    answer = (final_state.get("final_content") or "").strip()

    if not answer and int(final_state.get("iterations") or 0) >= rag_config.MAX_AGENT_ITERATIONS:
        answer = "达到最大推理轮次，请简化问题重试。"

    # ---- 5. 防幻觉兜底（与自研版一致，保证对比公平）----
    if should_refuse(answer, bool(final_state.get("kb_had_content")), False, query=user_message):
        logger.warning("[LangGraph] 幻觉兜底拦截 | session=%s", session_id)
        answer = REFUSAL_MESSAGE

    knowledge = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""

    if tool_trace is not None:
        for m in (final_state.get("messages") or []):
            if m.get("role") == "tool":
                tool_trace.append({"role": "tool", "content": m.get("content", ""),
                                   "tool_call_id": m.get("tool_call_id", "")})

    # ---- 6. 流式：本对照版先整段返回（token 级流式留作后续优化，也是对比项）----
    if stream:
        async def _gen():
            if answer:
                yield answer
        return _gen()

    return answer, knowledge
