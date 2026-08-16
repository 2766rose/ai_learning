# src/ai_rag/agent/runner.py
import json
import logging
from typing import AsyncGenerator, Union, List, Dict, Any, Tuple

from openai import AsyncOpenAI

from src.ai_rag.core.config import rag_config
from src.ai_rag.services.rag_service import knowledge_search_handler

logger = logging.getLogger(__name__)

llm_client = AsyncOpenAI(
    api_key=rag_config.OPENAI_API_KEY,
    base_url=rag_config.OPENAI_BASE_URL,
    timeout=rag_config.LLM_TIMEOUT,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "从企业知识库中检索与用户问题相关的文档片段。"
                "当用户询问公司内部制度、产品参数、业务数据、历史文档时必须调用此工具。"
                "返回带相似度评分和来源编号的文本片段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词或完整问题，应包含足够的语义信息以提高召回率"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """你是企业知识库助手，严格遵守以下规则：
1. 回答任何知识类问题前，必须先调用 knowledge_search 工具检索
2. 仅依据工具返回的内容作答，禁止编造或推测信息
3. 引用内容时使用来源编号标注，如 [1][2]
4. 若工具返回 "No relevant information found"，如实告知用户："抱歉，知识库中未找到与您问题相关的信息。"
5. 不要向用户提及工具名称、检索过程等内部实现细节"""


async def _execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    if tool_name == "knowledge_search":
        result = await knowledge_search_handler(query=arguments.get("query", ""))
        logger.info(
            "[Tool] %s executed | input_query='%s' | result_len=%d",
            tool_name, arguments.get("query", "")[:80], len(result),
        )
        return result
    logger.warning("[Tool] Unknown tool called: %s", tool_name)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


async def agent_run(
    session_id: str,
    user_message: str,
    collection=None,
    embed_model=None,
    stream: bool = False,
) -> Union[Tuple[str, str], AsyncGenerator[str, None]]:
    """
    Returns:
        stream=False: (ai_answer, retrieved_knowledge) 元组
        stream=True:  AsyncGenerator（流式场景暂不返回检索内容）
    """
    logger.info(
        "🤖 Agent Run | session=%s | model=%s | stream=%s | msg=%s",
        session_id, rag_config.OPENAI_MODEL, stream, user_message[:80]
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # ✅ 新增：收集所有工具返回的检索内容
    retrieved_chunks: List[str] = []

    max_iterations = 5
    for iteration in range(max_iterations):
        response = await llm_client.chat.completions.create(
            model=rag_config.OPENAI_MODEL,
            temperature=rag_config.LLM_TEMPERATURE,
            max_tokens=rag_config.LLM_MAX_TOKENS,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            final_content = msg.content or ""
            logger.info(
                "💬 Agent Final Answer | session=%s | iteration=%d | len=%d",
                session_id, iteration, len(final_content),
            )
            # ✅ 拼接所有检索内容为统一字符串
            retrieved_knowledge = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""

            if stream:
                async def _gen():
                    yield final_content
                return _gen()

            # ✅ 非流式返回元组
            return final_content, retrieved_knowledge

        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.error("[Agent] Invalid tool arguments: %s", tc.function.arguments)
                args = {}

            logger.info(
                "[Agent] Tool Call | session=%s | iteration=%d | tool=%s | args=%s",
                session_id, iteration, tc.function.name, args,
            )
            result = await _execute_tool(tc.function.name, args)

            # ✅ 收集 knowledge_search 的返回内容
            if tc.function.name == "knowledge_search":
                retrieved_chunks.append(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

    fallback = "⚠️ 达到最大推理轮次，请简化问题重试。"
    logger.warning("⚠️ Max iterations reached | session=%s", session_id)
    if stream:
        async def _gen():
            yield fallback
        return _gen()
    return fallback, ""


def get_available_tools() -> list[dict]:
    return [
        {"name": t["function"]["name"], "description": t["function"]["description"]}
        for t in TOOL_SCHEMAS
    ]