# src/ai_rag/api/rag_router.py
import os
import uuid
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import redis
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from src.ai_rag.tasks.celery_app import celery_app
from src.ai_rag.models.schemas import ChatRequest, ChatResponse, UploadResponse
from src.ai_rag.agent.runner import agent_run, get_available_tools
from src.ai_rag.core.observability import observe, start_observation, end_observation, safe_update_output
from src.ai_rag.core.semantic_cache import semantic_cache
from src.ai_rag.core.rate_limiter import rate_limiter
from src.ai_rag.core.embeddings import embedding_service
import asyncio
from src.ai_rag.core.config import rag_config
import src.ai_rag.tasks.document_tasks 

logger = logging.getLogger(__name__)
router = APIRouter()

_celery_publish_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="celery-pub")

# ✅ 启动时打印关键配置
logger.info("🔗 API Broker URL: %s", celery_app.conf.broker_url)
logger.info("📋 API Registered Tasks: %s", [t for t in celery_app.tasks.keys() if not t.startswith('celery.')])

_redis_client = redis.Redis.from_url(rag_config.CELERY_BROKER_URL)


def _get_queue_length(queue_name: str = "celery") -> int:
    try:
        return _redis_client.llen(queue_name)
    except Exception as e:
        logger.warning("⚠️ 获取队列长度失败: %s", e)
        return -1


# ==================== 1. 文件上传 ====================
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传文件 → 保存本地 → 投递 Celery 任务 → 立即返回"""
    try:
        task_id = str(uuid.uuid4())
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

        upload_dir = str(rag_config.UPLOAD_DIR)
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(upload_dir, f"{task_id}{ext}"))

        if ext not in rag_config.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type: " + str(ext))

        content = await file.read()
        if len(content) > rag_config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        if not content:
            raise HTTPException(status_code=400, detail="上传的文件内容为空")

        with open(file_path, "wb") as f:
            written = f.write(content)

        if written != len(content):
            os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail=f"文件写入不完整: expected {len(content)}, got {written}"
            )

        logger.info("📤 文件已保存: %s (%d bytes)", file_path, written)

        target_task_name = "ingest_document_task"
        if target_task_name not in celery_app.tasks:
            raise HTTPException(
                status_code=500,
                detail=f"任务 '{target_task_name}' 未在 Celery 中注册",
            )

        task_args = [file_path, {"source": file.filename, "task_id": task_id}]

        def _safe_send():
            conn = celery_app.connection_for_write()
            try:
                conn.ensure_connection(max_retries=3, interval_start=0.5, interval_max=2)
                result = celery_app.send_task(
                    name=target_task_name,
                    args=task_args,
                    task_id=task_id,
                    queue="celery",
                    connection=conn,
                )
                return result.id
            finally:
                conn.release()

        loop = asyncio.get_running_loop()
        try:
            result_id = await loop.run_in_executor(_celery_publish_executor, _safe_send)
        except Exception as e:
            logger.error("❌ Celery 投递异常: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"任务投递失败: {str(e)}")

        logger.info(
            "🚀 [SEND_TASK] 投递成功 | task_id=%s | file=%s",
            result_id, file_path,
        )

        return UploadResponse(
            status="processing",
            message="文件已接收，正在后台入库",
            task_id=result_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ 上传失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


# ==================== 2. 任务状态查询 ====================
@router.get("/upload/{task_id}/status")
async def get_upload_status(task_id: str):
    """轮询异步入库任务进度"""
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}

    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)

    return response


# ==================== 3. RAG 非流式问答 ====================
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, raw_request: Request):
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="messages 列表中必须包含至少一条 role='user' 的消息")

    session_id = request.session_id or f"sess-{uuid.uuid4().hex[:12]}"

    # 用户级限流
    _rl_key = request.user_id or (request.client.host if request.client else "anon")
    _ok, _retry = rate_limiter.allow(_rl_key)
    if not _ok:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {_retry} 秒后重试")

    # 语义缓存
    _loop = asyncio.get_running_loop()
    _q_emb = await _loop.run_in_executor(None, embedding_service.embed_query, user_message)
    _cached = semantic_cache.get(_q_emb)
    if _cached is not None:
        return ChatResponse(ai_answer=_cached, session_id=session_id, trace_id=str(uuid.uuid4()), retrieved_knowledge="")

    obs = start_observation("rag-chat", "agent", input={"question": user_message})
    try:
        result = await agent_run(
            session_id=session_id,
            user_message=user_message,
            stream=False,
            user_id=request.user_id or "anonymous",
        )

        if isinstance(result, tuple) and len(result) == 2:
            ai_answer, retrieved_knowledge = result
        else:
            ai_answer = str(result)
            retrieved_knowledge = ""

        resp = ChatResponse(
            ai_answer=ai_answer,
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            retrieved_knowledge=retrieved_knowledge,
        )
        semantic_cache.put(_q_emb, ai_answer)
        safe_update_output(obs, output=ai_answer)
        return resp
    finally:
        end_observation(obs)


# ==================== 4. RAG 流式问答 SSE ====================
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="messages 列表中必须包含至少一条 role='user' 的消息")

    session_id = request.session_id or f"sess-{uuid.uuid4().hex[:8]}"

    # 用户级限流
    _rl_key = request.user_id or (request.client.host if request.client else "anon")
    _ok, _retry = rate_limiter.allow(_rl_key)
    if not _ok:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {_retry} 秒后重试")

    # 语义缓存
    _loop = asyncio.get_running_loop()
    _q_emb = await _loop.run_in_executor(None, embedding_service.embed_query, user_message)
    _cached = semantic_cache.get(_q_emb)
    if _cached is not None:
        async def _cached_stream():
            yield f"data: {json.dumps({'type': 'chunk', 'data': _cached}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(_cached_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    generator = await agent_run(
        session_id=session_id,
        user_message=user_message,
        stream=True,
        user_id=request.user_id or "anonymous",
    )

    async def sse_wrapper():
        with observe("rag-chat-stream", as_type="agent", input={"question": user_message}) as obs:
            full_answer = ""
            try:
                # ✅ 防御性检查：确保 agent_run 返回的是异步生成器
                if not hasattr(generator, '__aiter__'):
                    error_payload = json.dumps(
                        {"type": "error", "message": "Internal: agent_run did not return an async generator"},
                        ensure_ascii=False
                    )
                    yield f"data: {error_payload}\n\n"
                    return

                async for chunk in generator:
                    full_answer += chunk
                    payload = json.dumps({"type": "chunk", "data": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            except Exception as e:
                logger.error("❌ [STREAM] SSE error: %s", e, exc_info=True)
                error_payload = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
                yield f"data: {error_payload}\n\n"

            semantic_cache.put(_q_emb, full_answer)
            safe_update_output(obs, output=full_answer)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        sse_wrapper(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ==================== 5. Agent 工具管理 ====================
@router.get("/agent/tools")
async def list_agent_tools():
    try:
        tools = get_available_tools()
        return {"status": "success", "tools": tools, "total": len(tools)}
    except Exception as e:
        logger.error("❌ 获取Agent工具列表失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/tools/{tool_name}/test")
async def test_agent_tool(tool_name: str, raw_request: Request, payload: dict = None):
    try:
        result = await agent_run(
            user_id="tool-test",
            session_id=f"tool-test-{uuid.uuid4().hex[:8]}",
            user_message=f"[SYSTEM] 直接调用工具 {tool_name}，参数: {json.dumps(payload or {})}",
            stream=False,
        )
        return {
            "status": "success",
            "tool_name": tool_name,
            "result": result if isinstance(result, (str, dict, list)) else str(result),
        }
    except Exception as e:
        logger.error("❌ 工具测试失败 [%s]: %s", tool_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")