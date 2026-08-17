# -*- coding: utf-8 -*-
"""会话管理 API：列表 / 新建 / 历史 / 删除 / 重命名"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai_rag.core.chat_store import (
    create_conversation, list_conversations, get_conversation,
    rename_conversation, delete_conversation, list_messages,
)

router = APIRouter()


class CreateConvReq(BaseModel):
    user_id: str
    title: str = "新对话"


class RenameReq(BaseModel):
    title: str


@router.get("/conversations")
async def api_list_conversations(user_id: str, limit: int = 50):
    convs = list_conversations(user_id, limit)
    return {"status": "success", "conversations": [c.model_dump() for c in convs]}


@router.post("/conversations")
async def api_create_conversation(req: CreateConvReq):
    conv = create_conversation(req.user_id, req.title)
    return {"status": "success", "conversation": conv.model_dump()}


@router.get("/conversations/{conv_id}/messages")
async def api_get_messages(conv_id: str):
    if not get_conversation(conv_id):
        raise HTTPException(404, "会话不存在")
    msgs = list_messages(conv_id)
    return {"status": "success", "messages": [m.model_dump() for m in msgs]}


@router.patch("/conversations/{conv_id}")
async def api_rename(conv_id: str, req: RenameReq):
    conv = rename_conversation(conv_id, req.title)
    if not conv:
        raise HTTPException(404, "会话不存在")
    return {"status": "success", "conversation": conv.model_dump()}


@router.delete("/conversations/{conv_id}")
async def api_delete(conv_id: str):
    ok = delete_conversation(conv_id)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"status": "success"}
