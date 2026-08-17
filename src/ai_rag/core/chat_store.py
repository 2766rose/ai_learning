# -*- coding: utf-8 -*-
"""多轮对话存储：会话 + 消息（SQLite via SQLModel）"""
import uuid
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Optional

from sqlmodel import SQLModel, Field, Session, create_engine, select

from ai_rag.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DB_DIR = PROJECT_ROOT / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_URL = f"sqlite:///{_DB_DIR / 'chat_history.db'}"
engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})


class Conversation(SQLModel, table=True):
    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(index=True)
    title: str = Field(default="新对话")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Message(SQLModel, table=True):
    id: int = Field(primary_key=True)
    conversation_id: str = Field(index=True, foreign_key="conversation.id")
    role: str
    content: str = Field(default="")
    tool_calls: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def init_db():
    SQLModel.metadata.create_all(engine)


def cleanup_old_conversations(days: int = 30) -> int:
    """Delete conversations not updated within `days` days (with their messages)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = 0
    with Session(engine) as s:
        stale = list(s.exec(select(Conversation).where(Conversation.updated_at < cutoff)))
        for conv in stale:
            for m in s.exec(select(Message).where(Message.conversation_id == conv.id)):
                s.delete(m)
            s.delete(conv)
            deleted += 1
        s.commit()
    if deleted:
        logger.info("[ChatStore] cleanup old conversations: %d", deleted)
    return deleted


def create_conversation(user_id: str, title: str = "新对话") -> Conversation:
    try:
        cleanup_old_conversations(days=30)
    except Exception as e:
        logger.warning("[ChatStore] auto cleanup failed: %s", e)
    with Session(engine) as s:
        conv = Conversation(user_id=user_id, title=title)
        s.add(conv); s.commit(); s.refresh(conv)
        return conv


def list_conversations(user_id: str, limit: int = 50) -> List[Conversation]:
    with Session(engine) as s:
        stmt = (select(Conversation).where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc()).limit(limit))
        return list(s.exec(stmt))


def get_conversation(conv_id: str) -> Optional[Conversation]:
    with Session(engine) as s:
        return s.get(Conversation, conv_id)


def rename_conversation(conv_id: str, title: str) -> Optional[Conversation]:
    with Session(engine) as s:
        conv = s.get(Conversation, conv_id)
        if conv:
            conv.title = title
            s.add(conv); s.commit(); s.refresh(conv)
        return conv


def delete_conversation(conv_id: str) -> bool:
    with Session(engine) as s:
        conv = s.get(Conversation, conv_id)
        if not conv:
            return False
        for m in s.exec(select(Message).where(Message.conversation_id == conv_id)):
            s.delete(m)
        s.delete(conv)
        s.commit()
        return True


def list_messages(conv_id: str) -> List[Message]:
    with Session(engine) as s:
        stmt = (select(Message).where(Message.conversation_id == conv_id)
                .order_by(Message.id.asc()))
        return list(s.exec(stmt))


def add_message(conv_id: str, role: str, content: str,
                tool_calls: Optional[str] = None, tool_call_id: Optional[str] = None) -> Message:
    with Session(engine) as s:
        m = Message(conversation_id=conv_id, role=role, content=content,
                    tool_calls=tool_calls, tool_call_id=tool_call_id)
        s.add(m)
        conv = s.get(Conversation, conv_id)
        if conv:
            conv.updated_at = datetime.now(timezone.utc)
            s.add(conv)
        s.commit(); s.refresh(m)
        return m


init_db()
