# -*- coding: utf-8 -*-
"""API tests: conversation ownership (IDOR) using a temp SQLite DB."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

from ai_rag.core import chat_store
from ai_rag.api.conversation_router import router


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "test_chat.db"
    chat_store.engine = create_engine("sqlite:///%s" % db, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(chat_store.engine)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c
    chat_store.engine.dispose()


def _create(client, user):
    r = client.post("/conversations", json={"user_id": user, "title": "t"})
    assert r.status_code == 200
    return r.json()["conversation"]["id"]


class TestConversationOwnership:
    def test_owner_reads_own_messages(self, client):
        cid = _create(client, "userA")
        r = client.get("/conversations/%s/messages?user_id=userA" % cid)
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_other_user_cannot_read(self, client):
        cid = _create(client, "userA")
        assert client.get("/conversations/%s/messages?user_id=userB" % cid).status_code == 404

    def test_other_user_cannot_delete(self, client):
        cid = _create(client, "userA")
        assert client.delete("/conversations/%s?user_id=userB" % cid).status_code == 404
        assert client.get("/conversations/%s/messages?user_id=userA" % cid).status_code == 200

    def test_owner_deletes_own(self, client):
        cid = _create(client, "userA")
        assert client.delete("/conversations/%s?user_id=userA" % cid).status_code == 200
        assert client.get("/conversations/%s/messages?user_id=userA" % cid).status_code == 404

    def test_other_user_cannot_rename(self, client):
        cid = _create(client, "userA")
        assert client.patch("/conversations/%s?user_id=userB" % cid, json={"title": "hack"}).status_code == 404

    def test_list_scoped_by_user(self, client):
        _create(client, "userA")
        _create(client, "userB")
        r = client.get("/conversations?user_id=userA")
        assert r.status_code == 200
        convs = r.json()["conversations"]
        assert len(convs) == 1
        assert all(c["user_id"] == "userA" for c in convs)
