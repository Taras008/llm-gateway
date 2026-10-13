import pytest

from app.llm import factory


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_returns_answer_from_provider(client):
    response = client.post("/chat", json={"message": "hello world"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer to: hello world"
    assert body["model"] == "fake"
    assert body["prompt_tokens"] == 2
    assert body["latency_ms"] >= 0


def test_chat_rejects_empty_message(client):
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_too_long_message(client):
    response = client.post("/chat", json={"message": "a" * 4001})

    assert response.status_code == 422


def test_chat_rejects_missing_field(client):
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(factory, "LLM_PROVIDER", "openaii")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        factory.get_provider()