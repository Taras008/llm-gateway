from app.llm.errors import LLMUnavailableError
from app.llm.factory import get_provider
from app.main import app


class FailingStreamProvider:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def chat(self, message, *, system=None, json_schema=None):
        raise self.exc

    async def stream(self, message, *, system=None):
        raise self.exc
        yield  # робить метод асинхронним генератором


class BrokenMidStreamProvider:
    async def chat(self, message, *, system=None, json_schema=None):
        raise NotImplementedError

    async def stream(self, message, *, system=None):
        from app.llm.base import LLMStreamChunk

        yield LLMStreamChunk(text="partial ")
        raise LLMUnavailableError("connection lost")


def test_stream_returns_full_text(client):
    response = client.post("/chat/stream", json={"message": "hello world"})

    assert response.status_code == 200
    assert response.text == "fake answer to: hello world"
    assert response.headers["content-type"].startswith("text/plain")


def test_stream_is_logged_with_tokens_after_completion(client, log_repo):
    client.post("/chat/stream", json={"message": "hello world"})

    entry = log_repo.entries[0]
    assert entry.endpoint == "/chat/stream"
    assert entry.status == "success"
    assert entry.model == "fake"
    assert entry.completion_tokens == 4


def test_stream_returns_503_when_provider_is_down_before_first_chunk(client, log_repo):
    app.dependency_overrides[get_provider] = lambda: FailingStreamProvider(
        LLMUnavailableError("down")
    )

    response = client.post("/chat/stream", json={"message": "hi"})

    assert response.status_code == 503
    assert log_repo.entries[0].error_type == "LLMUnavailableError"


def test_stream_interrupted_midway_is_logged_as_error(client, log_repo):
    app.dependency_overrides[get_provider] = lambda: BrokenMidStreamProvider()

    response = client.post("/chat/stream", json={"message": "hi"})

    assert response.status_code == 200  # заголовки вже відправлено
    assert response.text == "partial "
    assert log_repo.entries[0].status == "error"
    assert log_repo.entries[0].error_type == "LLMUnavailableError"
