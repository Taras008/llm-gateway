from app.llm.errors import LLMUnavailableError
from app.llm.factory import get_provider
from app.llm.fake import FakeProvider
from app.main import app
from app.request_log import get_request_log_repository


class FailingProvider:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def chat(self, message, *, system=None, json_schema=None):
        raise self.exc


class BrokenRepository:
    async def add(self, entry):
        raise RuntimeError("database is down")


def test_successful_chat_is_logged(client, log_repo):
    client.post("/chat", json={"message": "hello world"})

    assert len(log_repo.entries) == 1
    entry = log_repo.entries[0]
    assert entry.endpoint == "/chat"
    assert entry.status == "success"
    assert entry.error_type is None
    assert entry.model == "fake"
    assert entry.input_chars == len("hello world")
    assert entry.prompt_tokens == 2
    assert entry.completion_tokens == 4
    assert entry.latency_ms >= 0


def test_failed_call_is_logged_with_error_type(client, log_repo):
    app.dependency_overrides[get_provider] = lambda: FailingProvider(LLMUnavailableError("down"))

    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 503
    entry = log_repo.entries[0]
    assert entry.status == "error"
    assert entry.error_type == "LLMUnavailableError"
    assert entry.model is None
    assert entry.prompt_tokens is None


def test_invalid_structured_output_is_logged_with_spent_tokens(client, log_repo):
    app.dependency_overrides[get_provider] = lambda: FakeProvider(text="not json")

    response = client.post("/analyze", json={"text": "some review"})

    assert response.status_code == 502
    entry = log_repo.entries[0]
    assert entry.endpoint == "/analyze"
    assert entry.status == "error"
    assert entry.error_type == "LLMBadResponseError"
    assert entry.model == "fake"
    assert entry.completion_tokens == 4


def test_request_rejected_by_validation_is_not_logged(client, log_repo):
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422
    assert log_repo.entries == []


def test_log_failure_does_not_break_response(client):
    app.dependency_overrides[get_request_log_repository] = lambda: BrokenRepository()

    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == 200
