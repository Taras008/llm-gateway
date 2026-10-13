import pytest
from fastapi.testclient import TestClient

from app.llm.factory import get_provider
from app.llm.fake import FakeProvider
from app.main import app
from app.request_log import InMemoryRequestLogRepository, get_request_log_repository


@pytest.fixture
def log_repo() -> InMemoryRequestLogRepository:
    return InMemoryRequestLogRepository()


@pytest.fixture
def client(log_repo):
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_request_log_repository] = lambda: log_repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
