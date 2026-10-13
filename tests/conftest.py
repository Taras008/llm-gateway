import pytest
from fastapi.testclient import TestClient

from app.llm.factory import get_provider
from app.llm.fake import FakeProvider
from app.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
