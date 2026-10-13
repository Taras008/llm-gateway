from app.llm.factory import get_provider
from app.llm.fake import FakeProvider
from app.main import app


def use_fake_answer(text: str) -> None:
    app.dependency_overrides[get_provider] = lambda: FakeProvider(text=text)


def test_analyze_returns_parsed_analysis(client):
    use_fake_answer(
        '{"sentiment": "negative", "topics": ["delivery"], "summary": "Late delivery."}'
    )

    response = client.post("/analyze", json={"text": "Доставка запізнилась на тиждень"})

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["sentiment"] == "negative"
    assert analysis["topics"] == ["delivery"]


def test_analyze_returns_502_when_llm_returns_not_json(client):
    use_fake_answer("Sure! Here is the analysis: it's negative")

    response = client.post("/analyze", json={"text": "some review"})

    assert response.status_code == 502


def test_analyze_returns_502_when_sentiment_is_not_allowed(client):
    use_fake_answer('{"sentiment": "angry", "topics": [], "summary": "Angry."}')

    response = client.post("/analyze", json={"text": "some review"})

    assert response.status_code == 502
