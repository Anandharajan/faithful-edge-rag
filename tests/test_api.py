from fastapi.testclient import TestClient

from faithful_edge_rag.api.main import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_problem_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/research/problem")

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Open-Source Faithful Edge-Cloud RAG"
    assert "citation faithfulness" in payload["problem"]

