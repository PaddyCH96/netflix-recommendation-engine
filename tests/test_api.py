from fastapi.testclient import TestClient

from src.service.api import app


def test_recommend_endpoint_returns_recommendations():
    with TestClient(app) as client:
        response = client.post("/recommend", json={"user_id": 1})

    assert response.status_code == 200
    payload = response.json()
    assert "recommendations" in payload
    assert isinstance(payload["recommendations"], list)
