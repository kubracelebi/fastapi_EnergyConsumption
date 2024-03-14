from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict_daily():
    response = client.post("/prediction/daily", json={
        'Tarih': '01.01.2023'

    })

    assert response.status_code == 200
    assert isinstance(response.json()['result'], str), 'Result wrong type!'