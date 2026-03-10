def test_health_endpoints(client):
    live = client.get("/api/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/api/health/ready")
    assert ready.status_code in {200, 503}
    payload = ready.json()
    assert "database" in payload
    assert "smtp" in payload
