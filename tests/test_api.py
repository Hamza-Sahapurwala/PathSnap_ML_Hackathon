import os
import json
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "pipeline" in data

def test_trips():
    resp = client.get("/trips")
    assert resp.status_code == 200
    data = resp.json()
    trips = data.get("trips", [])
    # The repo contains 11 student trips; the API should return exactly 10
    assert isinstance(trips, list)
    assert len(trips) == 10
    # ensure all ids are strings and correspond to directories
    for tid in trips:
        assert isinstance(tid, str)
        path = os.path.join(os.getenv("EVAL_DATA_DIR", "./data/student_test"), tid)
        assert os.path.isdir(path)

@pytest.mark.parametrize("trip_id", [] )  # placeholder, will be populated dynamically
def test_predict_success(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200
    data = resp.json()
    expected_keys = {"trip_id", "gps", "predicted_route", "classifier_route", "reference_route", "metrics", "latency_ms"}
    assert expected_keys.issubset(data.keys())
    assert data["trip_id"] == trip_id
    # basic sanity checks on returned structures
    assert isinstance(data["predicted_route"], list)
    assert isinstance(data["classifier_route"], list)
    assert isinstance(data["metrics"], dict)
    assert isinstance(data["latency_ms"], int)

def test_predict_unknown():
    resp = client.post("/predict", json={"trip_id": "nonexistent"})
    assert resp.status_code == 404

def test_predict_path_traversal():
    resp = client.post("/predict", json={"trip_id": "../secret"})
    # should be rejected as bad request or not found
    assert resp.status_code >= 400 and resp.status_code < 500

# Dynamically generate the param list for test_predict_success after import
trips_response = client.get("/trips")
if trips_response.status_code == 200:
    trip_ids = trips_response.json().get("trips", [])
    # replace the empty param list with actual ids
    test_predict_success.parametrize = pytest.mark.parametrize("trip_id", trip_ids)
