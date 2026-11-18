"""
tests/test_api.py  –  Integration tests for the PathSnap FastAPI service.

Uses FastAPI TestClient (no live server needed).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, EVAL_DATA_DIR

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trip_ids() -> list[str]:
    resp = client.get("/trips")
    assert resp.status_code == 200
    return resp.json()["trips"]


# ---------------------------------------------------------------------------
# 1. /health
# ---------------------------------------------------------------------------

def test_health_status():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["pipeline"] == "real"


# ---------------------------------------------------------------------------
# 2. /trips
# ---------------------------------------------------------------------------

def test_trips_returns_200():
    resp = client.get("/trips")
    assert resp.status_code == 200


def test_trips_returns_exactly_10():
    trips = _trip_ids()
    assert len(trips) == 10, f"Expected 10 trips, got {len(trips)}: {trips}"


def test_trips_are_strings():
    for tid in _trip_ids():
        assert isinstance(tid, str)


def test_trips_correspond_to_real_dirs():
    for tid in _trip_ids():
        d = EVAL_DATA_DIR / tid
        assert d.is_dir(), f"Trip directory missing: {d}"


# ---------------------------------------------------------------------------
# 3. /predict – parametrised over all 10 trips
# ---------------------------------------------------------------------------

TRIP_IDS = _trip_ids()
EXPECTED_KEYS = {"trip_id", "gps", "predicted_route", "classifier_route",
                 "reference_route", "metrics", "latency_ms"}
METRIC_KEYS   = {"top1_accuracy", "macro_f1", "route_continuity", "seq_f1"}


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_200(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200, f"trip {trip_id}: {resp.text}"


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_response_keys(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200
    data = resp.json()
    assert EXPECTED_KEYS.issubset(data.keys()), f"Missing keys: {EXPECTED_KEYS - data.keys()}"


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_metrics_in_range(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    for method in ("classifier", "hmm"):
        assert method in data["metrics"], f"Missing metrics.{method}"
        m = data["metrics"][method]
        for k in METRIC_KEYS:
            assert k in m, f"metrics.{method} missing key {k!r}"
            v = m[k]
            assert isinstance(v, (int, float)), f"metrics.{method}.{k} is not numeric"
            assert 0.0 <= v <= 1.0, f"metrics.{method}.{k} = {v} outside [0, 1]"


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_trip_id_echoed(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200
    assert resp.json()["trip_id"] == trip_id


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_latency_ms_is_positive_int(trip_id):
    resp = client.post("/predict", json={"trip_id": trip_id})
    assert resp.status_code == 200
    ms = resp.json()["latency_ms"]
    assert isinstance(ms, int) and ms >= 0


@pytest.mark.parametrize("trip_id", TRIP_IDS)
def test_predict_accuracy_is_measured_against_the_real_reference(trip_id):
    """Guards against comparing predictions with an all -1 reference (Top-1 would be 0.0)."""
    data = client.post("/predict", json={"trip_id": trip_id}).json()
    hmm, clf = data["metrics"]["hmm"], data["metrics"]["classifier"]
    assert hmm["top1_accuracy"] > 0.5, f"trip {trip_id}: HMM Top-1 {hmm['top1_accuracy']}"
    assert clf["top1_accuracy"] > 0.3, f"trip {trip_id}: classifier Top-1 {clf['top1_accuracy']}"
    assert data["predicted_route"] and data["reference_route"]


# ---------------------------------------------------------------------------
# 4. Error cases
# ---------------------------------------------------------------------------

def test_predict_unknown_trip_returns_404():
    resp = client.post("/predict", json={"trip_id": "definitely_nonexistent_trip_xyz"})
    assert resp.status_code == 404


def test_predict_path_traversal_rejected():
    for bad in ("../secret", "../../etc/passwd", "..", "a/b", "a\\b"):
        resp = client.post("/predict", json={"trip_id": bad})
        assert resp.status_code in (400, 404), (
            f"Expected 4xx for trip_id={bad!r}, got {resp.status_code}"
        )
