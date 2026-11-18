"""
tests/test_pipeline.py  –  Smoke tests for src/pipeline.predict_trip.

Runs predict_trip on the first available student_test trip and asserts:
  - all three output lists exist
  - every list length equals the number of GPS points in the track
"""

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate one real student_test trip
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
_EVAL_DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", str(_BASE / "data" / "student_test")))


def _get_first_trip() -> Path:
    """Return the first trip directory that has .track, .nodes and .arcs files."""
    trips = sorted([d for d in _EVAL_DATA_DIR.iterdir() if d.is_dir()])
    for trip_dir in trips:
        tid = trip_dir.name
        if all((trip_dir / f"{tid}{ext}").exists() for ext in (".track", ".nodes", ".arcs")):
            return trip_dir
    pytest.skip(f"No complete trip directories (.track/.nodes/.arcs) found under {_EVAL_DATA_DIR}")


FIRST_TRIP = _get_first_trip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_trip_returns_correct_keys():
    from src.pipeline import predict_trip
    result = predict_trip(FIRST_TRIP)
    assert set(result.keys()) == {"point_arcs", "classifier_arcs", "hmm_arcs"}


def test_predict_trip_lengths_match_track():
    """All three output lists must have exactly N entries (one per GPS point)."""
    import numpy as np
    from src.pipeline import predict_trip

    # Load the track independently to get N
    trip_id = FIRST_TRIP.name
    track_file = FIRST_TRIP / f"{trip_id}.track"
    track = np.loadtxt(track_file, dtype=np.float64)
    N = len(track)

    result = predict_trip(FIRST_TRIP)

    assert len(result["point_arcs"]) == N, (
        f"point_arcs length {len(result['point_arcs'])} != track length {N}"
    )
    assert len(result["classifier_arcs"]) == N, (
        f"classifier_arcs length {len(result['classifier_arcs'])} != track length {N}"
    )
    assert len(result["hmm_arcs"]) == N, (
        f"hmm_arcs length {len(result['hmm_arcs'])} != track length {N}"
    )


def test_predict_trip_arc_ids_are_integers():
    from src.pipeline import predict_trip
    result = predict_trip(FIRST_TRIP)
    for key in ("point_arcs", "classifier_arcs", "hmm_arcs"):
        for v in result[key]:
            assert isinstance(v, (int, np.integer)), (
                f"{key} contains non-integer value {v!r}"
            )


def test_predict_trip_no_route_file_read(monkeypatch):
    """Ensure predict_trip never opens the .route file."""
    real_open = open
    opened_routes = []

    def patched_open(file, *args, **kwargs):
        if str(file).endswith(".route"):
            opened_routes.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    # Also patch np.loadtxt to intercept .route reads
    real_loadtxt = np.loadtxt
    loadtxt_routes = []

    def patched_loadtxt(fname, *args, **kwargs):
        if str(fname).endswith(".route"):
            loadtxt_routes.append(str(fname))
        return real_loadtxt(fname, *args, **kwargs)

    monkeypatch.setattr("numpy.loadtxt", patched_loadtxt)

    from src.pipeline import predict_trip
    predict_trip(FIRST_TRIP)

    assert not opened_routes, f".route file was opened: {opened_routes}"
    assert not loadtxt_routes, f".route file was read via np.loadtxt: {loadtxt_routes}"
