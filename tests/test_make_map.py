"""Tests for src/make_map.py (Member C). Synthetic trips only; nothing here is an experimental result."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hmm import RoadGraph  # noqa: E402
from src.make_map import build_trip_map, nearby_arcs, route_lines  # noqa: E402

LON0, LAT0 = 8.68, 49.41
M_PER_DEG_LON, M_PER_DEG_LAT = 111_320 * np.cos(np.radians(LAT0)), 110_540


def synthetic_trip():
    """A 4-arc one-way line (nodes 0..4, 100 m apart) plus a far-away arc 4 (nodes 5 -> 6, 5 km north)."""
    x = np.array([0, 100, 200, 300, 400, 0, 100], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 5000, 5000], dtype=float)
    tx = np.array([50, 150, 250, 350], dtype=float)
    ty = np.array([3, -2, 4, 1], dtype=float)
    t = np.arange(4, dtype=float) * 10
    return {"trip_id": "T1", "track_x": tx, "track_y": ty, "track_t": t,
            "track_raw": np.column_stack([LON0 + tx / M_PER_DEG_LON, LAT0 + ty / M_PER_DEG_LAT, t]),
            "nodes_x": x, "nodes_y": y,
            "nodes_raw": np.column_stack([LON0 + x / M_PER_DEG_LON, LAT0 + y / M_PER_DEG_LAT]),
            "arcs": np.array([(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)])}


def test_route_lines_split_where_arcs_do_not_meet():
    trip = synthetic_trip()
    g = RoadGraph.from_trip(trip)
    assert route_lines([0, 1, 2], g.u, g.v) == [[0, 1, 2, 3]]
    assert route_lines([0, 2], g.u, g.v) == [[0, 1], [2, 3]]
    assert route_lines([0, 4, 3], g.u, g.v) == [[0, 1], [5, 6], [3, 4]]
    assert route_lines([], g.u, g.v) == []


def test_network_filter_keeps_only_arcs_near_the_track():
    trip = synthetic_trip()
    g = RoadGraph.from_trip(trip)
    xy = np.column_stack([trip["track_x"], trip["track_y"]])
    assert nearby_arcs(xy, g, 250.0, 100).tolist() == [0, 1, 2, 3]   # far arc 4 dropped
    assert nearby_arcs(xy, g, 250.0, 2).size == 2                    # cap keeps the nearest
    assert nearby_arcs(np.full((2, 2), np.nan), g, 250.0, 100).size == 0


def test_map_has_every_layer_legend_and_control():
    trip = synthetic_trip()
    preds = {"classifier_top1": np.array([0, 4, 2, 3]), "hmm_tuned": np.array([0, 1, 2, 3])}
    fmap = build_trip_map(trip, preds, reference_route=[0, 1, 2, 3],
                          reference_point_arcs=np.array([0, 1, 2, 3]), notes=["note A"])
    page = fmap.get_root().render()
    for text in ["Road network (4 of 5 arcs", "Reference route", "Classifier-only route",
                 "HMM route (tuned)", "Raw GPS (4 points)", "Mismatches: Classifier-only (1)",
                 "Mismatches: HMM (tuned) (0)", "L.control.layers", "note A", "Trip T1", "t = 30 s"]:
        assert text in page, text


def test_map_without_reference_has_no_reference_layers():
    trip = synthetic_trip()
    fmap = build_trip_map(trip, {"hmm": np.array([0, 1, 2, 3])})
    page = fmap.get_root().render()
    assert "Reference route" not in page and "Mismatches" not in page
    assert "HMM route" in page


def test_prediction_length_must_match_the_track():
    with pytest.raises(ValueError, match="4 GPS points"):
        build_trip_map(synthetic_trip(), {"hmm": np.array([0, 1])})


def test_build_trip_map_never_reads_the_route_key():
    class Tripwire(dict):
        def __getitem__(self, key):
            if key == "route":
                raise AssertionError("make_map must not read trip['route']")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == "route":
                raise AssertionError("make_map must not read trip['route']")
            return super().get(key, default)

    trip = Tripwire(synthetic_trip(), route=np.array([0, 1, 2, 3]))
    build_trip_map(trip, {"hmm": np.array([0, 1, 2, 3])})
