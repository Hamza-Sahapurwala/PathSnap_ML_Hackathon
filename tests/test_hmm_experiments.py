"""Tests for route filling (src/hmm.py) and the evaluation metrics in src/hmm_experiments.py.

Synthetic inputs verify the algorithms only; nothing here is an experimental result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hmm_experiments as ex  # noqa: E402
from src.hmm import HMMConfig, NO_ARC, RoadGraph, fill_route  # noqa: E402


def naive_levenshtein(a, b):
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1][-1]


def chain_graph():
    """0 -> 1 -> 2 -> 3 (arcs 0, 1, 2), a reverse arc 3 = (1 -> 0), and an island arc 4 = (4 -> 5)."""
    return RoadGraph([0, 100, 200, 300, 0, 100], [0, 0, 0, 0, 5000, 5000],
                     [(0, 1), (1, 2), (2, 3), (1, 0), (4, 5)])


# ----------------------------------------------------------------------------- edit distance
def test_levenshtein_known_cases():
    kitten, sitting = [ord(c) for c in "kitten"], [ord(c) for c in "sitting"]
    assert ex.levenshtein(kitten, sitting) == 3
    assert ex.levenshtein([], [1, 2, 3]) == 3
    assert ex.levenshtein([1, 2], []) == 2
    assert ex.levenshtein([5, 6, 7], [5, 6, 7]) == 0


def test_levenshtein_matches_naive_dp_on_random_sequences():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = rng.integers(0, 4, rng.integers(0, 12)).tolist()
        b = rng.integers(0, 4, rng.integers(0, 12)).tolist()
        assert ex.levenshtein(a, b) == naive_levenshtein(a, b)


# ----------------------------------------------------------------------------- sequence metrics
def test_sequence_metrics_count_repeats_as_a_multiset():
    m = ex.sequence_metrics([1, 2, 1], [1, 2, 3])
    assert m["n_seq_match"] == 2  # the second 1 has no partner in the reference
    assert m["seq_precision"] == pytest.approx(2 / 3) and m["seq_recall"] == pytest.approx(2 / 3)
    assert m["edit_distance"] == 1 and m["edit_similarity"] == pytest.approx(2 / 3)
    revisit = ex.sequence_metrics([1, 2, 1], [1, 2, 1])
    assert revisit["seq_f1"] == 1.0 and revisit["edit_distance"] == 0


def test_sequence_metrics_empty_prediction():
    m = ex.sequence_metrics([], [1, 2])
    assert np.isnan(m["seq_precision"]) and m["seq_recall"] == 0 and m["seq_f1"] == 0
    assert m["edit_similarity"] == 0


# ----------------------------------------------------------------------------- path + route filling
def test_shortest_path_arcs_matches_scipy_and_is_contiguous():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    rng = np.random.default_rng(5)
    n = 150
    xs, ys = rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)
    arcs = sorted({(int(a), int(b)) for a, b in rng.integers(0, n, (600, 2)) if a != b})
    g = RoadGraph(xs, ys, arcs)
    dist = dijkstra(csr_matrix((g.length, (g.u, g.v)), shape=(n, n)), directed=True)
    checked = 0
    for s, t in rng.integers(0, n, (300, 2)):
        path = g.shortest_path_arcs(int(s), int(t), radius=1500.0)
        if s == t:
            assert path == []
            continue
        if dist[s, t] > 1500.0:
            assert path is None
            continue
        assert path is not None
        assert g.u[path[0]] == s and g.v[path[-1]] == t
        assert all(g.v[a] == g.u[b] for a, b in zip(path, path[1:]))
        assert g.length[path].sum() == pytest.approx(dist[s, t])
        checked += 1
    assert checked > 50


def test_fill_route_inserts_missing_arcs_and_counts_breaks():
    g = chain_graph()
    assert fill_route([0, 2], g) == ([0, 1, 2], 0)          # one arc skipped between observations
    assert fill_route([0, 1, 2], g) == ([0, 1, 2], 0)       # already connected: unchanged
    assert fill_route([0, 3], g) == ([0, 3], 0)             # U-turn is a direct connection
    assert fill_route([0, 4], g) == ([0, 4], 1)             # island: kept adjacent, one break
    assert fill_route([2, 0], g) == ([2, 0], 1)             # one-way network: no path back
    assert fill_route([], g) == ([], 0)


def test_fill_route_respects_the_distance_bound():
    g = chain_graph()
    assert fill_route([0, 2], g, max_gap_m=50.0) == ([0, 2], 1)


# ----------------------------------------------------------------------------- point + continuity metrics
def tiny_frame():
    # point 0: reference arc 0 among candidates; point 1: reference 1 among candidates;
    # point 2: reference arc 2 NOT among candidates (candidate miss)
    return pd.DataFrame({
        "point_idx":        [0, 0, 1, 1, 2],
        "arc_id":           [0, 3, 1, 4, 4],
        "proj_frac":        [0.5, 0.5, 0.5, 0.5, 0.5],
        "reference_arc_id": [0, 0, 1, 1, 2],
    })


def test_point_metrics_counts_candidate_misses_as_errors():
    m = ex.point_metrics(np.array([0, 4, 4]), tiny_frame())
    assert m["n_eval_points"] == 3 and m["n_correct"] == 1
    assert m["top1_accuracy"] == pytest.approx(1 / 3)
    assert m["accuracy_when_reference_in_candidates"] == pytest.approx(1 / 2)
    assert m["candidate_miss_rate"] == pytest.approx(1 / 3)
    # chosen rows: (0,0)=TP, (1,4)=FP, (2,4)=FP; unchosen: (0,3)=TN, (1,1)=FN
    from sklearn.metrics import f1_score
    expected = f1_score([1, 0, 1, 0, 0], [1, 0, 0, 1, 1], average="macro")
    assert m["selection_macro_f1"] == pytest.approx(expected)


def test_continuity_direct_vs_connected():
    g = chain_graph()
    frame = pd.DataFrame({"point_idx": [0, 1, 2, 3], "arc_id": [0, 2, 2, 4], "proj_frac": [0.5] * 4})
    trip = {"track_x": [50, 250, 260, 270], "track_y": [0, 0, 0, 0], "track_t": [0, 10, 11, 12]}
    m = ex.continuity_metrics(np.array([0, 2, 2, 4]), frame, trip, g, HMMConfig())
    # pairs: 0->2 multi-hop (connected, not direct), 2->2 same, 2->4 island (break)
    assert m["n_pairs"] == 3 and m["n_direct"] == 1 and m["n_connected"] == 2 and m["n_breaks"] == 1


def test_continuity_skips_unpredicted_points():
    g = chain_graph()
    frame = pd.DataFrame({"point_idx": [0, 1, 2], "arc_id": [0, 1, 2], "proj_frac": [0.5] * 3})
    trip = {"track_x": [50, 150, 250], "track_y": [0, 0, 0], "track_t": [0, 1, 2]}
    m = ex.continuity_metrics(np.array([0, NO_ARC, 2]), frame, trip, g, HMMConfig())
    assert m["n_pairs"] == 0 and np.isnan(m["continuity_direct"])


# ----------------------------------------------------------------------------- aggregation
def test_pooled_aggregate_rebuilds_ratios_from_counts():
    rows = []
    for tid, n, c in [("a", 10, 9), ("b", 90, 45)]:
        row = {"trip_id": tid, "method": "hmm", "n_eval_points": n, "n_correct": c,
               "n_reference_in_candidates": n, "top1_accuracy": c / n, "selection_macro_f1": 0.5,
               "n_direct": 1, "n_connected": 1, "n_pairs": 1}
        for p in ("raw", "filled"):
            row.update({f"{p}_n_seq_match": 1, f"{p}_len_pred": 2, f"{p}_len_ref": 2,
                        f"{p}_edit_distance": 1, f"{p}_seq_precision": 0.5, f"{p}_seq_recall": 0.5})
        rows.append(row)
    agg = ex.aggregate(pd.DataFrame(rows)).set_index("trip_id")
    assert agg.loc["ALL_pooled", "top1_accuracy"] == pytest.approx(54 / 100)
    assert agg.loc["ALL_trip_mean", "top1_accuracy"] == pytest.approx((0.9 + 0.5) / 2)
    assert np.isnan(agg.loc["ALL_pooled", "selection_macro_f1"])  # a ratio must never be summed
    assert agg.loc["ALL_pooled", "filled_edit_similarity"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- leakage guard
def test_evaluate_trip_loads_prediction_trip_without_route_and_reads_reference_last(monkeypatch):
    g = chain_graph()
    events = []

    def fake_load(data_dir, trip_id, load_route=True):
        events.append(("load", load_route))
        return {"trip_id": trip_id, "track_x": np.array([50.0, 150.0]), "track_y": np.zeros(2),
                "track_t": np.array([0.0, 1.0]), "nodes_x": g.nodes_x, "nodes_y": g.nodes_y,
                "arcs": np.stack([g.u, g.v], axis=1)}

    def fake_route(data_dir, trip_id):
        events.append(("route", None))
        return np.array([0, 1])

    monkeypatch.setattr(ex, "load_trip_data", fake_load)
    monkeypatch.setattr(ex, "read_reference_route", fake_route)
    frame = pd.DataFrame({"point_idx": [0, 0, 1], "arc_id": [0, 3, 1], "proj_frac": [0.5] * 3,
                          "reference_arc_id": [0, 0, 1]})
    rows, points = ex.evaluate_trip("t", frame, Path("."), {}, HMMConfig(),
                                    probabilities=np.array([0.9, 0.1, 0.8]))
    assert events == [("load", False), ("route", None)]
    assert [r["method"] for r in rows] == ["classifier_top1", "hmm"]
    assert rows[1]["top1_accuracy"] == 1.0 and rows[1]["filled_seq_f1"] == 1.0
    assert points.hmm_arc_id.tolist() == [0, 1]
