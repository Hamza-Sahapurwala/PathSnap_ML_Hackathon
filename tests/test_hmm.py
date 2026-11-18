"""Tests for src/hmm.py (Member C): transition model, Viterbi, graph search, edge cases.

Synthetic graphs are used for algorithm verification only; nothing here is an experimental result.
"""
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hmm import (  # noqa: E402
    KIND_DIRECT, KIND_DISCONNECTED, KIND_IMPLAUSIBLE, KIND_MULTIHOP, KIND_SAME, NO_ARC,
    HMMConfig, RoadGraph, _prepare_observations, positive_class_probabilities, predict_route,
    predict_route_for_trip, route_from_arcs, score_candidates, top1_predictions,
    transition_log_matrix, viterbi,
)

CFG = HMMConfig()


# ----------------------------------------------------------------------------- fixtures
def line_graph(n=5, spacing=100.0, bidirectional=False):
    """Nodes on the x axis. Forward arc i = (i -> i+1); reverse of arc i has id (n-1)+i."""
    xs = np.arange(n) * spacing
    arcs = [(i, i + 1) for i in range(n - 1)]
    if bidirectional:
        arcs += [(i + 1, i) for i in range(n - 1)]
    return RoadGraph(xs, np.zeros(n), arcs)


def island_graph():
    """One-way line (arcs 0..3) plus a disconnected island arc 4 far away."""
    xs = [0, 100, 200, 300, 400, 0, 100]
    ys = [0, 0, 0, 0, 0, 1000, 1000]
    return RoadGraph(xs, ys, [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)])


def cands(rows):
    """rows: iterable of (point_idx, arc_id, proj_frac, prob) -> (frame, probs)."""
    rows = list(rows)
    frame = pd.DataFrame({"point_idx": [r[0] for r in rows], "arc_id": [r[1] for r in rows],
                          "proj_frac": [r[2] for r in rows]})
    return frame, np.array([r[3] for r in rows], dtype=float)


def tm(graph, a, fa, b, fb, gps, dt=1.0, cfg=CFG):
    return transition_log_matrix(graph, np.array(a), np.array(fa), np.array(b), np.array(fb),
                                 gps, dt, cfg)


# ----------------------------------------------------------------------------- transition model
def test_same_arc_transition_matches_closed_form():
    lt, kind = tm(line_graph(), [0], [0.2], [0], [0.5], gps=30.0)
    assert kind[0, 0] == KIND_SAME
    assert lt[0, 0] == pytest.approx(-math.log(2 * CFG.beta_m))  # d_net == d_gps -> zero error


def test_direct_connection_transition():
    lt, kind = tm(line_graph(), [0], [0.8], [1], [0.2], gps=40.0)
    assert kind[0, 0] == KIND_DIRECT          # 20 m left on arc0 + 20 m into arc1
    assert lt[0, 0] == pytest.approx(-math.log(2 * CFG.beta_m))


def test_multi_hop_transition_uses_intermediate_arc():
    lt, kind = tm(line_graph(), [0], [0.5], [2], [0.5], gps=200.0, dt=10.0)
    assert kind[0, 0] == KIND_MULTIHOP        # 50 + 100 (arc1) + 50
    beta = CFG.beta_m + CFG.beta_gap_m_per_s * 9
    assert lt[0, 0] == pytest.approx(-math.log(2 * beta))


def test_disconnected_transition_gets_bounded_floor_not_minus_inf():
    lt, kind = tm(island_graph(), [0], [0.5], [4], [0.5], gps=100.0, dt=5.0)
    assert kind[0, 0] == KIND_DISCONNECTED
    assert lt[0, 0] == CFG.log_disconnect_penalty and np.isfinite(lt).all()


def test_directed_graph_is_not_treated_as_bidirectional():
    one_way, two_way = line_graph(bidirectional=False), line_graph(bidirectional=True)
    # arc1 (1->2) then the *previous* arc0 (0->1): needs travelling against the one-way arcs
    _, kind = tm(one_way, [1], [0.5], [0], [0.5], gps=100.0, dt=10.0)
    assert kind[0, 0] == KIND_DISCONNECTED
    lt, kind = tm(two_way, [1], [0.5], [0], [0.5], gps=100.0, dt=10.0)
    assert kind[0, 0] == KIND_MULTIHOP        # reachable via reverse arcs 5 (2->1) and 4 (1->0)
    beta = CFG.beta_m + CFG.beta_gap_m_per_s * 9
    assert lt[0, 0] == pytest.approx(-abs(300.0 - 100.0) / beta - math.log(2 * beta))


def test_gps_displacement_inconsistent_with_network_distance_is_penalised():
    g = line_graph()
    ok, _ = tm(g, [0], [0.5], [2], [0.5], gps=200.0, dt=10.0)
    bad, _ = tm(g, [0], [0.5], [2], [0.5], gps=20.0, dt=10.0)
    beta = CFG.beta_m + CFG.beta_gap_m_per_s * 9
    assert ok[0, 0] > bad[0, 0]
    assert ok[0, 0] - bad[0, 0] == pytest.approx(180.0 / beta)


def test_impossible_speed_is_implausible_not_connected():
    _, kind = tm(line_graph(), [0], [0.0], [3], [1.0], gps=400.0, dt=1.0)  # 400 m in 1 s
    assert kind[0, 0] == KIND_IMPLAUSIBLE


def test_uturn_penalty_applies_only_to_reverse_arc():
    g = line_graph(bidirectional=True)
    base, kind = tm(g, [0], [0.5], [4], [0.5], gps=0.0, dt=5.0)  # arc 4 is the reverse of arc 0
    pen, _ = tm(g, [0], [0.5], [4], [0.5], gps=0.0, dt=5.0, cfg=HMMConfig(uturn_log_penalty=-3.0))
    assert kind[0, 0] == KIND_DIRECT
    assert pen[0, 0] == pytest.approx(base[0, 0] - 3.0)
    same_arc, _ = tm(g, [0], [0.5], [0], [0.5], gps=0.0, dt=5.0, cfg=HMMConfig(uturn_log_penalty=-3.0))
    assert same_arc[0, 0] == pytest.approx(-math.log(2 * (CFG.beta_m + CFG.beta_gap_m_per_s * 4)))


def test_normalised_transition_rows_sum_to_one():
    g = line_graph(bidirectional=True)
    lt, _ = tm(g, [0, 1], [0.3, 0.6], [0, 1, 2, 5], [0.2, 0.5, 0.7, 0.1], gps=60.0, dt=3.0,
               cfg=HMMConfig(normalize_transitions=True))
    np.testing.assert_allclose(np.exp(lt).sum(axis=1), 1.0)


# ----------------------------------------------------------------------------- viterbi core
def brute_force_best(log_emit, log_trans):
    best = (-np.inf, None)
    for path in itertools.product(*[range(len(e)) for e in log_emit]):
        s = sum(log_emit[k][path[k]] for k in range(len(path)))
        s += sum(log_trans[k][path[k - 1], path[k]] for k in range(1, len(path)))
        if s > best[0]:
            best = (s, path)
    return best


def path_score(path, log_emit, log_trans):
    return (sum(log_emit[k][path[k]] for k in range(len(path)))
            + sum(log_trans[k][path[k - 1], path[k]] for k in range(1, len(path))))


@pytest.mark.parametrize("seed", range(40))
def test_viterbi_equals_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, size=rng.integers(1, 7))
    log_emit = [np.log(rng.uniform(1e-4, 1, s)) for s in sizes]
    log_trans = [None] + [np.log(rng.uniform(1e-6, 1, (sizes[k - 1], sizes[k])))
                          for k in range(1, len(sizes))]
    states, segment = viterbi(log_emit, lambda k: log_trans[k])
    best_score, _ = brute_force_best(log_emit, log_trans)
    assert path_score(states, log_emit, log_trans) == pytest.approx(best_score)
    assert (segment == 0).all()


def test_viterbi_reset_segments_are_decoded_independently():
    rng = np.random.default_rng(7)
    sizes = [3, 2, 4, 3, 2]
    log_emit = [np.log(rng.uniform(0.01, 1, s)) for s in sizes]
    trans = {k: np.log(rng.uniform(0.01, 1, (sizes[k - 1], sizes[k]))) for k in (1, 3, 4)}
    states, segment = viterbi(log_emit, lambda k: trans.get(k))  # reset into obs 2
    assert segment.tolist() == [0, 0, 1, 1, 1]
    left, _ = viterbi(log_emit[:2], lambda k: trans[k])
    right, _ = viterbi(log_emit[2:], lambda k: trans[k + 2])
    assert states.tolist() == left.tolist() + right.tolist()


def test_viterbi_all_minus_inf_step_starts_new_segment_instead_of_failing():
    log_emit = [np.log([0.5, 0.5]), np.log([0.9, 0.1])]
    states, segment = viterbi(log_emit, lambda k: np.full((2, 2), -np.inf))
    assert segment.tolist() == [0, 1] and states[1] == 0


def test_viterbi_rejects_malformed_inputs():
    with pytest.raises(ValueError):
        viterbi([np.zeros(2), np.zeros(2)], lambda k: np.zeros((3, 2)))
    with pytest.raises(ValueError):
        viterbi([np.zeros(2), np.zeros(2)], lambda k: np.full((2, 2), np.nan))
    with pytest.raises(ValueError):
        viterbi([np.zeros(2), np.array([np.nan, 0.0])], lambda k: np.zeros((2, 2)))


def test_viterbi_handles_variable_candidate_counts_and_empty_input():
    states, segment = viterbi([], lambda k: None)
    assert states.size == 0 and segment.size == 0
    log_emit = [np.log([1.0]), np.log([0.2, 0.7, 0.1]), np.log([0.5, 0.5])]
    trans = [None, np.zeros((1, 3)), np.zeros((3, 2))]
    states, _ = viterbi(log_emit, lambda k: trans[k])
    assert states.tolist() == [0, 1, 0]


@pytest.mark.parametrize("seed", range(15))
def test_full_decoder_matches_exhaustive_search_on_random_graphs(seed):
    rng = np.random.default_rng(100 + seed)
    xs, ys = rng.uniform(0, 300, 8), rng.uniform(0, 300, 8)
    pairs = [(a, b) for a in range(8) for b in range(8) if a != b]
    arcs = [pairs[i] for i in rng.choice(len(pairs), 16, replace=False)]
    graph = RoadGraph(xs, ys, arcs)
    cfg = HMMConfig(gap_reset_s=1e9, max_speed_mps=1e6)
    V = 5
    tx, ty, tt = rng.uniform(0, 300, V), rng.uniform(0, 300, V), np.arange(V) * 3.0
    rows, per_obs = [], []
    for k in range(V):
        chosen = sorted(rng.choice(16, rng.integers(1, 4), replace=False))
        per_obs.append([(int(a), float(rng.uniform()), float(rng.uniform(0.02, 0.98))) for a in chosen])
        rows += [(k, a, f, p) for a, f, p in per_obs[-1]]
    frame, probs = cands(rows)
    result = predict_route(tx, ty, tt, graph, frame, probs, cfg)

    log_emit = [np.log([max(p, cfg.prob_eps) for _, _, p in o]) for o in per_obs]
    log_trans = [None]
    for k in range(1, V):
        a = np.array([x[0] for x in per_obs[k - 1]]); b = np.array([x[0] for x in per_obs[k]])
        fa = np.array([x[1] for x in per_obs[k - 1]]); fb = np.array([x[1] for x in per_obs[k]])
        log_trans.append(transition_log_matrix(graph, a, fa, b, fb,
                                               math.hypot(tx[k] - tx[k - 1], ty[k] - ty[k - 1]),
                                               tt[k] - tt[k - 1], cfg)[0])
    decoded = [[x[0] for x in per_obs[k]].index(int(result.arc_ids[k])) for k in range(V)]
    best_score, _ = brute_force_best(log_emit, log_trans)
    assert path_score(decoded, log_emit, log_trans) == pytest.approx(best_score)


# ----------------------------------------------------------------------------- end-to-end behaviour
def test_hmm_overrules_classifier_pick_that_is_disconnected():
    g = island_graph()
    frame, probs = cands([(0, 0, 0.5, 0.9),
                          (1, 4, 0.5, 0.55), (1, 1, 0.5, 0.45),   # island arc 4 looks better alone
                          (2, 2, 0.5, 0.9)])
    tx, ty, tt = [50, 150, 250], [0, 0, 0], [0, 5, 10]
    base = top1_predictions(3, g.n_arcs, frame, probs)
    hmm = predict_route(tx, ty, tt, g, frame, probs)
    assert base.arc_ids.tolist() == [0, 4, 2]
    assert hmm.arc_ids.tolist() == [0, 1, 2]
    assert hmm.route == [0, 1, 2]
    assert hmm.diagnostics["path_transition_kinds"]["disconnected"] == 0


def test_missing_observation_is_preserved_and_bridged():
    g = line_graph()
    frame, probs = cands([(0, 0, 0.5, 0.9), (2, 2, 0.5, 0.9)])   # no candidates for point 1
    res = predict_route([50, 150, 250], [0, 0, 0], [0, 1, 2], g, frame, probs)
    assert res.arc_ids.tolist() == [0, NO_ARC, 2]
    assert res.segment_ids.tolist() == [0, -1, 0]               # bridged: same segment
    assert res.diagnostics["n_missing_points"] == 1
    assert res.route == [0, 2]


def test_trip_with_no_usable_candidates_returns_all_missing():
    g = line_graph()
    frame, probs = cands([])
    res = predict_route([0, 1], [0, 0], [0, 1], g, frame, probs)
    assert res.arc_ids.tolist() == [NO_ARC, NO_ARC] and res.route == []


def test_large_timestamp_gap_resets_instead_of_forcing_a_route():
    g = island_graph()
    frame, probs = cands([(0, 0, 0.5, 0.9), (1, 4, 0.5, 0.9)])
    res = predict_route([50, 50], [0, 1000], [0, 1000], g, frame, probs)
    assert res.arc_ids.tolist() == [0, 4]
    assert res.segment_ids.tolist() == [0, 1] and res.diagnostics["n_gap_resets"] == 1
    near = predict_route([50, 50], [0, 1000], [0, 5], g, frame, probs)  # same data, small gap
    assert near.diagnostics["n_gap_resets"] == 0 and near.segment_ids.tolist() == [0, 0]


def test_single_candidate_per_point():
    g = line_graph()
    frame, probs = cands([(0, 0, 0.4, 0.3), (1, 0, 0.6, 0.3)])
    res = predict_route([40, 60], [0, 0], [0, 1], g, frame, probs)
    assert res.arc_ids.tolist() == [0, 0] and res.route == [0]


def test_repeated_road_segments_preserve_nonconsecutive_revisits():
    xs, ys = [0, 100, 100, 0], [0, 0, 100, 100]
    g = RoadGraph(xs, ys, [(0, 1), (1, 2), (2, 3), (3, 0)])          # one-way square loop
    truth = [0, 1, 2, 3, 0]
    pts = [(50, 0), (100, 50), (50, 100), (0, 50), (50, 0)]
    rows = [(k, a, 0.5, 0.7 if a == truth[k] else 0.1) for k in range(5) for a in range(4)]
    frame, probs = cands(rows)
    res = predict_route([p[0] for p in pts], [p[1] for p in pts], np.arange(5) * 5.0, g, frame, probs)
    assert res.arc_ids.tolist() == truth
    assert res.route == [0, 1, 2, 3, 0]
    assert route_from_arcs([0, 0, 1, 1, 0, NO_ARC, 0]) == [0, 1, 0]


def test_invalid_candidate_rows_are_dropped_and_counted():
    g = line_graph()
    frame, probs = cands([(0, 0, 0.5, 0.9), (0, 99, 0.5, 0.99), (0, -5, 0.5, 0.99),
                          (0, 1, 0.5, 0.2), (7, 0, 0.5, 0.9),            # point 7 out of range
                          (1, 99, 0.5, 0.9)])                              # point 1: only invalid
    frame.loc[3, "arc_id"] = np.nan
    res = predict_route([50, 150], [0, 0], [0, 5], g, frame, probs)
    assert res.arc_ids.tolist() == [0, NO_ARC]
    assert res.diagnostics["n_invalid_rows"] == 5


def test_duplicate_candidate_rows_keep_most_probable():
    g = line_graph()
    frame, probs = cands([(0, 0, 0.5, 0.2), (0, 0, 0.5, 0.8), (0, 1, 0.5, 0.5)])
    res = predict_route([50], [0], [0], g, frame, probs)
    assert res.arc_ids.tolist() == [0] and res.diagnostics["n_duplicate_rows"] == 1


@pytest.mark.parametrize("arcs, message", [
    ([(0, 9)], "invalid node"), ([(-1, 1)], "invalid node"),
    ([(0, 1.5)], "integer"), (np.zeros((2, 3), dtype=int), "shape"),
])
def test_invalid_node_references_are_rejected(arcs, message):
    with pytest.raises(ValueError, match=message):
        RoadGraph([0, 1, 2], [0, 0, 0], arcs)


def test_graph_rejects_non_finite_or_mismatched_coordinates():
    with pytest.raises(ValueError):
        RoadGraph([0, np.nan], [0, 0], [(0, 1)])
    with pytest.raises(ValueError):
        RoadGraph([0, 1], [0], [(0, 1)])


def test_numerically_extreme_probabilities_stay_finite():
    g = line_graph()
    probs = np.array([0.0, 1e-320, 5e-324, np.nan, np.inf, -3.0, 2.0, 1.0])
    frame, _ = cands([(0, a, 0.5, 0.0) for a in range(4)] + [(1, a, 0.5, 0.0) for a in range(4)])
    obs = _prepare_observations(2, g.n_arcs, frame, probs, CFG)
    assert np.isfinite(obs.log_emit).all() and (obs.log_emit <= 0).all()
    assert (obs.prob >= 0).all() and (obs.prob <= 1).all()
    res = predict_route([50, 150], [0, 0], [0, 5], g, frame, probs)
    assert (res.arc_ids != NO_ARC).all()


def test_long_sequence_of_tiny_probabilities_does_not_underflow():
    g = line_graph(n=3)
    n = 4000
    frame, probs = cands([(k, a, 0.5, 1e-300 if a == 0 else 1e-299) for k in range(n) for a in (0, 1)])
    cfg = HMMConfig(prob_eps=1e-300, gap_reset_s=1e9)
    res = predict_route(np.full(n, 50.0), np.zeros(n), np.arange(n, dtype=float), g, frame, probs, cfg)
    assert res.arc_ids.shape == (n,) and (res.arc_ids != NO_ARC).all()
    # a naive product of 4000 probabilities of 1e-300 underflows to 0; the log-space decoder does not
    assert 1e-300 ** n == 0.0


def test_output_is_aligned_with_original_point_order_regardless_of_row_order():
    g = line_graph()
    rows = [(0, 0, 0.5, 0.9), (0, 3, 0.5, 0.1), (1, 1, 0.5, 0.8), (1, 0, 0.5, 0.2),
            (2, 2, 0.5, 0.7), (2, 1, 0.5, 0.3)]
    frame, probs = cands(rows)
    args = ([50, 150, 250], [0, 0, 0], [0, 5, 10], g)
    ref = predict_route(*args, frame, probs)
    perm = np.random.default_rng(3).permutation(len(rows))
    shuffled = predict_route(*args, frame.iloc[perm].reset_index(drop=True), probs[perm])
    assert ref.arc_ids.tolist() == [0, 1, 2] == shuffled.arc_ids.tolist()


def test_non_finite_gps_points_are_treated_as_missing():
    g = line_graph()
    frame, probs = cands([(0, 0, 0.5, 0.9), (1, 1, 0.5, 0.9), (2, 2, 0.5, 0.9)])
    res = predict_route([50, np.nan, 250], [0, 0, 0], [0, 5, 10], g, frame, probs)
    assert res.arc_ids.tolist() == [0, NO_ARC, 2]


def test_prediction_never_reads_ground_truth_route():
    class Tripwire(dict):
        def __getitem__(self, key):
            assert key != "route", "decoder touched the ground-truth route"
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key != "route", "decoder touched the ground-truth route"
            return super().get(key, default)

    g = line_graph()
    trip = Tripwire(trip_id="t", track_x=[50, 150], track_y=[0, 0], track_t=[0, 5],
                    nodes_x=g.nodes_x, nodes_y=g.nodes_y, arcs=np.c_[g.u, g.v], route=[3, 3, 3])
    frame, probs = cands([(0, 0, 0.5, 0.9), (1, 1, 0.5, 0.9)])
    res = predict_route_for_trip(trip, frame, probs)
    assert res.arc_ids.tolist() == [0, 1] and res.trip_id == "t"


def test_top1_baseline_breaks_ties_by_lowest_arc_id_and_marks_missing():
    frame, probs = cands([(0, 5, 0.5, 0.4), (0, 2, 0.5, 0.4), (0, 9, 0.5, np.nan), (2, 1, 0.5, 0.1)])
    res = top1_predictions(3, 10, frame, probs)
    assert res.arc_ids.tolist() == [2, NO_ARC, 1]


# ----------------------------------------------------------------------------- classifier interface
def test_positive_class_is_selected_by_label_not_position():
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert positive_class_probabilities(proba, [0, 1]).tolist() == [0.1, 0.8]
    assert positive_class_probabilities(proba, [1, 0]).tolist() == [0.9, 0.2]   # reversed order
    assert positive_class_probabilities(proba, [False, True]).tolist() == [0.1, 0.8]
    with pytest.raises(ValueError):
        positive_class_probabilities(proba, ["neg", "pos"])
    with pytest.raises(ValueError):
        positive_class_probabilities(proba, [0, 1, 2])


def test_score_candidates_with_sklearn_pipeline_and_feature_validation():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"distance": rng.uniform(0, 50, 200), "angle_diff": rng.uniform(0, 3, 200)})
    y = (X.distance + 10 * X.angle_diff < 25).astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X, y)
    bundle = {"pipeline": pipe, "features": ["distance", "angle_diff"], "model_name": "toy"}
    out = score_candidates(bundle, X)
    np.testing.assert_allclose(out, pipe.predict_proba(X)[:, list(pipe.classes_).index(1)])
    with pytest.raises(ValueError, match="lacks model feature"):
        score_candidates(bundle, X[["distance"]])


# ----------------------------------------------------------------------------- graph search
def test_bounded_dijkstra_matches_scipy_oracle():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    rng = np.random.default_rng(11)
    n = 200
    xs, ys = rng.uniform(0, 1000, n), rng.uniform(0, 1000, n)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, n, (900, 2)) if a != b}
    arcs = sorted(pairs)
    g = RoadGraph(xs, ys, arcs)
    mat = csr_matrix((g.length, (g.u, g.v)), shape=(n, n))
    for src, radius in [(int(s), float(r)) for s, r in zip(rng.integers(0, n, 25), rng.uniform(50, 600, 25))]:
        ours = g.distances_from(src, radius)
        ref = dijkstra(mat, directed=True, indices=src, limit=radius)
        for node in range(n):
            d = ours.get(node)
            if math.isfinite(ref[node]):
                assert d is not None and d == pytest.approx(ref[node])
            else:
                assert d is None or d > radius


def test_cache_gives_identical_answers_regardless_of_history():
    g1, g2 = line_graph(n=30), line_graph(n=30)
    g1.distances_from(0, 2500.0)                      # warm cache with a large radius
    small = g1.distances_from(0, 450.0)
    fresh = g2.distances_from(0, 450.0)
    within = lambda d: {k: v for k, v in d.items() if v <= 450.0}
    assert within(small) == within(fresh)
    assert g1.stats["cache_hits"] == 1 and g2.stats["cache_hits"] == 0
    g1.distances_from(0, 450.0, max_hops=3)           # different caps must not reuse the cache
    assert g1.stats["searches"] == 2


def test_paths_beyond_the_requested_radius_are_rejected_even_if_the_bucket_found_them():
    g = line_graph(n=8, spacing=70.0)
    # radius = 2 * 10 + 100 = 120 m; searches are rounded up to 150 m, so node 3 (140 m from node 1)
    # is present in the cached result but lies outside the requested radius.
    assert 140.0 in g.distances_from(1, 120.0).values()
    _, kind = tm(g, [0], [0.5], [3], [0.5], gps=10.0, dt=100.0)
    assert kind[0, 0] == KIND_DISCONNECTED
    _, kind = tm(g, [0], [0.5], [2], [0.5], gps=10.0, dt=100.0)   # 70 m away -> inside the radius
    assert kind[0, 0] == KIND_MULTIHOP


def test_hop_cap_and_node_budget_are_enforced_without_crashing():
    g = line_graph(n=10)
    assert set(g.distances_from(0, 1e6, max_hops=2)) == {0, 1, 2}
    g.distances_from(0, 1e6, max_hops=60, max_settled=3)
    assert g.stats["truncated_searches"] == 1
    lt, kind = tm(g, [0], [0.5], [8], [0.5], gps=800.0, dt=20.0, cfg=HMMConfig(max_hops=2))
    assert kind[0, 0] == KIND_DISCONNECTED and np.isfinite(lt).all()


def test_search_cost_is_local_on_a_large_network():
    side = 200                                         # 40,000 nodes, ~160,000 directed arcs
    ids = np.arange(side * side).reshape(side, side)
    xs, ys = np.meshgrid(np.arange(side) * 10.0, np.arange(side) * 10.0)
    horiz = np.c_[ids[:, :-1].ravel(), ids[:, 1:].ravel()]
    vert = np.c_[ids[:-1, :].ravel(), ids[1:, :].ravel()]
    arcs = np.vstack([horiz, horiz[:, ::-1], vert, vert[:, ::-1]])
    g = RoadGraph(xs.ravel(), ys.ravel(), arcs)
    t0 = time.perf_counter()
    reach = g.distances_from(int(ids[100, 100]), 50.0)
    assert time.perf_counter() - t0 < 3.0
    assert len(reach) <= 61 and g.stats["settled_nodes"] <= 61   # diamond of radius 5 hops


# ----------------------------------------------------------------------------- configuration
def test_config_validation_and_json_round_trip(tmp_path):
    with pytest.raises(ValueError):
        HMMConfig(beta_m=0)
    with pytest.raises(ValueError):
        HMMConfig(log_disconnect_penalty=1.0)
    with pytest.raises(ValueError):
        HMMConfig.from_dict({"not_a_parameter": 1})
    cfg = HMMConfig(beta_m=7.5, uturn_log_penalty=-2.0, normalize_transitions=True)
    cfg.save(tmp_path / "c.json", note="unit test")
    assert HMMConfig.load(tmp_path / "c.json") == cfg


# ----------------------------------------------------------------------------- real data (Members A/B)
PARQUET = ROOT / "results" / "features_student.parquet"
DATA = ROOT / "data" / "student_test"


@pytest.mark.skipif(not (PARQUET.exists() and DATA.exists()), reason="Member A's outputs not present")
def test_integration_with_member_a_candidates_on_a_real_trip():
    from src.preprocess import load_trip_data
    tid = "00000052"
    trip = load_trip_data(DATA, tid, load_route=False)         # ground truth never loaded here
    df = pd.read_parquet(PARQUET, filters=[("trip_id", "==", tid)]).reset_index(drop=True)
    graph = RoadGraph.from_trip(trip)
    # same CRS / arc indexing as Member A: lengths agree with A's arc_length feature, node ids too
    np.testing.assert_allclose(graph.length[df.arc_id.to_numpy()], df.arc_length.to_numpy(), atol=0.05)
    assert (graph.u[df.arc_id.to_numpy()] == df.from_node_id.to_numpy()).all()
    assert (graph.v[df.arc_id.to_numpy()] == df.to_node_id.to_numpy()).all()
    assert df.point_idx.nunique() == len(trip["track_x"])

    rng = np.random.default_rng(0)                              # noisy stand-in scores, test only
    probs = np.clip(0.55 * df.label.to_numpy() + 0.6 * rng.random(len(df)), 0, 1)
    res = predict_route_for_trip(trip, df, probs, graph=graph)
    top1 = top1_predictions(len(trip["track_x"]), graph.n_arcs, df, probs)
    ref = df.groupby("point_idx").reference_arc_id.first().to_numpy()
    assert res.arc_ids.shape == ref.shape and (res.arc_ids != NO_ARC).all()
    acc_hmm, acc_top1 = (res.arc_ids == ref).mean(), (top1.arc_ids == ref).mean()
    assert acc_hmm >= acc_top1 - 0.01                           # strong signal must not be corrupted
    assert res.diagnostics["graph_searches"] > 0
