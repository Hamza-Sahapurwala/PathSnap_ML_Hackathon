"""PathSnap HMM / Viterbi route decoder (Member C).

Pipeline position
-----------------
GPS track + road network -> candidates + features (Member A) -> positive-class
probabilities (Member B's classifier) -> **this module** -> predicted arc per GPS point.

Model
-----
* Hidden states   : the candidate arcs of each GPS observation (original ``arc_id`` kept).
* Emissions       : classifier probability of the positive class, in log space.
* Transitions     : directed road-network distance between the projected positions of two
  consecutive observations compared with the straight-line GPS displacement
  (Newson-Krumm style, Laplace error), plus a bounded floor for disconnected pairs.
* Decoding        : Viterbi in log space with variable candidate counts, missing
  observations and gap resets.

Design constraints (see ``results/hmm_integration_notes.md``)
-------------------------------------------------------------
* Prediction code never receives or reads the ground-truth ``.route``.
* No all-pairs shortest paths: bounded, cached Dijkstra over a CSR adjacency.
* The graph is directed; arcs are never silently treated as bidirectional.
* Only numpy / pandas are required at inference time (scipy is used by the tests).
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NO_ARC = -1  # value used in per-point outputs where no prediction can be made

# Transition kinds (reported in diagnostics and used by route-continuity analysis).
KIND_SAME = 0           # both observations on the same directed arc
KIND_DIRECT = 1         # arc a ends where arc b starts
KIND_MULTIHOP = 2       # connected through intermediate arcs (bounded search)
KIND_DISCONNECTED = 3   # no directed path found within the search bounds
KIND_IMPLAUSIBLE = 4    # connected, but would need an impossible speed
KIND_NAMES = {
    KIND_SAME: "same_arc",
    KIND_DIRECT: "direct",
    KIND_MULTIHOP: "multi_hop",
    KIND_DISCONNECTED: "disconnected",
    KIND_IMPLAUSIBLE: "implausible_speed",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HMMConfig:
    """Decoder hyper-parameters. Defaults are the *untuned baseline*."""

    # --- emission ---
    prob_eps: float = 1e-6          # probabilities are clipped to [prob_eps, 1]
    emission_power: float = 1.0     # log-emission = power * log(p); <1 flattens, >1 sharpens
    distance_weight: float = 0.0    # weight of the optional Gaussian distance term (0 = off)
    gps_sigma_m: float = 20.0       # sigma of that Gaussian term (metres)
    # --- transition ---
    beta_m: float = 10.0            # Laplace scale of |route distance - GPS displacement| (m)
    beta_gap_m_per_s: float = 2.0   # extra scale per second beyond 1 s between observations
    log_disconnect_penalty: float = -15.0  # bounded log-score of a disconnected transition
    uturn_log_penalty: float = 0.0  # extra log-score for a->reverse(a) transitions (0 = off)
    backward_log_penalty: float = 0.0  # extra log-score for moving backwards on the same arc
    normalize_transitions: bool = False  # row-normalise exp(log T) over the next candidate set
    max_speed_mps: float = 60.0     # transitions needing more than this are 'implausible'
    # --- bounded graph search ---
    route_factor: float = 2.0       # search radius = route_factor * GPS displacement + slack
    search_slack_m: float = 100.0
    max_search_m: float = 2000.0
    max_hops: int = 60
    max_settled: int = 4000         # per-search node budget (safety valve on dense networks)
    # --- gaps ---
    gap_reset_s: float = 60.0       # a larger time gap between observations starts a new segment

    def __post_init__(self) -> None:
        positive = ["prob_eps", "gps_sigma_m", "beta_m", "route_factor", "max_search_m",
                    "max_speed_mps", "gap_reset_s", "emission_power"]
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"HMMConfig.{name} must be > 0, got {getattr(self, name)!r}")
        if not 0 < self.prob_eps < 1:
            raise ValueError("HMMConfig.prob_eps must lie in (0, 1)")
        for name in ("beta_gap_m_per_s", "search_slack_m", "distance_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"HMMConfig.{name} must be >= 0")
        if self.max_hops < 1 or self.max_settled < 1:
            raise ValueError("HMMConfig.max_hops and max_settled must be >= 1")
        if self.log_disconnect_penalty > 0:
            raise ValueError("HMMConfig.log_disconnect_penalty must be <= 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HMMConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown HMMConfig keys: {sorted(unknown)}")
        return cls(**dict(data))

    def save(self, path: Path, **metadata: Any) -> None:
        payload = {"config": self.to_dict(), **({"meta": metadata} if metadata else {})}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HMMConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data.get("config", data))


# --------------------------------------------------------------------------- #
# Road graph with bounded, cached shortest-path search
# --------------------------------------------------------------------------- #
class RoadGraph:
    """Directed road network of one trip (nodes in metres, arcs ``u -> v``).

    ``arc_id`` is the zero-based row position in the ``.arcs`` file, exactly as in
    Member A's pipeline. Shortest paths are computed on demand by a bounded Dijkstra
    (distance radius, hop cap and node budget) and cached per source node, so no
    all-pairs matrix is ever built.
    """

    _BUCKET_M = 50.0  # search radii are rounded up to this to raise the cache hit rate

    def __init__(self, nodes_x: Sequence[float], nodes_y: Sequence[float],
                 arcs: Any, max_cache_sources: int = 50_000) -> None:
        nx = np.asarray(nodes_x, dtype=np.float64).ravel()
        ny = np.asarray(nodes_y, dtype=np.float64).ravel()
        if nx.shape != ny.shape:
            raise ValueError("nodes_x and nodes_y must have the same length")
        if not (np.isfinite(nx).all() and np.isfinite(ny).all()):
            raise ValueError("node coordinates must be finite")
        arr = np.asarray(arcs)
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"arcs must have shape (n_arcs, 2), got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.isfinite(arr).all() or (arr != np.round(arr)).any():
                raise ValueError("arcs must contain integer node ids")
        arr = arr.astype(np.int64)
        self.n_nodes = int(nx.size)
        bad = np.flatnonzero((arr < 0).any(axis=1) | (arr >= self.n_nodes).any(axis=1))
        if bad.size:
            k = int(bad[0])
            raise ValueError(f"arc {k} references invalid node id(s) {arr[k].tolist()} "
                             f"(valid range 0..{self.n_nodes - 1}); {bad.size} bad arc(s) in total")
        self.nodes_x, self.nodes_y = nx, ny
        self.u, self.v = arr[:, 0].copy(), arr[:, 1].copy()
        self.n_arcs = int(arr.shape[0])
        self.length = np.hypot(nx[self.v] - nx[self.u], ny[self.v] - ny[self.u])

        order = np.argsort(self.u, kind="stable")
        counts = np.bincount(self.u, minlength=self.n_nodes)
        self._ptr_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._head_np = self.v[order]
        self._w_np = self.length[order]
        self._arc_np = order.astype(np.int64)  # arc id of each CSR edge
        self._ptr: Optional[List[int]] = None  # python lists are built lazily (fast heap loop)
        self._head: Optional[List[int]] = None
        self._w: Optional[List[float]] = None
        self._arc: Optional[List[int]] = None

        self.max_cache_sources = int(max_cache_sources)
        self._cache: Dict[int, Tuple[float, int, int, Dict[int, float]]] = {}
        self.stats: Dict[str, float] = {}
        self.reset_stats()

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_trip(cls, trip: Mapping[str, Any], **kwargs: Any) -> "RoadGraph":
        """Build from a trip dict as returned by ``src.preprocess.load_trip_data``.

        Only ``nodes_x``, ``nodes_y`` and ``arcs`` are read; the reference route is never touched.
        """
        return cls(trip["nodes_x"], trip["nodes_y"], trip["arcs"], **kwargs)

    def reset_stats(self) -> None:
        self.stats = {"searches": 0, "cache_hits": 0, "settled_nodes": 0,
                      "truncated_searches": 0, "search_seconds": 0.0}

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- search --------------------------------------------------------------
    def _lists(self) -> Tuple[List[int], List[int], List[float]]:
        if self._ptr is None:
            self._ptr = self._ptr_np.tolist()
            self._head = self._head_np.tolist()
            self._w = self._w_np.tolist()
        return self._ptr, self._head, self._w  # type: ignore[return-value]

    def distances_from(self, source: int, radius: float, max_hops: int = 60,
                       max_settled: int = 4000) -> Dict[int, float]:
        """Directed shortest-path distances (m) from ``source``.

        The returned dict is exact for every node whose distance is <= ``radius`` (it may also
        contain farther nodes from a cached larger search; callers must compare to ``radius``).
        ``max_hops`` and ``max_settled`` are safety caps; reaching them can hide far nodes.
        """
        source = int(source)
        if not 0 <= source < self.n_nodes:
            raise IndexError(f"node {source} out of range 0..{self.n_nodes - 1}")
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= radius and hit[1] == max_hops and hit[2] == max_settled:
            self.stats["cache_hits"] += 1
            return hit[3]
        bucketed = math.ceil(max(radius, 1.0) / self._BUCKET_M) * self._BUCKET_M
        t0 = perf_counter()
        dist, settled, truncated = self._dijkstra(source, bucketed, max_hops, max_settled)
        self.stats["search_seconds"] += perf_counter() - t0
        self.stats["searches"] += 1
        self.stats["settled_nodes"] += settled
        self.stats["truncated_searches"] += int(truncated)
        self._cache.pop(source, None)
        self._cache[source] = (bucketed, max_hops, max_settled, dist)
        while len(self._cache) > self.max_cache_sources:
            del self._cache[next(iter(self._cache))]  # evict oldest insertion
        return dist

    def _dijkstra(self, source: int, radius: float, max_hops: int,
                  max_settled: int) -> Tuple[Dict[int, float], int, bool]:
        ptr, head, weight = self._lists()
        inf = math.inf
        dist: Dict[int, float] = {source: 0.0}
        hops: Dict[int, int] = {source: 0}
        heap: List[Tuple[float, int]] = [(0.0, source)]
        settled = 0
        truncated = False
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, n = pop(heap)
            if d > dist[n]:
                continue  # stale heap entry
            settled += 1
            if settled > max_settled:
                truncated = True
                break
            h = hops[n]
            if h >= max_hops:
                continue
            for k in range(ptr[n], ptr[n + 1]):
                nd = d + weight[k]
                if nd > radius:
                    continue
                m = head[k]
                if nd < dist.get(m, inf):
                    dist[m] = nd
                    hops[m] = h + 1
                    push(heap, (nd, m))
        return dist, settled, truncated

    def shortest_path_arcs(self, source: int, target: int, radius: float,
                           max_settled: int = 20_000) -> Optional[List[int]]:
        """Arc ids of a shortest directed path ``source -> target`` (``[]`` if they are equal).

        Bounded like ``distances_from``: returns ``None`` if the target is farther than
        ``radius`` or the node budget runs out. Not cached; it is only used to fill routes.
        """
        source, target = int(source), int(target)
        if source == target:
            return []
        ptr, head, weight = self._lists()
        if self._arc is None:
            self._arc = self._arc_np.tolist()
        arc_of = self._arc
        inf = math.inf
        dist: Dict[int, float] = {source: 0.0}
        via: Dict[int, int] = {}  # node -> CSR edge used to reach it
        heap: List[Tuple[float, int]] = [(0.0, source)]
        settled = 0
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, n = pop(heap)
            if d > dist[n]:
                continue
            if n == target:
                break
            settled += 1
            if settled > max_settled:
                return None
            for k in range(ptr[n], ptr[n + 1]):
                nd = d + weight[k]
                if nd > radius:
                    continue
                m = head[k]
                if nd < dist.get(m, inf):
                    dist[m] = nd
                    via[m] = k
                    push(heap, (nd, m))
        else:
            return None
        path: List[int] = []
        n = target
        while n != source:
            arc = arc_of[via[n]]
            path.append(arc)
            n = int(self.u[arc])
        path.reverse()
        return path


# --------------------------------------------------------------------------- #
# Transition model
# --------------------------------------------------------------------------- #
def transition_log_matrix(graph: RoadGraph, a_arc: np.ndarray, a_frac: np.ndarray,
                          b_arc: np.ndarray, b_frac: np.ndarray, gps_dist: float, dt: float,
                          cfg: HMMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Log transition scores ``log T[i, j]`` from candidate ``i`` (obs t-1) to ``j`` (obs t).

    For a pair of projected positions the vehicle travels forward along arc ``a`` to its end,
    along the shortest directed path to the start of arc ``b``, then along ``b`` up to its
    projection. That network distance is compared with the straight-line GPS displacement::

        log T = -| d_net - d_gps | / beta  -  log(2 beta)          (Laplace error)

    ``beta`` grows with the time between observations. Pairs without a directed path inside the
    search bounds (or needing an impossible speed) receive the bounded floor
    ``cfg.log_disconnect_penalty`` instead of -inf, so decoding never collapses.

    Returns ``(log_T, kind)`` where ``kind`` uses the ``KIND_*`` codes.
    """
    a_arc = np.asarray(a_arc, dtype=np.int64)
    b_arc = np.asarray(b_arc, dtype=np.int64)
    a_frac = np.asarray(a_frac, dtype=np.float64)
    b_frac = np.asarray(b_frac, dtype=np.float64)
    m, n = a_arc.size, b_arc.size
    La, Lb = graph.length[a_arc], graph.length[b_arc]
    ua, va = graph.u[a_arc], graph.v[a_arc]
    ub, vb = graph.u[b_arc], graph.v[b_arc]
    rem_a = (1.0 - a_frac) * La   # distance from the projection on a to the end of a
    off_b = b_frac * Lb           # distance from the start of b to the projection on b

    d_net = np.full((m, n), np.inf)
    kind = np.full((m, n), KIND_DISCONNECTED, dtype=np.int8)

    same = a_arc[:, None] == b_arc[None, :]
    if same.any():
        d_same = np.abs(b_frac[None, :] - a_frac[:, None]) * La[:, None]
        d_net[same] = d_same[same]
        kind[same] = KIND_SAME

    direct = (va[:, None] == ub[None, :]) & ~same
    if direct.any():
        d_dir = rem_a[:, None] + off_b[None, :]
        d_net[direct] = d_dir[direct]
        kind[direct] = KIND_DIRECT

    todo = ~(same | direct)
    if todo.any():
        radius = min(cfg.max_search_m, cfg.route_factor * gps_dist + cfg.search_slack_m)
        rows_needed = np.flatnonzero(todo.any(axis=1))
        for src in np.unique(va[rows_needed]):
            rows = rows_needed[va[rows_needed] == src]
            reach = graph.distances_from(int(src), radius, cfg.max_hops, cfg.max_settled)
            for j in np.flatnonzero(todo[rows].any(axis=0)):
                sp = reach.get(int(ub[j]))
                if sp is None or sp > radius:
                    continue
                for i in rows:
                    if todo[i, j]:
                        d_net[i, j] = rem_a[i] + sp + off_b[j]
                        kind[i, j] = KIND_MULTIHOP

    implausible = np.isfinite(d_net) & (d_net > cfg.max_speed_mps * max(dt, 1.0))
    if implausible.any():
        d_net[implausible] = np.inf
        kind[implausible] = KIND_IMPLAUSIBLE
    valid = np.isfinite(d_net)

    beta = cfg.beta_m + cfg.beta_gap_m_per_s * max(dt - 1.0, 0.0)
    log_t = np.full((m, n), cfg.log_disconnect_penalty, dtype=np.float64)
    log_t[valid] = -np.abs(d_net[valid] - gps_dist) / beta - math.log(2.0 * beta)

    if cfg.uturn_log_penalty != 0.0:
        uturn = (ub[None, :] == va[:, None]) & (vb[None, :] == ua[:, None]) & ~same & valid
        log_t[uturn] += cfg.uturn_log_penalty
    if cfg.backward_log_penalty != 0.0:
        backward = same & (b_frac[None, :] < a_frac[:, None]) & valid
        log_t[backward] += cfg.backward_log_penalty
    if cfg.normalize_transitions:
        top = log_t.max(axis=1, keepdims=True)
        log_t = log_t - (top + np.log(np.exp(log_t - top).sum(axis=1, keepdims=True)))
    return log_t, kind


# --------------------------------------------------------------------------- #
# Viterbi (pure, graph-independent)
# --------------------------------------------------------------------------- #
def viterbi(log_emit: Sequence[np.ndarray],
            trans_fn: Callable[[int], Optional[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Log-space Viterbi over observations with a variable number of states.

    ``log_emit[k]`` holds the emission scores of the states of observation ``k``.
    ``trans_fn(k)`` returns the ``(n_{k-1}, n_k)`` log-transition matrix into observation ``k``,
    or ``None`` to *reset* (start an independent segment at ``k``). A step whose every score is
    -inf also starts a new segment instead of failing.

    Returns ``(states, segment)``: per observation, the chosen state index and the segment id.
    Ties resolve to the lowest state index (callers order states by arc id).
    """
    total = len(log_emit)
    states = np.zeros(total, dtype=np.int64)
    segment = np.zeros(total, dtype=np.int64)
    if total == 0:
        return states, segment
    back: List[Optional[np.ndarray]] = [None] * total
    seg_starts = [0]
    seg_ends: List[int] = []
    delta = np.asarray(log_emit[0], dtype=np.float64)
    if delta.size == 0 or np.isnan(delta).any():
        raise ValueError("emission scores must be non-empty and NaN-free")
    for k in range(1, total):
        emit = np.asarray(log_emit[k], dtype=np.float64)
        if emit.size == 0 or np.isnan(emit).any():
            raise ValueError("emission scores must be non-empty and NaN-free")
        trans = trans_fn(k)
        reset = trans is None
        if not reset:
            trans = np.asarray(trans, dtype=np.float64)
            if trans.shape != (delta.size, emit.size):
                raise ValueError(f"transition {k} has shape {trans.shape}, "
                                 f"expected {(delta.size, emit.size)}")
            if np.isnan(trans).any() or (trans == np.inf).any():
                raise ValueError("transition scores must not contain NaN or +inf")
            scores = delta[:, None] + trans
            best_prev = np.argmax(scores, axis=0)
            new_delta = scores[best_prev, np.arange(emit.size)] + emit
            reset = not np.isfinite(new_delta).any()
        if reset:
            seg_ends.append(int(np.argmax(delta)))
            seg_starts.append(k)
            delta = emit.copy()
        else:
            back[k] = best_prev
            delta = new_delta
    seg_ends.append(int(np.argmax(delta)))
    bounds = seg_starts + [total]
    for s in range(len(seg_starts)):
        lo, hi = bounds[s], bounds[s + 1]
        state = seg_ends[s]
        states[hi - 1] = state
        for k in range(hi - 1, lo, -1):
            state = int(back[k][state])  # type: ignore[index]
            states[k - 1] = state
        segment[lo:hi] = s
    return states, segment


# --------------------------------------------------------------------------- #
# Candidate preparation and classifier interface
# --------------------------------------------------------------------------- #
@dataclass
class _Observations:
    """Valid observations (those with >= 1 usable candidate), candidates stored flat."""
    obs_point: np.ndarray   # (V,) original GPS point index of each valid observation
    offsets: np.ndarray     # (V+1,) slice boundaries into the flat arrays below
    arc: np.ndarray         # candidate arc ids (ascending within an observation)
    frac: np.ndarray        # projection fraction in [0, 1] along the arc
    prob: np.ndarray        # sanitised positive-class probability in [0, 1]
    log_emit: np.ndarray    # emission log-score
    n_input_rows: int
    n_invalid_rows: int
    n_duplicate_rows: int


def positive_class_probabilities(proba: np.ndarray, classes: Sequence[Any],
                                 positive: Any = 1) -> np.ndarray:
    """Select the positive-class column of ``predict_proba`` output by *label*, not position."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = list(classes)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(f"predict_proba shape {proba.shape} does not match {len(classes)} classes")
    matches = [i for i, c in enumerate(classes) if c == positive]
    if len(matches) != 1:
        raise ValueError(f"positive class {positive!r} not found in model classes {classes}")
    return proba[:, matches[0]]


def load_classifier(path: Path) -> Dict[str, Any]:
    """Load Member B's ``artifacts/final_model.joblib`` bundle (pipeline, features, model_name)."""
    import joblib  # local import: inference-only dependency
    bundle = joblib.load(path)
    for key in ("pipeline", "features"):
        if key not in bundle:
            raise ValueError(f"classifier bundle {path} lacks key {key!r}")
    return bundle


def score_candidates(bundle: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for each candidate row (aligned with ``frame`` row order)."""
    features = list(bundle["features"])
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(f"candidate frame lacks model feature columns: {missing}")
    pipe = bundle["pipeline"]
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        classes = pipe[-1].classes_
    return positive_class_probabilities(pipe.predict_proba(frame[features]), classes)


def _prepare_observations(n_points: int, n_arcs: int, candidates: pd.DataFrame,
                          probabilities: Any, cfg: HMMConfig, need_geometry: bool = True,
                          point_ok: Optional[np.ndarray] = None) -> _Observations:
    required = ["point_idx", "arc_id"] + (["proj_frac"] if need_geometry else [])
    if cfg.distance_weight > 0:
        required.append("distance")
    absent = [c for c in required if c not in candidates.columns]
    if absent:
        raise ValueError(f"candidates lack required column(s): {absent}")
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    if p.size != len(candidates):
        raise ValueError(f"{p.size} probabilities for {len(candidates)} candidate rows")
    pt = pd.to_numeric(candidates["point_idx"], errors="coerce").to_numpy(dtype=np.float64)
    arc = pd.to_numeric(candidates["arc_id"], errors="coerce").to_numpy(dtype=np.float64)
    frac = (pd.to_numeric(candidates["proj_frac"], errors="coerce").to_numpy(dtype=np.float64)
            if need_geometry else np.full(len(p), 0.5))
    dist = (pd.to_numeric(candidates["distance"], errors="coerce").to_numpy(dtype=np.float64)
            if cfg.distance_weight > 0 else np.zeros(len(p)))

    with np.errstate(invalid="ignore"):
        ok = (np.isfinite(pt) & (pt == np.round(pt)) & (pt >= 0) & (pt < n_points)
              & np.isfinite(arc) & (arc == np.round(arc)) & (arc >= 0) & (arc < n_arcs))
    if point_ok is not None:  # observations with non-finite GPS values are treated as missing
        with np.errstate(invalid="ignore"):
            ok &= point_ok[np.clip(np.nan_to_num(pt, nan=0.0), 0, n_points - 1).astype(np.int64)]
    n_invalid = int((~ok).sum())
    pt, arc = pt[ok].astype(np.int64), arc[ok].astype(np.int64)
    frac = np.clip(np.nan_to_num(frac[ok], nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    dist = np.nan_to_num(dist[ok], nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(np.nan_to_num(p[ok], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # one row per (point, arc): keep the most probable duplicate; order rows by point then arc id
    order = np.lexsort((-prob, arc, pt))
    pt, arc, frac, dist, prob = pt[order], arc[order], frac[order], dist[order], prob[order]
    first = np.ones(pt.size, dtype=bool)
    first[1:] = (pt[1:] != pt[:-1]) | (arc[1:] != arc[:-1])
    n_dup = int((~first).sum())
    pt, arc, frac, dist, prob = pt[first], arc[first], frac[first], dist[first], prob[first]

    log_emit = cfg.emission_power * np.log(np.maximum(prob, cfg.prob_eps))
    if cfg.distance_weight > 0:
        log_emit = log_emit - cfg.distance_weight * 0.5 * (dist / cfg.gps_sigma_m) ** 2
    obs_point, starts = np.unique(pt, return_index=True)
    offsets = np.concatenate([starts, [pt.size]]).astype(np.int64)
    return _Observations(obs_point.astype(np.int64), offsets, arc, frac, prob, log_emit,
                         int(len(p)), n_invalid, n_dup)


# --------------------------------------------------------------------------- #
# Public prediction API
# --------------------------------------------------------------------------- #
@dataclass
class RouteResult:
    """Decoder output, aligned with the original GPS observations."""
    n_points: int
    arc_ids: np.ndarray               # (n_points,) predicted arc id, NO_ARC (-1) where missing
    segment_ids: np.ndarray           # (n_points,) decoding segment, -1 where missing
    route: List[int]                  # arc sequence with consecutive repeats removed
    trip_id: Optional[str] = None
    diagnostics: Dict[str, Any] = None  # type: ignore[assignment]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"point_idx": np.arange(self.n_points, dtype=np.int64),
                             "arc_id": self.arc_ids, "segment_id": self.segment_ids})


def route_from_arcs(arc_ids: Sequence[int]) -> List[int]:
    """Collapse *consecutive* repeats and skip missing points; revisits are preserved."""
    route: List[int] = []
    for a in np.asarray(arc_ids, dtype=np.int64).tolist():
        if a == NO_ARC:
            continue
        if not route or route[-1] != a:
            route.append(a)
    return route


def fill_route(route: Sequence[int], graph: RoadGraph, max_gap_m: float = 3000.0,
               max_settled: int = 20_000) -> Tuple[List[int], int]:
    """Insert the shortest directed connecting arcs between consecutive route arcs.

    GPS points are sparser than arcs, so a per-point route skips the arcs driven between
    observations. Arcs that already meet (``v[a] == u[b]``) are kept as they are; when no
    path is found within ``max_gap_m`` the two arcs stay adjacent and count as a break.
    Returns ``(filled_route, n_breaks)``.
    """
    out: List[int] = []
    breaks = 0
    for b in np.asarray(route, dtype=np.int64).tolist():
        if out and out[-1] != b:
            a = out[-1]
            if graph.v[a] != graph.u[b]:
                path = graph.shortest_path_arcs(graph.v[a], graph.u[b], max_gap_m, max_settled)
                if path is None:
                    breaks += 1
                else:
                    out.extend(path)
        if not out or out[-1] != b:
            out.append(b)
    return out, breaks


def top1_predictions(n_points: int, n_arcs: int, candidates: pd.DataFrame, probabilities: Any,
                     config: Optional[HMMConfig] = None) -> RouteResult:
    """Classifier-only baseline: highest-probability candidate per point (ties -> lowest arc id)."""
    cfg = config or HMMConfig()
    t0 = perf_counter()
    obs = _prepare_observations(n_points, n_arcs, candidates, probabilities, cfg, need_geometry=False)
    arc_ids = np.full(n_points, NO_ARC, dtype=np.int64)
    for k, point in enumerate(obs.obs_point):
        lo, hi = obs.offsets[k], obs.offsets[k + 1]
        arc_ids[point] = obs.arc[lo + int(np.argmax(obs.prob[lo:hi]))]
    segment = np.where(arc_ids == NO_ARC, -1, 0).astype(np.int64)
    diag = {"method": "classifier_top1", "n_points": n_points,
            "n_missing_points": int((arc_ids == NO_ARC).sum()),
            "n_invalid_rows": obs.n_invalid_rows, "n_duplicate_rows": obs.n_duplicate_rows,
            "seconds_total": perf_counter() - t0}
    return RouteResult(n_points, arc_ids, segment, route_from_arcs(arc_ids), None, diag)


def predict_route(track_x: Sequence[float], track_y: Sequence[float], track_t: Sequence[float],
                  graph: RoadGraph, candidates: pd.DataFrame, probabilities: Any,
                  config: Optional[HMMConfig] = None,
                  trip_id: Optional[str] = None) -> RouteResult:
    """Decode the most likely arc for every GPS observation of one trip.

    Parameters
    ----------
    track_x, track_y, track_t
        Projected GPS positions (metres, same CRS as ``graph``) and timestamps (seconds).
    graph
        ``RoadGraph`` of this trip. Re-use the object across calls to benefit from its cache.
    candidates
        One row per (GPS point, candidate arc) with at least ``point_idx``, ``arc_id`` and
        ``proj_frac`` (plus ``distance`` if ``config.distance_weight > 0``).
    probabilities
        Positive-class probability for each candidate row, aligned with ``candidates``.
    config
        ``HMMConfig`` (defaults to the untuned baseline).

    No ground-truth information is accepted or required.
    """
    cfg = config or HMMConfig()
    t_start = perf_counter()
    tx = np.asarray(track_x, dtype=np.float64).ravel()
    ty = np.asarray(track_y, dtype=np.float64).ravel()
    tt = np.asarray(track_t, dtype=np.float64).ravel()
    if not (tx.size == ty.size == tt.size):
        raise ValueError("track_x, track_y and track_t must have the same length")
    n_points = int(tx.size)
    search_before = dict(graph.stats)

    t0 = perf_counter()
    point_ok = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(tt)
    obs = _prepare_observations(n_points, graph.n_arcs, candidates, probabilities, cfg,
                                point_ok=point_ok)
    t_prep = perf_counter() - t0
    n_valid = int(obs.obs_point.size)
    log_emit = [obs.log_emit[obs.offsets[k]:obs.offsets[k + 1]] for k in range(n_valid)]

    kinds: List[Optional[np.ndarray]] = [None] * n_valid
    counters = {"gap_resets": 0}
    t_trans = [0.0]

    def trans_fn(k: int) -> Optional[np.ndarray]:
        i, j = int(obs.obs_point[k - 1]), int(obs.obs_point[k])
        dt = tt[j] - tt[i]
        if dt > cfg.gap_reset_s:
            counters["gap_resets"] += 1
            return None
        t1 = perf_counter()
        a, b = slice(obs.offsets[k - 1], obs.offsets[k]), slice(obs.offsets[k], obs.offsets[k + 1])
        log_t, kind = transition_log_matrix(
            graph, obs.arc[a], obs.frac[a], obs.arc[b], obs.frac[b],
            float(math.hypot(tx[j] - tx[i], ty[j] - ty[i])), float(dt), cfg)
        kinds[k] = kind
        t_trans[0] += perf_counter() - t1
        return log_t

    t0 = perf_counter()
    states, segment = viterbi(log_emit, trans_fn)
    t_viterbi = perf_counter() - t0

    arc_ids = np.full(n_points, NO_ARC, dtype=np.int64)
    segment_ids = np.full(n_points, -1, dtype=np.int64)
    for k in range(n_valid):
        point = obs.obs_point[k]
        arc_ids[point] = obs.arc[obs.offsets[k] + states[k]]
        segment_ids[point] = segment[k]

    kind_counts = {name: 0 for name in KIND_NAMES.values()}
    for k in range(1, n_valid):
        if segment[k] == segment[k - 1] and kinds[k] is not None:
            kind_counts[KIND_NAMES[int(kinds[k][states[k - 1], states[k]])]] += 1

    stats = graph.stats
    diagnostics = {
        "method": "hmm_viterbi", "n_points": n_points, "n_valid_observations": n_valid,
        "n_missing_points": n_points - n_valid, "n_segments": int(segment.max()) + 1 if n_valid else 0,
        "n_gap_resets": counters["gap_resets"], "n_invalid_rows": obs.n_invalid_rows,
        "n_duplicate_rows": obs.n_duplicate_rows, "path_transition_kinds": kind_counts,
        "graph_searches": int(stats["searches"] - search_before["searches"]),
        "graph_cache_hits": int(stats["cache_hits"] - search_before["cache_hits"]),
        "graph_settled_nodes": int(stats["settled_nodes"] - search_before["settled_nodes"]),
        "graph_truncated_searches": int(stats["truncated_searches"] - search_before["truncated_searches"]),
        "seconds_graph_search": stats["search_seconds"] - search_before["search_seconds"],
        "seconds_prepare": t_prep, "seconds_transitions": t_trans[0],
        "seconds_viterbi_total": t_viterbi, "seconds_total": perf_counter() - t_start,
    }
    return RouteResult(n_points, arc_ids, segment_ids, route_from_arcs(arc_ids), trip_id, diagnostics)


def predict_route_for_trip(trip: Mapping[str, Any], candidates: pd.DataFrame, probabilities: Any,
                           config: Optional[HMMConfig] = None,
                           graph: Optional[RoadGraph] = None) -> RouteResult:
    """Convenience wrapper for a trip dict from ``src.preprocess.load_trip_data``.

    Reads only ``track_x``, ``track_y``, ``track_t``, ``nodes_x``, ``nodes_y``, ``arcs`` and
    ``trip_id``. The ``route`` entry (ground truth) is deliberately never accessed.
    """
    graph = graph or RoadGraph.from_trip(trip)
    return predict_route(trip["track_x"], trip["track_y"], trip["track_t"], graph,
                         candidates, probabilities, config, trip.get("trip_id"))
