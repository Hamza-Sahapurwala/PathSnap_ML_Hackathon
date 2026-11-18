"""
PathSnap Preprocessing & Candidate Generation Module
Member A: Setup, EDA, candidates, labels, features
"""

import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from pyproj import Transformer

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for p in [RESULTS_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Coordinate Projection
# -----------------------------------------------------------------------------
# Choice of Metric CRS Justification:
# Longitude and latitude in the raw dataset are angular coordinates on the WGS84
# ellipsoid (EPSG:4326). Euclidean distance in degrees induces severe spatial
# distortion that varies with latitude due to meridian convergence (dx = R*cos(lat)*dlon).
# We dynamically project coordinates to the appropriate local Universal Transverse
# Mercator (UTM) conformal projection (e.g. EPSG:32632 for UTM Zone 32N covering
# Western/Central Germany) using pyproj. UTM preserves local angles, maintains
# conformal mapping, and ensures physical distance calculations in metres and kinematic
# speeds in metres per second are metric, accurate, and free from coordinate distortion.
# -----------------------------------------------------------------------------

def get_utm_epsg(lon: float, lat: float) -> int:
    """Calculate appropriate UTM EPSG code for a given lon/lat."""
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def project_coordinates(
    lons: np.ndarray, 
    lats: np.ndarray, 
    epsg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Project longitude and latitude coordinates to local metric UTM CRS.
    Returns: (x_metres, y_metres, epsg)
    """
    if epsg is None:
        mean_lon = float(np.mean(lons))
        mean_lat = float(np.mean(lats))
        epsg = get_utm_epsg(mean_lon, mean_lat)

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), epsg

def load_trip_data(trip_dir: Path, trip_id: str, load_route: bool = True) -> Dict[str, Any]:
    """Load and project trip data files (.track, .nodes, .arcs, .route)."""
    trip_path = Path(trip_dir)
    base_file = trip_path / trip_id / trip_id

    track = np.loadtxt(f"{base_file}.track", dtype=np.float64)
    nodes = np.loadtxt(f"{base_file}.nodes", dtype=np.float64)
    arcs = np.loadtxt(f"{base_file}.arcs", dtype=np.int64)

    route = None
    route_file = Path(f"{base_file}.route")
    if load_route and route_file.exists():
        route = np.loadtxt(route_file, dtype=np.int64)
        if route.ndim == 0:
            route = np.array([route], dtype=np.int64)

    mean_lon = float(np.mean(track[:, 0]))
    mean_lat = float(np.mean(track[:, 1]))
    epsg = get_utm_epsg(mean_lon, mean_lat)

    tx, ty, _ = project_coordinates(track[:, 0], track[:, 1], epsg=epsg)
    nx, ny, _ = project_coordinates(nodes[:, 0], nodes[:, 1], epsg=epsg)

    return {
        "trip_id": trip_id,
        "epsg": epsg,
        "track_raw": track,       # lon, lat, t
        "track_x": tx,            # UTM x (m)
        "track_y": ty,            # UTM y (m)
        "track_t": track[:, 2],   # timestamp (s)
        "nodes_raw": nodes,       # lon, lat
        "nodes_x": nx,            # UTM x (m)
        "nodes_y": ny,            # UTM y (m)
        "arcs": arcs,             # from_node, to_node
        "route": route            # ground truth arc_ids
    }

# -----------------------------------------------------------------------------
# 2. Candidate Arc Generation (Top-K via KD-Tree)
# -----------------------------------------------------------------------------
class CandidateArcGenerator:
    """
    Spatial candidate generator using dual KD-trees over arc midpoints and incident nodes.
    For each GPS observation, retrieves nearest candidate segments, computes exact
    point-to-segment perpendicular distance and projection fraction t in [0, 1].
    """
    def __init__(
        self,
        nodes_x: np.ndarray,
        nodes_y: np.ndarray,
        arcs: np.ndarray,
        top_k: int = 10,
        radius_cap: float = 120.0,
        node_query_k: int = 25,
        midpoint_query_k: int = 25
    ):
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.arcs = arcs
        self.top_k = top_k
        self.radius_cap = radius_cap
        self.node_query_k = min(node_query_k, len(nodes_x))
        self.midpoint_query_k = min(midpoint_query_k, len(arcs))

        # Node degree dictionary (in + out)
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        self.node_to_arcs = defaultdict(list)
        for arc_id, (u, v) in enumerate(arcs):
            self.node_to_arcs[u].append(arc_id)
            self.node_to_arcs[v].append(arc_id)
            self.out_degree[u] += 1
            self.in_degree[v] += 1

        # KDTree on nodes
        self.node_tree = KDTree(np.column_stack([nodes_x, nodes_y]))

        # Arc endpoints and midpoints
        u = arcs[:, 0]
        v = arcs[:, 1]
        self.ax = nodes_x[u]
        self.ay = nodes_y[u]
        self.bx = nodes_x[v]
        self.by = nodes_y[v]

        mx = (self.ax + self.bx) / 2.0
        my = (self.ay + self.by) / 2.0
        self.mid_tree = KDTree(np.column_stack([mx, my]))

    def generate_candidates(
        self, 
        track_x: np.ndarray, 
        track_y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate top_k candidate arcs for each GPS point.
        Returns:
            - cand_arcs: (N, K) arc IDs
            - distances: (N, K) perpendicular distance in metres
            - proj_fracs: (N, K) projection fraction t in [0, 1]
            - proj_xs: (N, K) projected x in metres
            - proj_ys: (N, K) projected y in metres
        """
        N = len(track_x)
        pts = np.column_stack([track_x, track_y])

        # Query KD-trees
        _, node_indices = self.node_tree.query(pts, k=self.node_query_k)
        if node_indices.ndim == 1:
            node_indices = node_indices[:, None]

        _, mid_indices = self.mid_tree.query(pts, k=self.midpoint_query_k)
        if mid_indices.ndim == 1:
            mid_indices = mid_indices[:, None]

        out_cand_arcs = np.zeros((N, self.top_k), dtype=np.int64)
        out_distances = np.zeros((N, self.top_k), dtype=np.float32)
        out_proj_fracs = np.zeros((N, self.top_k), dtype=np.float32)
        out_proj_xs = np.zeros((N, self.top_k), dtype=np.float32)
        out_proj_ys = np.zeros((N, self.top_k), dtype=np.float32)

        for i in range(N):
            px, py = track_x[i], track_y[i]
            # Gather unique candidate arcs from nearest nodes and nearest midpoints
            c_set = set(mid_indices[i])
            for nid in node_indices[i]:
                c_set.update(self.node_to_arcs[nid])

            cand_ids = np.fromiter(c_set, dtype=np.int64, count=len(c_set))
            c_ax = self.ax[cand_ids]
            c_ay = self.ay[cand_ids]
            c_bx = self.bx[cand_ids]
            c_by = self.by[cand_ids]

            dx = c_bx - c_ax
            dy = c_by - c_ay
            l2 = np.where(dx*dx + dy*dy == 0, 1e-9, dx*dx + dy*dy)

            t = np.clip(((px - c_ax)*dx + (py - c_ay)*dy) / l2, 0.0, 1.0)
            proj_x = c_ax + t * dx
            proj_y = c_ay + t * dy
            dists = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

            # Sort by exact perpendicular distance
            sort_order = np.argsort(dists)
            top_indices = sort_order[:self.top_k]
            k_found = len(top_indices)

            chosen_arcs = cand_ids[top_indices]
            chosen_dists = dists[top_indices]
            chosen_t = t[top_indices]
            chosen_px = proj_x[top_indices]
            chosen_py = proj_y[top_indices]

            out_cand_arcs[i, :k_found] = chosen_arcs
            out_distances[i, :k_found] = chosen_dists
            out_proj_fracs[i, :k_found] = chosen_t
            out_proj_xs[i, :k_found] = chosen_px
            out_proj_ys[i, :k_found] = chosen_py

            # Fill up to top_k if fewer found
            if k_found < self.top_k:
                for fill_idx in range(k_found, self.top_k):
                    out_cand_arcs[i, fill_idx] = chosen_arcs[-1]
                    out_distances[i, fill_idx] = chosen_dists[-1] + self.radius_cap
                    out_proj_fracs[i, fill_idx] = chosen_t[-1]
                    out_proj_xs[i, fill_idx] = chosen_px[-1]
                    out_proj_ys[i, fill_idx] = chosen_py[-1]

        return {
            "cand_arcs": out_cand_arcs,
            "distances": out_distances,
            "proj_fracs": out_proj_fracs,
            "proj_xs": out_proj_xs,
            "proj_ys": out_proj_ys
        }

# -----------------------------------------------------------------------------
# 3. Ground Truth Labels via Monotonic Dynamic Programming Alignment
# -----------------------------------------------------------------------------
# Assumption Documentation:
# Ground truth vehicle movement traverses the reference route sequence monotonically
# in time. Although raw GPS observations exhibit sensor noise, multipath jumps, and
# latency, the vehicle cannot physically travel backwards along its traversed route
# sequence in a forward journey. We formulate the ground-truth assignment as a dynamic
# program that finds the optimal monotonic path s_0 <= s_1 <= ... <= s_{N-1} through
# the reference route arcs minimizing cumulative perpendicular squared projection
# error. This yields robust, leakage-free per-point ground-truth arc labels that
# naturally resolve junction ambiguities, self-intersections, and parallel roadways.
# -----------------------------------------------------------------------------

def align_gps_to_route_dp(
    track_x: np.ndarray,
    track_y: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    max_advance: int = 5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Monotonic DP alignment:
    Maps each GPS track point to its reference arc along the route sequence.
    Returns:
        - aligned_arcs: (N,) arc IDs
        - alignment_metrics: {mean_dist, median_dist, max_dist}
    """
    N = len(track_x)
    M = len(route)
    if M == 1:
        aligned = np.full(N, route[0], dtype=np.int64)
        return aligned, {"mean_dist": 0.0, "median_dist": 0.0, "max_dist": 0.0}

    route_u, route_v = arcs[route, 0], arcs[route, 1]
    ax, ay = nodes_x[route_u], nodes_y[route_u]
    bx, by = nodes_x[route_v], nodes_y[route_v]

    px = track_x[:, None]
    py = track_y[:, None]
    dx = (bx - ax)[None, :]
    dy = (by - ay)[None, :]
    l2 = np.where(dx*dx + dy*dy == 0, 1e-9, dx*dx + dy*dy)
    c_ax = ax[None, :]
    c_ay = ay[None, :]

    t_proj = np.clip(((px - c_ax)*dx + (py - c_ay)*dy) / l2, 0.0, 1.0)
    proj_x = c_ax + t_proj * dx
    proj_y = c_ay + t_proj * dy
    d2_mat = (px - proj_x)**2 + (py - proj_y)**2

    dp = np.full((N, M), np.inf, dtype=np.float32)
    parent = np.zeros((N, M), dtype=np.int32)
    init_window = min(12, M)
    dp[0, :init_window] = d2_mat[0, :init_window]

    for i in range(1, N):
        prev = dp[i - 1]
        stacked = []
        for shift in range(max_advance + 1):
            if shift == 0:
                s = prev
            else:
                s = np.pad(prev[:-shift], (shift, 0), constant_values=np.inf)
            stacked.append(s)
        stacked = np.array(stacked)
        best_shift = np.argmin(stacked, axis=0)
        min_vals = np.min(stacked, axis=0)
        dp[i] = min_vals + d2_mat[i]
        parent[i] = np.clip(np.arange(M) - best_shift, 0, M - 1)

    best_end = np.argmin(dp[-1])
    path = [best_end]
    for i in range(N - 1, 0, -1):
        path.append(parent[i, path[-1]])
    path.reverse()
    path_arr = np.array(path, dtype=np.int64)

    aligned_arcs = route[path_arr]
    aligned_dists = np.sqrt(d2_mat[np.arange(N), path_arr])

    alignment_metrics = {
        "mean_dist": float(np.mean(aligned_dists)),
        "median_dist": float(np.median(aligned_dists)),
        "max_dist": float(np.max(aligned_dists))
    }
    return aligned_arcs, alignment_metrics
