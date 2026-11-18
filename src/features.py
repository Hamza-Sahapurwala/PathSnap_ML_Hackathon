"""
PathSnap Feature Engineering & Parquet Caching Module
Member A: Setup, EDA, candidates, labels, features
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.preprocess import load_trip_data, CandidateArcGenerator, align_gps_to_route_dp

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for p in [RESULTS_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "distance",
    "proj_frac",
    "arc_length",
    "arc_bearing",
    "gps_heading",
    "angle_diff",
    "speed",
    "dt",
    "cand_rank",
    "dist_diff_best",
    "from_node_degree",
    "to_node_degree",
    "connects_to_prev_top",
    "rolling_mean_dist"
]

def extract_features_for_trip(
    trip_data: Dict[str, Any],
    top_k: int = 10,
    radius_cap: float = 120.0,
    window_size: int = 3
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Extract candidate features and ground-truth labels for a single trip.
    Returns:
        - df_features: DataFrame with [trip_id, point_idx, arc_id, label] + FEATURE_COLUMNS
        - trip_metrics: {miss_rate, mean_align_dist, median_align_dist, max_align_dist}
    """
    trip_id = trip_data["trip_id"]
    tx = trip_data["track_x"]
    ty = trip_data["track_y"]
    tt = trip_data["track_t"]
    nx = trip_data["nodes_x"]
    ny = trip_data["nodes_y"]
    arcs = trip_data["arcs"]
    route = trip_data.get("route", None)
    N = len(tx)

    # 1. Candidate Generation via KD-tree
    gen = CandidateArcGenerator(nx, ny, arcs, top_k=top_k, radius_cap=radius_cap)
    cands = gen.generate_candidates(tx, ty)
    cand_arcs = cands["cand_arcs"] # (N, K)
    distances = cands["distances"] # (N, K)
    proj_fracs = cands["proj_fracs"] # (N, K)
    K = top_k

    # 2. Kinematics (Speed, GPS heading, dt) from neighbouring points
    dx = np.gradient(tx)
    dy = np.gradient(ty)
    dt_arr = np.gradient(tt)
    dt_arr = np.where(dt_arr <= 0, 1.0, dt_arr)

    gps_heading_pts = np.arctan2(dy, dx)
    speed_pts = np.sqrt(dx*dx + dy*dy) / dt_arr

    # Broadcast to (N, K)
    gps_heading = np.repeat(gps_heading_pts[:, None], K, axis=1)
    speed = np.repeat(speed_pts[:, None], K, axis=1)
    dt = np.repeat(dt_arr[:, None], K, axis=1)

    # 3. Candidate arc geometry & bearings
    u = arcs[cand_arcs, 0] # (N, K)
    v = arcs[cand_arcs, 1] # (N, K)
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]

    arc_length = np.sqrt((bx - ax)**2 + (by - ay)**2)
    arc_bearing = np.arctan2(by - ay, bx - ax)

    # Angle difference between vehicle movement heading and arc bearing
    raw_angle_diff = np.abs(gps_heading - arc_bearing)
    angle_diff = np.where(raw_angle_diff > np.pi, 2 * np.pi - raw_angle_diff, raw_angle_diff)

    # 4. Relative rank & distance features
    cand_rank = np.tile(np.arange(K, dtype=np.int32), (N, 1))
    dist_diff_best = distances - distances[:, [0]]

    # 5. Graph node degrees (in + out)
    node_degrees = np.bincount(arcs[:, 0], minlength=len(nx)) + np.bincount(arcs[:, 1], minlength=len(nx))
    from_node_deg = node_degrees[u]
    to_node_deg = node_degrees[v]

    # 6. Topological connection to previous point's top candidate
    connects_to_prev_top = np.zeros((N, K), dtype=np.int32)
    for i in range(1, N):
        prev_top_arc = cand_arcs[i - 1, 0]
        prev_v = arcs[prev_top_arc, 1]
        is_connected = (u[i] == prev_v) | (cand_arcs[i] == prev_top_arc)
        connects_to_prev_top[i] = is_connected.astype(np.int32)

    # 7. Rolling mean of distance over window
    d_best = distances[:, 0]
    rolling_best = pd.Series(d_best).rolling(window=window_size, min_periods=1, center=True).mean().values
    rolling_mean_dist = np.repeat(rolling_best[:, None], K, axis=1)

    # 8. Ground truth labels via Monotonic DP Alignment (if route present)
    labels = np.zeros((N, K), dtype=np.int32)
    miss_rate = 0.0
    align_metrics = {"mean_dist": 0.0, "median_dist": 0.0, "max_dist": 0.0}

    if route is not None and len(route) > 0:
        aligned_arcs, align_metrics = align_gps_to_route_dp(tx, ty, nx, ny, arcs, route)
        labels = (cand_arcs == aligned_arcs[:, None]).astype(np.int32)
        in_cand = np.any(cand_arcs == aligned_arcs[:, None], axis=1)
        miss_rate = float(1.0 - np.mean(in_cand))

    # Reshape all features to 1D table (N * K rows)
    point_idx = np.repeat(np.arange(N, dtype=np.int32), K)
    trip_id_col = np.full(N * K, trip_id)

    df_trip = pd.DataFrame({
        "trip_id": trip_id_col,
        "point_idx": point_idx,
        "arc_id": cand_arcs.reshape(-1),
        "label": labels.reshape(-1),
        "distance": distances.reshape(-1).astype(np.float32),
        "proj_frac": proj_fracs.reshape(-1).astype(np.float32),
        "arc_length": arc_length.reshape(-1).astype(np.float32),
        "arc_bearing": arc_bearing.reshape(-1).astype(np.float32),
        "gps_heading": gps_heading.reshape(-1).astype(np.float32),
        "angle_diff": angle_diff.reshape(-1).astype(np.float32),
        "speed": speed.reshape(-1).astype(np.float32),
        "dt": dt.reshape(-1).astype(np.float32),
        "cand_rank": cand_rank.reshape(-1).astype(np.int32),
        "dist_diff_best": dist_diff_best.reshape(-1).astype(np.float32),
        "from_node_degree": from_node_deg.reshape(-1).astype(np.int32),
        "to_node_degree": to_node_deg.reshape(-1).astype(np.int32),
        "connects_to_prev_top": connects_to_prev_top.reshape(-1).astype(np.int32),
        "rolling_mean_dist": rolling_mean_dist.reshape(-1).astype(np.float32)
    })

    trip_metrics = {
        "miss_rate": miss_rate,
        "mean_align_dist": align_metrics["mean_dist"],
        "median_align_dist": align_metrics["median_dist"],
        "max_align_dist": align_metrics["max_dist"],
        "n_points": N
    }
    return df_trip, trip_metrics

def generate_and_cache_features(
    top_k: int = 10,
    scale_features: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process both train and student_test partitions:
    - Extracts candidates, DP aligned ground-truth labels, and 14 kinematic/topological features
    - Fits StandardScaler STRICTLY on train partition (zero leakage)
    - Saves scaler to artifacts/scaler.pkl
    - Caches features to results/features_train.parquet and results/features_student.parquet
    - Prints dataset shapes, class balance, and miss rate
    """
    print("==================================================================")
    print(f"  PathSnap Candidate Generation & Feature Extraction (K={top_k})  ")
    print("==================================================================")

    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "student_test"

    train_trip_ids = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    test_trip_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    # 1. Process Train Trips
    print(f"\n[1/3] Processing 80 train trips (candidate generation + features)...")
    t0 = time.time()
    train_dfs = []
    train_misses = []
    train_align_dists = []
    total_train_pts = 0

    for idx, tid in enumerate(train_trip_ids):
        trip_data = load_trip_data(train_dir, tid, load_route=True)
        df_t, met = extract_features_for_trip(trip_data, top_k=top_k)
        train_dfs.append(df_t)
        train_misses.append(met["miss_rate"])
        train_align_dists.append(met["mean_align_dist"])
        total_train_pts += met["n_points"]
        if (idx + 1) % 20 == 0 or (idx + 1) == len(train_trip_ids):
            print(f"  --> Processed {idx + 1}/{len(train_trip_ids)} train trips in {time.time() - t0:.1f}s")

    df_train = pd.concat(train_dfs, ignore_index=True)
    train_time = time.time() - t0

    # 2. Process Student-Test Trips
    print(f"\n[2/3] Processing 10 student-test trips (candidate generation + features)...")
    t0 = time.time()
    test_dfs = []
    test_misses = []
    test_align_dists = []
    total_test_pts = 0

    for idx, tid in enumerate(test_trip_ids):
        trip_data = load_trip_data(test_dir, tid, load_route=True)
        df_t, met = extract_features_for_trip(trip_data, top_k=top_k)
        test_dfs.append(df_t)
        test_misses.append(met["miss_rate"])
        test_align_dists.append(met["mean_align_dist"])
        total_test_pts += met["n_points"]

    df_student = pd.concat(test_dfs, ignore_index=True)
    test_time = time.time() - t0
    print(f"  --> Processed {len(test_trip_ids)} student-test trips in {test_time:.1f}s")

    # 3. Fit StandardScaler STRICTLY on Train features (Zero Leakage)
    print(f"\n[3/3] Fitting StandardScaler strictly on train partition (Zero Data Leakage)...")
    scaler = StandardScaler()
    scaler.fit(df_train[FEATURE_COLUMNS])
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  --> Fitted and saved StandardScaler to: {scaler_path}")

    # 4. Save to Parquet
    parquet_train_path = RESULTS_DIR / "features_train.parquet"
    parquet_student_path = RESULTS_DIR / "features_student.parquet"

    df_train.to_parquet(parquet_train_path, index=False, engine="pyarrow")
    df_student.to_parquet(parquet_student_path, index=False, engine="pyarrow")
    print(f"  --> Saved: {parquet_train_path} ({parquet_train_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  --> Saved: {parquet_student_path} ({parquet_student_path.stat().st_size / (1024*1024):.1f} MB)")

    # 5. Print Diagnostic Metrics
    n_train_pos = int((df_train["label"] == 1).sum())
    n_train_neg = int((df_train["label"] == 0).sum())
    n_train_total = len(df_train)
    train_miss_pct = float(np.mean(train_misses) * 100.0)
    train_mean_dist = float(np.mean(train_align_dists))

    n_test_pos = int((df_student["label"] == 1).sum())
    n_test_neg = int((df_student["label"] == 0).sum())
    n_test_total = len(df_student)
    test_miss_pct = float(np.mean(test_misses) * 100.0)
    test_mean_dist = float(np.mean(test_align_dists))

    print("\n" + "="*60)
    print("                 DATASET DIAGNOSTICS & SUMMARY                ")
    print("="*60)
    print(f"TRAIN PARTITION ({len(train_trip_ids)} trips, {total_train_pts:,} GPS points):")
    print(f"  - Parquet Shape: {df_train.shape} ({n_train_total:,} candidate rows)")
    print(f"  - Class Balance: Positive (True Arc): {n_train_pos:,} ({n_train_pos/n_train_total*100:.2f}%)")
    print(f"                   Negative (False Arc): {n_train_neg:,} ({n_train_neg/n_train_total*100:.2f}%)")
    print(f"                   Ratio Negative : Positive = {n_train_neg/n_train_pos:.2f} : 1 (~9:1 for K=10)")
    print(f"  - Candidate Recall (Top-{top_k}): {100.0 - train_miss_pct:.2f}%")
    print(f"  - Candidate Miss Rate (Top-{top_k}): {train_miss_pct:.2f}%")
    print(f"  - Alignment Quality (Mean Distance to Aligned Arc): {train_mean_dist:.2f} metres")
    print("-" * 60)
    print(f"STUDENT-TEST PARTITION ({len(test_trip_ids)} trips, {total_test_pts:,} GPS points):")
    print(f"  - Parquet Shape: {df_student.shape} ({n_test_total:,} candidate rows)")
    print(f"  - Class Balance: Positive (True Arc): {n_test_pos:,} ({n_test_pos/n_test_total*100:.2f}%)")
    print(f"                   Negative (False Arc): {n_test_neg:,} ({n_test_neg/n_test_total*100:.2f}%)")
    print(f"                   Ratio Negative : Positive = {n_test_neg/n_test_pos:.2f} : 1")
    print(f"  - Candidate Recall (Top-{top_k}): {100.0 - test_miss_pct:.2f}%")
    print(f"  - Candidate Miss Rate (Top-{top_k}): {test_miss_pct:.2f}%")
    print(f"  - Alignment Quality (Mean Distance to Aligned Arc): {test_mean_dist:.2f} metres")
    print("="*60)

    return df_train, df_student

if __name__ == "__main__":
    generate_and_cache_features(top_k=10)
