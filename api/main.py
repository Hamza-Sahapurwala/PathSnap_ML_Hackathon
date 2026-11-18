"""
api/main.py  –  PathSnap FastAPI service (real pipeline).

Environment variables:
    EVAL_DATA_DIR   directory containing student_test trip folders  (default ./data/student_test)
    MODEL_DIR       directory containing artifacts/                  (default ./artifacts)
"""
from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Defaults are anchored to the repository root so the API works from any working directory.
_REPO_ROOT    = Path(__file__).resolve().parent.parent
EVAL_DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", str(_REPO_ROOT / "data" / "student_test")))
MODEL_DIR     = Path(os.environ.get("MODEL_DIR",     str(_REPO_ROOT / "artifacts")))

# ---------------------------------------------------------------------------
# Startup: import real pipeline and trigger artifact loading.
# Any failure here aborts startup immediately with a clear traceback.
# ---------------------------------------------------------------------------
from src.pipeline import predict_trip, _get_artifacts  # noqa: E402  (must import at module level)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once on startup; fail loudly if any are missing."""
    _get_artifacts()   # raises FileNotFoundError / ValueError if anything is wrong
    yield
    # (no teardown needed)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAFE_TRIP_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

def _validate_trip_id(trip_id: str) -> None:
    """Reject path traversal and exotic characters immediately."""
    if not _SAFE_TRIP_RE.match(trip_id):
        raise HTTPException(status_code=400, detail="Invalid trip_id")
    if ".." in trip_id or "/" in trip_id or "\\" in trip_id:
        raise HTTPException(status_code=400, detail="Invalid trip_id")


def _arc_ids_to_polyline(arc_ids: List[int], trip_data: Dict[str, Any]) -> List[List[float]]:
    """Convert a sequence of arc ids to [[lat, lon], ...] using the trip's nodes."""
    nodes_raw = trip_data["nodes_raw"]   # shape (N, 2) lon/lat (col 0: lon, col 1: lat)
    arcs      = trip_data["arcs"]        # shape (M, 2) from_node, to_node
    poly: List[List[float]] = []
    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= len(arcs):
            continue
        u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
        u_pt = [float(nodes_raw[u, 1]), float(nodes_raw[u, 0])]  # [lat, lon]
        v_pt = [float(nodes_raw[v, 1]), float(nodes_raw[v, 0])]  # [lat, lon]
        if not poly or poly[-1] != u_pt:
            poly.append(u_pt)
        if not poly or poly[-1] != v_pt:
            poly.append(v_pt)
    return poly


def _real_metrics(
    arc_ids: np.ndarray,
    df_features,
    trip_data: Dict[str, Any],
    graph,
    hmm_config,
    reference: np.ndarray,
) -> Dict[str, float]:
    """Compute point + continuity + sequence metrics using existing src/ functions."""
    from src.hmm import fill_route, route_from_arcs, HMMConfig
    from src.hmm_experiments import point_metrics, continuity_metrics, sequence_metrics

    judge = HMMConfig()
    filled, _ = fill_route(route_from_arcs(arc_ids), graph)
    pm  = point_metrics(arc_ids, df_features)
    cm  = continuity_metrics(arc_ids, df_features, trip_data, graph, judge)
    sm  = sequence_metrics(filled, reference.tolist())

    # Clamp all ratio values to [0, 1]
    def _clamp(v: float) -> float:
        if v != v:   # NaN → 0
            return 0.0
        return float(max(0.0, min(1.0, v)))

    top1_val = _clamp(pm["top1_accuracy"])
    f1_val   = _clamp(pm["selection_macro_f1"])
    cont_val = _clamp(cm["continuity_connected"])
    seq_val  = _clamp(sm["seq_f1"])

    return {
        "top1_accuracy":        top1_val,
        "macro_f1":             f1_val,
        "route_continuity":     cont_val,
        "seq_f1":               seq_val,
        # Aliases for UI compatibility
        "top1":                 top1_val,
        "continuity":           cont_val,
    }


def _trips_with_data() -> List[str]:
    """Return sorted trip ids that have at least .track / .nodes / .arcs."""
    if not EVAL_DATA_DIR.is_dir():
        return []
    result = []
    for d in sorted(EVAL_DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        tid = d.name
        if all((d / f"{tid}{ext}").exists() for ext in (".track", ".nodes", ".arcs")):
            result.append(tid)
    return result[:10]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "real"}


@app.get("/trips")
def list_trips():
    trips = _trips_with_data()
    if not trips:
        raise HTTPException(status_code=404, detail="No trips found in EVAL_DATA_DIR")
    return {"trips": trips}


class PredictRequest(BaseModel):
    trip_id: str


@app.post("/predict")
def predict(req: PredictRequest):
    t_start = time.time()

    _validate_trip_id(req.trip_id)

    trip_path = EVAL_DATA_DIR / req.trip_id
    if not trip_path.is_dir():
        raise HTTPException(status_code=404, detail="Trip not found")

    tid = req.trip_id
    if not all((trip_path / f"{tid}{ext}").exists() for ext in (".track", ".nodes", ".arcs")):
        raise HTTPException(status_code=404, detail="Trip data files incomplete")

    # --- Run real pipeline (never touches .route) ---------------------------
    try:
        pipeline_result = predict_trip(trip_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    classifier_arcs = np.array(pipeline_result["classifier_arcs"], dtype=np.int64)
    hmm_arcs        = np.array(pipeline_result["hmm_arcs"],        dtype=np.int64)

    # --- Reload trip data for metric computation and polyline conversion ----
    from src.preprocess import load_trip_data
    from src.features   import extract_features_for_trip
    from src.hmm        import RoadGraph, HMMConfig
    from src.hmm_experiments import read_reference_route, sequence_metrics

    trip_data = load_trip_data(EVAL_DATA_DIR, tid, load_route=False)
    graph     = RoadGraph.from_trip(trip_data)

    # Feature frame (label-free) needed for point / continuity metrics
    df_features, _ = extract_features_for_trip(trip_data, top_k=10)

    # --- Load ground-truth reference (server-side only, never into predictor)
    route_file = trip_path / f"{tid}.route"
    if route_file.exists():
        reference = np.atleast_1d(np.loadtxt(route_file, dtype=np.int64))
    else:
        reference = np.array([], dtype=np.int64)

    hmm_config = _get_artifacts()[1]

    # --- Metrics for classifier and HMM ------------------------------------
    # The feature frame was built without the route, so its reference_arc_id is -1 everywhere.
    # Align the reference route to the GPS points (same alignment as the training labels) so
    # point-level metrics compare against the real reference. Prediction has already finished.
    from src.hmm import fill_route, route_from_arcs
    from src.preprocess import align_gps_to_route_dp
    if len(reference):
        aligned, _ = align_gps_to_route_dp(trip_data["track_x"], trip_data["track_y"],
                                           trip_data["nodes_x"], trip_data["nodes_y"],
                                           trip_data["arcs"], reference)
        df_features["reference_arc_id"] = aligned[df_features["point_idx"].to_numpy()]
        classifier_metrics = _real_metrics(
            classifier_arcs, df_features, trip_data, graph, hmm_config, reference
        )
        hmm_metrics = _real_metrics(
            hmm_arcs, df_features, trip_data, graph, hmm_config, reference
        )
    else:
        classifier_metrics = hmm_metrics = None  # no reference available for this trip

    latency_ms = int((time.time() - t_start) * 1000)

    # --- Polylines (full driven route: per-point arcs plus the connecting arcs between them)
    predicted_route   = _arc_ids_to_polyline(fill_route(route_from_arcs(hmm_arcs), graph)[0], trip_data)
    classifier_route  = _arc_ids_to_polyline(fill_route(route_from_arcs(classifier_arcs), graph)[0], trip_data)

    # reference_route: convert arc ids from .route to lat/lon
    reference_route = _arc_ids_to_polyline(reference.tolist(), trip_data) if len(reference) else []

    # GPS track
    track_raw = trip_data["track_raw"]  # lon, lat, t
    gps = [{"lat": float(row[1]), "lon": float(row[0])} for row in track_raw]

    # --- Log ----------------------------------------------------------------
    if hmm_metrics is not None:
        print(
            f"[predict] trip_id={tid} latency_ms={latency_ms} "
            f"clf_top1={classifier_metrics['top1_accuracy']:.3f} "
            f"hmm_top1={hmm_metrics['top1_accuracy']:.3f} "
            f"hmm_f1={hmm_metrics['macro_f1']:.3f} "
            f"hmm_cont={hmm_metrics['route_continuity']:.3f}",
            flush=True,
        )
    else:
        print(f"[predict] trip_id={tid} latency_ms={latency_ms} (no reference route)", flush=True)

    return {
        "trip_id":          tid,
        "gps":              gps,
        "predicted_route":  predicted_route,
        "classifier_route": classifier_route,
        "reference_route":  reference_route,
        "metrics": {
            "classifier": classifier_metrics,
            "hmm":        hmm_metrics,
        },
        "latency_ms": latency_ms,
    }
