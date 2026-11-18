"""
src/pipeline.py  –  Label-free inference pipeline for PathSnap.

predict_trip(trip_dir) runs the full pipeline on one trip directory
(reads .track / .nodes / .arcs only – never .route) and returns
per-GPS-point arc predictions from both the classifier and the HMM.

Artifacts are loaded ONCE at module level (lazy singleton).
If any artifact is missing the module raises immediately with a clear error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_BASE_DIR / "artifacts")))

# ---------------------------------------------------------------------------
# Lazy singleton – loaded once, never reloaded
# ---------------------------------------------------------------------------
_bundle: Dict[str, Any] | None = None
_hmm_config = None


def _load_artifacts() -> None:
    """Load model bundle and HMM config once; raise clearly if anything is missing."""
    global _bundle, _hmm_config

    model_path = MODEL_DIR / "final_model.joblib"
    hmm_path   = MODEL_DIR / "hmm_config.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}\n"
            "Run the training pipeline first to populate the artifacts/ directory."
        )
    if not hmm_path.exists():
        raise FileNotFoundError(
            f"HMM config not found: {hmm_path}\n"
            "Run the HMM tuning step to generate hmm_config.json."
        )

    _bundle = joblib.load(model_path)
    for key in ("pipeline", "features"):
        if key not in _bundle:
            raise ValueError(f"Model bundle at {model_path!r} is missing required key {key!r}.")

    # Import here to avoid circular imports at module load time
    from src.hmm import HMMConfig
    _hmm_config = HMMConfig.load(hmm_path)


def _get_artifacts():
    """Return (bundle, hmm_config), loading on first call."""
    if _bundle is None:
        _load_artifacts()
    return _bundle, _hmm_config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_trip(trip_dir: str | Path) -> Dict[str, List[int]]:
    """Run the full PathSnap inference pipeline on one trip directory.

    Parameters
    ----------
    trip_dir:
        Directory that contains <trip_id>.track, <trip_id>.nodes and
        <trip_id>.arcs  (the .route file is intentionally NOT read).

    Returns
    -------
    dict with keys:
        "point_arcs"      – top-1 candidate arc per GPS point (before model)
        "classifier_arcs" – highest model-probability arc per GPS point
        "hmm_arcs"        – Viterbi-decoded arc per GPS point
    All lists have length == number of GPS points in the track.
    Missing predictions are represented as -1 (NO_ARC).
    """
    trip_dir = Path(trip_dir)
    # The trip_id is the directory name; files live directly inside as <id>/<id>.track etc.
    trip_id = trip_dir.name

    # --- 1. Load data (no route) -------------------------------------------
    from src.preprocess import load_trip_data, CandidateArcGenerator
    from src.features import FEATURE_COLUMNS, extract_features_for_trip

    trip_data = load_trip_data(trip_dir.parent, trip_id, load_route=False)

    N = len(trip_data["track_x"])

    # --- 2. Extract features (label-free: route=None so labels stay 0) ------
    df_features, _ = extract_features_for_trip(trip_data, top_k=10, radius_cap=120.0)

    # --- 3. Load artifacts --------------------------------------------------
    bundle, hmm_config = _get_artifacts()
    features: List[str] = list(bundle["features"])
    pipe = bundle["pipeline"]

    # --- 4. Classifier: predict_proba -> positive-class probability ----------
    missing_cols = [c for c in features if c not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Feature DataFrame is missing columns required by the model: {missing_cols}")

    proba_matrix = pipe.predict_proba(df_features[features])

    # Retrieve the pipeline's classes_ safely (it's a numpy array, so avoid bare `or`)
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        classes = pipe[-1].classes_

    from src.hmm import positive_class_probabilities
    pos_proba = positive_class_probabilities(proba_matrix, classes, positive=1)

    # --- 5. Top-1 raw candidate (rank 0 from CandidateArcGenerator) ---------
    # The candidate generator outputs rows sorted by distance; rank 0 is the nearest arc.
    # We reconstruct per-point top-1 by taking the first occurrence of each point_idx.
    point_arcs = (
        df_features.groupby("point_idx", sort=True)["arc_id"].first().reindex(range(N), fill_value=-1).tolist()
    )

    # --- 6. Classifier top-1 per point (highest positive-class probability) --
    df_scored = df_features[["point_idx", "arc_id", "proj_frac", "distance"]].copy()
    df_scored["prob"] = pos_proba

    classifier_arcs_series = (
        df_scored.loc[df_scored.groupby("point_idx")["prob"].idxmax()]
        .set_index("point_idx")["arc_id"]
        .reindex(range(N), fill_value=-1)
    )
    classifier_arcs = classifier_arcs_series.tolist()

    # --- 7. HMM Viterbi decoding --------------------------------------------
    from src.hmm import RoadGraph, predict_route

    graph = RoadGraph.from_trip(trip_data)

    hmm_result = predict_route(
        track_x=trip_data["track_x"],
        track_y=trip_data["track_y"],
        track_t=trip_data["track_t"],
        graph=graph,
        candidates=df_features[["point_idx", "arc_id", "proj_frac", "distance"]],
        probabilities=pos_proba,
        config=hmm_config,
        trip_id=trip_id,
    )

    hmm_arcs: List[int] = hmm_result.arc_ids.tolist()

    return {
        "point_arcs": point_arcs,
        "classifier_arcs": classifier_arcs,
        "hmm_arcs": hmm_arcs,
    }
