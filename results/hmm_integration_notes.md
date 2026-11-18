# HMM / Viterbi decoder: integration notes (Member C)

Status: **decoder implemented, unit-tested, tuned on the training trips and evaluated on the student-test trips.**
Code: `src/hmm.py`, `src/hmm_experiments.py`; tests: `tests/test_hmm.py`, `tests/test_hmm_experiments.py`.
Results: `results/hmm_comparison.csv` (per trip + aggregates), `results/hmm_point_predictions.csv`.
Written against the pipeline as pushed on `origin/main` (commit `7c08aa4`, Member A) and `origin/feature-models`
(commit `caba802`, Member B). Where something below is marked **OPEN**, it depends on a teammate.

## 1. Data flow

```
GPS track + road network                      src.preprocess.load_trip_data(dir, trip_id, load_route=False)   [A]
        |
candidates + 14 features per (point, arc)     src.features.extract_features_for_trip(trip_data)               [A]
        |
P(candidate is correct)                       src.hmm.score_candidates(bundle, candidates)                    [C, wraps B's model]
        |
HMM / Viterbi                                 src.hmm.predict_route_for_trip(trip, candidates, probs)         [C]
        |
predicted arc per GPS point (+ route)         RouteResult
```

## 2. What the decoder needs (verified against A's real parquet)

`candidates` is a pandas DataFrame with **one row per (GPS point, candidate arc)** for a *single trip*:

| column | dtype | meaning | required |
|---|---|---|---|
| `point_idx` | int | zero-based row of the `.track` file | yes |
| `arc_id` | int | zero-based row of the `.arcs` file (**never a dataframe position**) | yes |
| `proj_frac` | float in [0, 1] | projection of the GPS point along the arc `from_node -> to_node` (A's feature) | yes |
| `distance` | float, metres | perpendicular distance to the arc (A's feature) | only if `distance_weight > 0` |

All other columns (A's remaining features, `label`, `reference_arc_id`, `trip_id`, ...) are ignored by the decoder.
Rows may be in any order. Duplicate `(point_idx, arc_id)` rows keep the most probable one. Rows with an invalid
`point_idx`/`arc_id` are dropped and counted (`diagnostics["n_invalid_rows"]`).

`probabilities` is a 1-D array, **same length and row order as `candidates`**: P(class = 1).

Coordinates must be in the **same metric CRS as A's pipeline** (UTM from `load_trip_data`): `track_x/track_y/track_t`
and `nodes_x/nodes_y/arcs`. This was checked on a real trip: `RoadGraph.length` equals A's `arc_length` feature
and node ids equal A's `from_node_id/to_node_id` (see `test_integration_with_member_a_candidates_on_a_real_trip`).

### Classifier interface (Member B)
`artifacts/final_model.joblib` is a dict `{"pipeline", "features", "model_name"}` (B's `MODEL_TRAINING.md`).
`score_candidates(bundle, frame)` calls `pipeline.predict_proba(frame[features])` and picks the column whose class label
is `1` by **label, not by position** (verified with reversed class order). The pipeline already imputes/scales:
**do not apply A's `artifacts/scaler.pkl` or B's `scaler.joblib` on top** (double scaling).
**Provenance of `artifacts/final_model.joblib` (22 MB):** Member B's selected model is the *untuned* RandomForest, but B's
run saved no model file and B's Optuna tuning was not finished (stopped; the team chose to skip classifier tuning).
Member C rebuilt it with B's own `pipeline("RandomForest")` from `src/train_models.py` (`origin/main` 8be1f3f),
A's `results/features_train.parquet`, B's 14 features in B's order and seed 42 (scikit-learn 1.9.1). Its student-test
Top-1 0.8597599165, route continuity 0.9293472585, macro F1 0.9198221443 and log loss 0.0712856819 equal B's
`results/model_comparison.csv` to 10 digits. It is a rebuild of B's untuned RandomForest, not a file B produced.

## 3. Public API (`src/hmm.py`)

```python
HMMConfig(...)                                   # frozen dataclass; defaults = untuned baseline
HMMConfig.load("artifacts/hmm_config.json")      # will be written after tuning; accepts {"config": {...}} or flat dict

RoadGraph(nodes_x, nodes_y, arcs)                # directed, validated; raises ValueError on bad node ids
RoadGraph.from_trip(trip)                        # reads only nodes_x, nodes_y, arcs

score_candidates(bundle, frame) -> np.ndarray    # positive-class probability per candidate row
load_classifier(path) -> dict                    # joblib.load + key check

predict_route(track_x, track_y, track_t, graph, candidates, probabilities,
              config=None, trip_id=None) -> RouteResult
predict_route_for_trip(trip, candidates, probabilities, config=None, graph=None) -> RouteResult
top1_predictions(n_points, n_arcs, candidates, probabilities) -> RouteResult   # classifier-only baseline
route_from_arcs(arc_ids) -> list[int]            # drops -1, collapses *consecutive* repeats only
fill_route(route, graph) -> (list[int], n_breaks) # inserts shortest directed connecting arcs between route arcs
```

`RouteResult.route` only holds arcs that some GPS point was matched to; the reference `.route` also lists the arcs
driven between observations. `fill_route(result.route, graph)` adds them (bounded search, 3 km); pairs it cannot connect
stay adjacent and are counted in `n_breaks`. **Use the filled route for display and route-level scoring.**

`RouteResult`:

| field | meaning |
|---|---|
| `arc_ids` | `int64[n_points]`, predicted arc per GPS point **in original order**; `-1` = no prediction |
| `segment_ids` | `int64[n_points]`, decoding segment (`-1` where missing); increments at each reset |
| `route` | arcs with consecutive repeats removed, missing points skipped (revisits preserved) |
| `diagnostics` | counts and timings (missing points, resets, transition kinds on the chosen path, graph-search stats) |
| `to_frame()` | DataFrame `point_idx, arc_id, segment_id` |

`predict_route*` takes **no ground truth**. `predict_route_for_trip` reads only `track_x/track_y/track_t/nodes_x/nodes_y/arcs/trip_id`
from the trip dict; a test fails if it ever touches `trip["route"]`. Evaluation against `.route` lives elsewhere (not in this module).

### Example for the API (Member D)

```python
from src.preprocess import load_trip_data
from src.features import extract_features_for_trip
from src.hmm import (HMMConfig, RoadGraph, fill_route, load_classifier, predict_route_for_trip,
                     score_candidates)

bundle = load_classifier("artifacts/final_model.joblib")            # load once at startup
cfg = HMMConfig.load("artifacts/hmm_config.json")                   # after tuning; HMMConfig() until then

trip = load_trip_data(eval_dir, trip_id, load_route=False)          # never load the .route at inference
candidates, _ = extract_features_for_trip(trip)                     # A: 10 candidates per point
probs = score_candidates(bundle, candidates)                        # B's model
graph = RoadGraph.from_trip(trip)
result = predict_route_for_trip(trip, candidates, probs, cfg, graph) # C
route, n_breaks = fill_route(result.route, graph)                   # C: full arc sequence
response = {"trip_id": trip_id, "route": route, "arc_per_point": result.arc_ids.tolist()}
# server-side accuracy: compare arc_per_point with the aligned reference arcs and `route` with the .route file,
# using the same functions as src/hmm_experiments.py (point_metrics, sequence_metrics)
```

## 4. Behaviour guarantees

| situation | behaviour |
|---|---|
| point with no usable candidates | `arc_ids[i] = -1`, `segment_ids[i] = -1`; decoding bridges over it if the time gap is small |
| time gap between observations `> gap_reset_s` (default 60 s) | new independent segment (no forced connection) |
| non-finite GPS value | that point is treated as missing |
| arc unreachable from the previous arc | bounded floor `log_disconnect_penalty` (never `-inf`); counted in `diagnostics["path_transition_kinds"]` |
| reachable only at an impossible speed (`> max_speed_mps`) | treated like disconnected, counted separately as `implausible_speed` |
| invalid `arc_id` / `point_idx` rows | dropped, counted |
| invalid node ids in the network | `RoadGraph` raises `ValueError` (fail loudly, do not guess) |
| every score `-inf` at a step | new segment instead of a crash |

## 5. Things Member D must know

1. `RoadGraph` caches shortest paths and is **not thread-safe**. Build one per request (cost about 0.03 s even for a 307k-arc network).
2. Do **not** use B's `read_candidates()` for inference: it hardcodes `Student folder/{train,student_test}`, rejects unknown trip ids and needs `reference_arc_id`.
3. Output order is the order of the `.track` file; `arc_ids` always has `n_points` entries.
4. `extract_features_for_trip` on a trip loaded with `load_route=False` yields `label = 0` and `reference_arc_id = -1`; the decoder ignores both. Ground truth must stay server-side.
5. Determinism: no randomness; ties resolve to the lowest arc id.

## 6. Measured on the real student-test trips (untuned baseline, **stand-in classifier, not Member B's model**)

Pipeline check only: A's real candidates, real graphs, default `HMMConfig()`, and a throwaway HistGradientBoosting classifier
trained on A's train partition with B's recipe. These are timing and sanity facts, **not** experimental results, and nothing
was written to `results/`. Decode time per trip: 0.07 s to 1.0 s for the 10 student-test trips (262 to 3,958 points);
0.67 s for the largest network (training trip 00000013, 307,392 arcs); 4.6 s for the longest trip (training trip 00000054, 19,313 points).
Graph search accounted for under 10% of decode time (cache hit rate above 90%); no search hit the node budget.
Real results with the final model (student-test, pooled over the 10 trips) are in `results/hmm_comparison.csv`:
Top-1 0.8598 (classifier only) -> 0.9284 (HMM untuned) -> 0.9390 (HMM tuned). Full write-up: `results/hmm_notes.md`.

## 7. Status

Done: decoder tuning on the training trips (`artifacts/hmm_config.json`, `results/hmm_tuning_trials.csv`), ablations
(`results/hmm_ablations.csv`), Folium maps (`src/make_map.py`, `maps/hmm_student_test_*.html/.png`), write-up
(`results/hmm_notes.md`). **The API should load `artifacts/hmm_config.json`** (tuned), not `HMMConfig()`.
`src/make_map.build_trip_map(trip, {"hmm_tuned": result.arc_ids})` can draw the API's prediction (no reference layer
unless one is passed).
