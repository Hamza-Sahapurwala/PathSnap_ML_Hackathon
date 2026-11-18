"""Member C experiments: classifier-only vs classifier + HMM/Viterbi route reconstruction.

Ground truth is used **only here, after prediction**: Member A's aligned ``reference_arc_id``
for point-level metrics and the trip's ``.route`` file for sequence-level metrics. Neither is
ever passed to ``src.hmm``; prediction trips are loaded with ``load_route=False``.

Both methods are scored on the same candidates, the same classifier probabilities and the
same metric code, so any difference comes from the decoder alone.

Usage (from the repository root)::

    python -m src.hmm_experiments --partition student_test
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.hmm import (KIND_DIRECT, KIND_MULTIHOP, KIND_SAME, NO_ARC, HMMConfig, RoadGraph,
                     fill_route, load_classifier, predict_route_for_trip, score_candidates,
                     top1_predictions, transition_log_matrix)
from src.preprocess import load_trip_data

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILES = {"train": "features_train.parquet", "student_test": "features_student.parquet"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Edit distance (insert / delete / substitute, unit costs), one numpy row at a time."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return int(max(a.size, b.size))
    idx = np.arange(b.size + 1, dtype=np.int64)
    prev = idx.copy()
    for i in range(1, a.size + 1):
        best = np.empty_like(prev)
        best[0] = i
        # deletion (from the row above) or substitution / match (diagonal)
        best[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != a[i - 1]))
        # insertions chain along the row: row[j] = min_k(best[k] + j - k)
        prev = np.minimum.accumulate(best - idx) + idx
    return int(prev[-1])


def sequence_metrics(pred: Sequence[int], ref: Sequence[int]) -> Dict[str, float]:
    """Route-level agreement. Precision/recall count arcs as a multiset, so repeats matter."""
    pred, ref = list(pred), list(ref)
    match = sum((Counter(pred) & Counter(ref)).values())
    precision = match / len(pred) if pred else float("nan")
    recall = match / len(ref) if ref else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if match else 0.0
    ed = levenshtein(pred, ref)
    longest = max(len(pred), len(ref))
    return {"seq_precision": precision, "seq_recall": recall, "seq_f1": f1,
            "edit_distance": ed, "edit_similarity": 1 - ed / longest if longest else float("nan"),
            "n_seq_match": match, "len_pred": len(pred), "len_ref": len(ref)}


def point_metrics(arc_ids: np.ndarray, frame: pd.DataFrame) -> Dict[str, float]:
    """Observation-level scores against the aligned reference arc of each exported point.

    The denominator is every point in the candidate export (as in Member B's Top-1), so points
    whose reference arc is missing from the candidates count as errors.
    """
    ref = frame.groupby("point_idx", sort=True)["reference_arc_id"].first()
    points = ref.index.to_numpy()
    pred = arc_ids[points]
    correct = pred == ref.to_numpy()
    has_ref = frame.assign(hit=frame["arc_id"].eq(frame["reference_arc_id"])) \
        .groupby("point_idx", sort=True)["hit"].any().to_numpy()
    chosen = frame["arc_id"].to_numpy() == arc_ids[frame["point_idx"].to_numpy()]
    truth = frame["arc_id"].eq(frame["reference_arc_id"]).to_numpy()
    return {"n_eval_points": int(points.size), "n_correct": int(correct.sum()),
            "n_reference_in_candidates": int(has_ref.sum()),
            "top1_accuracy": float(correct.mean()),
            "accuracy_when_reference_in_candidates": float(correct[has_ref].mean()) if has_ref.any() else float("nan"),
            "candidate_miss_rate": float(1 - has_ref.mean()),
            "n_unpredicted_points": int((pred == NO_ARC).sum()),
            # candidate-level binary view: the chosen candidate is "1", the rest "0"
            "selection_macro_f1": float(f1_score(truth, chosen, labels=[False, True],
                                                 average="macro", zero_division=0))}


def continuity_metrics(arc_ids: np.ndarray, frame: pd.DataFrame, trip: Mapping[str, Any],
                       graph: RoadGraph, cfg: HMMConfig) -> Dict[str, float]:
    """Adjacent-observation continuity of the predicted arcs.

    * ``direct``: same arc, or arc a ends where arc b starts (Member B's definition).
    * ``connected``: additionally allows a multi-hop directed path found within the decoder's
      search bounds at a plausible speed (same rule the HMM uses for its transitions).
    Pairs are consecutive exported points that both have a prediction.
    """
    pts = np.sort(frame["point_idx"].unique())
    i, j = pts[:-1], pts[1:]
    keep = (j == i + 1) & (arc_ids[i] != NO_ARC) & (arc_ids[j] != NO_ARC)
    i, j = i[keep], j[keep]
    a, b = arc_ids[i], arc_ids[j]
    direct = (a == b) | (graph.v[a] == graph.u[b])
    frac = dict(zip(zip(frame["point_idx"].to_numpy().tolist(), frame["arc_id"].to_numpy().tolist()),
                    frame["proj_frac"].to_numpy().tolist()))
    tx, ty, tt = (np.asarray(trip[k], dtype=np.float64) for k in ("track_x", "track_y", "track_t"))
    connected = direct.copy()
    for k in np.flatnonzero(~direct):
        p, q = int(i[k]), int(j[k])
        _, kind = transition_log_matrix(
            graph, np.array([a[k]]), np.array([frac[(p, int(a[k]))]]),
            np.array([b[k]]), np.array([frac[(q, int(b[k]))]]),
            float(math.hypot(tx[q] - tx[p], ty[q] - ty[p])), float(tt[q] - tt[p]), cfg)
        connected[k] = kind[0, 0] in (KIND_SAME, KIND_DIRECT, KIND_MULTIHOP)
    n = int(direct.size)
    return {"n_pairs": n, "n_direct": int(direct.sum()), "n_connected": int(connected.sum()),
            "continuity_direct": float(direct.mean()) if n else float("nan"),
            "continuity_connected": float(connected.mean()) if n else float("nan"),
            "n_breaks": int(n - connected.sum())}


# --------------------------------------------------------------------------- #
# Running a partition
# --------------------------------------------------------------------------- #
def load_candidates(partition: str, results_dir: Path = ROOT / "results") -> Dict[str, pd.DataFrame]:
    """Member A's candidate export, one frame per trip (row order reset per trip)."""
    frame = pd.read_parquet(results_dir / FEATURE_FILES[partition])
    frame["trip_id"] = frame["trip_id"].astype(str)
    return {tid: df.reset_index(drop=True) for tid, df in frame.groupby("trip_id", sort=True)}


def read_reference_route(data_dir: Path, trip_id: str) -> np.ndarray:
    """Ground-truth arc sequence. Evaluation only: never handed to the decoder."""
    return np.atleast_1d(np.loadtxt(data_dir / trip_id / f"{trip_id}.route", dtype=np.int64))


def evaluate_trip(trip_id: str, frame: pd.DataFrame, data_dir: Path, bundle: Mapping[str, Any],
                  configs: Union[HMMConfig, Mapping[str, HMMConfig]],
                  probabilities: Optional[np.ndarray] = None
                  ) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Score classifier-only and each HMM config on one trip. Returns metric rows and per-point predictions.

    ``configs`` maps a method name to a decoder config (a single config is named ``"hmm"``).
    Continuity is always judged with the default search bounds so every method faces the same rule.
    """
    configs = {"hmm": configs} if isinstance(configs, HMMConfig) else dict(configs)
    trip = load_trip_data(data_dir, trip_id, load_route=False)
    t0 = perf_counter()
    graph = RoadGraph.from_trip(trip)
    t_graph = perf_counter() - t0
    t0 = perf_counter()
    probs = score_candidates(bundle, frame) if probabilities is None else probabilities
    t_score = perf_counter() - t0
    n_points = len(trip["track_x"])

    results = {"classifier_top1": top1_predictions(n_points, graph.n_arcs, frame, probs)}
    for name, cfg in configs.items():
        results[name] = predict_route_for_trip(trip, frame, probs, cfg, graph)
    reference = read_reference_route(data_dir, trip_id)  # only after every prediction exists
    judge = HMMConfig()

    rows = []
    for method, res in results.items():
        d = res.diagnostics
        t0 = perf_counter()
        filled, n_fill_breaks = fill_route(res.route, graph)
        t_fill = perf_counter() - t0
        row: Dict[str, Any] = {"trip_id": trip_id, "method": method, "n_points": n_points,
                               "n_arcs": graph.n_arcs, "reference_route_len": int(reference.size)}
        row.update(point_metrics(res.arc_ids, frame))
        row.update(continuity_metrics(res.arc_ids, frame, trip, graph, judge))
        row.update({f"raw_{k}": v for k, v in sequence_metrics(res.route, reference).items()})
        row.update({f"filled_{k}": v for k, v in sequence_metrics(filled, reference).items()})
        searches = d.get("graph_searches", 0)
        hits = d.get("graph_cache_hits", 0)
        row.update({"n_fill_breaks": n_fill_breaks, "n_segments": d.get("n_segments", 1),
                    "n_gap_resets": d.get("n_gap_resets", 0),
                    "seconds_graph_build": t_graph, "seconds_classifier": t_score,
                    "seconds_decode": d["seconds_total"], "seconds_route_fill": t_fill,
                    "seconds_graph_search": d.get("seconds_graph_search", 0.0),
                    "graph_cache_hit_rate": hits / (searches + hits) if searches + hits else float("nan"),
                    "graph_truncated_searches": d.get("graph_truncated_searches", 0)})
        rows.append(row)

    ref_point = frame.groupby("point_idx")["reference_arc_id"].first()
    points = pd.DataFrame({"trip_id": trip_id, "point_idx": np.arange(n_points),
                           "reference_arc_id": ref_point.reindex(range(n_points), fill_value=NO_ARC).to_numpy(),
                           "top1_arc_id": results["classifier_top1"].arc_ids})
    top1_ok = points.top1_arc_id.eq(points.reference_arc_id)
    for row, name in zip(rows[1:], configs):
        points[f"{name}_arc_id"] = results[name].arc_ids
        points[f"{name}_segment_id"] = results[name].segment_ids
        ok = points[f"{name}_arc_id"].eq(points.reference_arc_id)
        row["n_points_fixed_by_hmm"] = int((ok & ~top1_ok).sum())
        row["n_points_broken_by_hmm"] = int((top1_ok & ~ok).sum())
    return rows, points


POOLED = {  # ratio metrics recomputed from summed counts for the pooled aggregate
    "top1_accuracy": ("n_correct", "n_eval_points"),
    "continuity_direct": ("n_direct", "n_pairs"),
    "continuity_connected": ("n_connected", "n_pairs"),
    **{f"{p}_seq_precision": (f"{p}_n_seq_match", f"{p}_len_pred") for p in ("raw", "filled")},
    **{f"{p}_seq_recall": (f"{p}_n_seq_match", f"{p}_len_ref") for p in ("raw", "filled")},
}


def _is_additive(column: str) -> bool:
    return (column.startswith(("n_", "seconds_")) or "_n_" in column or "_len_" in column
            or column.endswith("edit_distance") or column in ("reference_route_len", "graph_truncated_searches"))


def aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """Per method: mean and std across trips, plus pooled ratios (each point / arc weighs equally)."""
    numeric = [c for c in table.columns if c not in ("trip_id", "method")
               and pd.api.types.is_numeric_dtype(table[c])]
    out = []
    for method, group in table.groupby("method", sort=False):
        mean = group[numeric].mean().to_dict()
        std = group[numeric].std(ddof=0).to_dict()
        # pooled: only counts and timings can be summed; every ratio is rebuilt from counts
        pooled = {c: (v if _is_additive(c) else float("nan")) for c, v in group[numeric].sum().items()}
        for metric, (num, den) in POOLED.items():
            pooled[metric] = pooled[num] / pooled[den] if pooled[den] else float("nan")
        pooled["candidate_miss_rate"] = 1 - pooled["n_reference_in_candidates"] / pooled["n_eval_points"]
        for prefix in ("raw", "filled"):
            p, r = pooled[f"{prefix}_seq_precision"], pooled[f"{prefix}_seq_recall"]
            pooled[f"{prefix}_seq_f1"] = 2 * p * r / (p + r) if p + r else 0.0
            longest = group[[f"{prefix}_len_pred", f"{prefix}_len_ref"]].max(axis=1).sum()
            pooled[f"{prefix}_edit_similarity"] = 1 - pooled[f"{prefix}_edit_distance"] / longest
        for name, values in (("ALL_trip_mean", mean), ("ALL_trip_std", std), ("ALL_pooled", pooled)):
            out.append({"trip_id": name, "method": method, **values})
    return pd.DataFrame(out)


def run_partition(partition: str, model_path: Path, configs: Mapping[str, HMMConfig],
                  data_root: Path = ROOT / "data", results_dir: Path = ROOT / "results",
                  trip_ids: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = load_classifier(model_path)
    candidates = load_candidates(partition, results_dir)
    rows, points = [], []
    for tid in trip_ids or sorted(candidates):
        trip_rows, trip_points = evaluate_trip(tid, candidates[tid], data_root / partition, bundle, configs)
        rows += trip_rows
        points.append(trip_points)
        top1, hmm = trip_rows[0], trip_rows[-1]
        print(f"{tid}: top1 acc {top1['top1_accuracy']:.4f} -> {hmm['method']} {hmm['top1_accuracy']:.4f} | "
              f"connected {top1['continuity_connected']:.4f} -> {hmm['continuity_connected']:.4f} | "
              f"filled seq F1 {top1['filled_seq_f1']:.4f} -> {hmm['filled_seq_f1']:.4f} | "
              f"decode {hmm['seconds_decode']:.2f}s", flush=True)
    table = pd.DataFrame(rows)
    return pd.concat([table, aggregate(table)], ignore_index=True), pd.concat(points, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--partition", choices=sorted(FEATURE_FILES), default="student_test")
    args.add_argument("--model", type=Path, default=ROOT / "artifacts" / "final_model.joblib")
    args.add_argument("--config", type=Path, default=None,
                      help="tuned HMMConfig JSON, scored as 'hmm_tuned' next to the untuned baseline")
    args.add_argument("--out", type=Path, default=None, help="comparison CSV path")
    args.add_argument("--points-out", type=Path, default=None, help="per-point predictions CSV path")
    a = args.parse_args(argv)
    configs = {"hmm_untuned": HMMConfig()}
    if a.config:
        configs["hmm_tuned"] = HMMConfig.load(a.config)
    suffix = "" if a.partition == "student_test" else f"_{a.partition}"
    out = a.out or ROOT / "results" / f"hmm_comparison{suffix}.csv"
    points_out = a.points_out or ROOT / "results" / f"hmm_point_predictions{suffix}.csv"
    table, points = run_partition(a.partition, a.model, configs)
    table.to_csv(out, index=False)
    points.to_csv(points_out, index=False)
    out.with_suffix(".config.json").write_text(json.dumps(
        {"partition": a.partition, "model": str(a.model), "config_file": str(a.config) if a.config else None,
         "configs": {k: c.to_dict() for k, c in configs.items()}}, indent=2) + "\n", encoding="utf-8")
    summary = table[table.trip_id == "ALL_pooled"].set_index("method")
    cols = ["top1_accuracy", "continuity_direct", "continuity_connected", "raw_seq_f1", "filled_seq_f1",
            "filled_edit_similarity", "seconds_decode"]
    print("\npooled over trips:\n" + summary[cols].T.to_string())
    print(f"\nwrote {out}, {points_out}")


if __name__ == "__main__":
    main()
