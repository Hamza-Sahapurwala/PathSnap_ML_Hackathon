"""Member C: tune the HMM decoder (``HMMConfig``) on the training trips with Optuna TPE.

Protocol
--------
* Training-trip probabilities are **out-of-fold**: GroupKFold(5) by trip, each fold scored by a
  clone of the final model's pipeline (Member B's RandomForest recipe) fitted on the other four
  folds. In-sample probabilities would be over-confident and teach the decoder to over-trust
  the classifier.
* Only training trips are used here. The student-test trips are scored once, afterwards, by
  ``python -m src.hmm_experiments --config artifacts/hmm_config.json``.
* Trips 00000033 and 00000054 are excluded from the objective (GPS far from the reference route;
  they hold most of the training candidate misses) and reported separately.
* Objective: pooled observation-level Top-1 accuracy over the objective trips. Route-level
  metrics are recorded for every trial but not optimised.
* Fixed seeds; every trial is appended to ``results/hmm_tuning_trials.csv`` as it finishes.

Usage (from the repository root)::

    python -m src.hmm_tuning --trials 60 --n-jobs 12
"""
from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.hmm import (NO_ARC, HMMConfig, RoadGraph, fill_route, load_classifier, positive_class_probabilities,
                     predict_route, top1_predictions)
from src.hmm_experiments import point_metrics, read_reference_route, sequence_metrics
from src.preprocess import load_trip_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_FOLDS = 5
EXCLUDED_TRIPS = ("00000033", "00000054")
CANDIDATE_COLUMNS = ["point_idx", "arc_id", "proj_frac", "distance", "reference_arc_id"]


# --------------------------------------------------------------------------- #
# Out-of-fold probabilities and per-trip inputs
# --------------------------------------------------------------------------- #
def out_of_fold_probabilities(train: pd.DataFrame, bundle: Mapping[str, Any],
                              n_jobs: int = -1) -> tuple[np.ndarray, Dict[str, int]]:
    """P(class 1) for every training candidate from a model that never saw its trip."""
    features = list(bundle["features"])
    x = train[features].astype("float32")
    y = train["arc_id"].eq(train["reference_arc_id"]).astype(int)
    probs = np.full(len(train), np.nan)
    trip_fold: Dict[str, int] = {}
    for fold, (fit, held) in enumerate(GroupKFold(N_FOLDS).split(x, y, train["trip_id"])):
        t0 = perf_counter()
        model = clone(bundle["pipeline"])
        if "model__n_jobs" in model.get_params():
            model.set_params(model__n_jobs=n_jobs)  # does not change a seeded forest's result
        model.fit(x.iloc[fit], y.iloc[fit])
        probs[held] = positive_class_probabilities(model.predict_proba(x.iloc[held]), model.classes_)
        trip_fold.update({t: fold for t in train["trip_id"].iloc[held].unique()})
        print(f"  fold {fold + 1}/{N_FOLDS}: {train['trip_id'].iloc[held].nunique()} held-out trips, "
              f"{perf_counter() - t0:.0f}s", flush=True)
    assert not np.isnan(probs).any()
    return probs, trip_fold


def build_trip_cache(train: pd.DataFrame, probs: np.ndarray, trip_fold: Mapping[str, int],
                     data_dir: Path, cache_dir: Path) -> List[Path]:
    """One pickle per trip with everything a worker needs (GPS, network, candidates, OOF probs, route)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid, idx in train.groupby("trip_id", sort=True).indices.items():
        trip = load_trip_data(data_dir, tid, load_route=False)
        payload = {"trip_id": tid, "fold": trip_fold[tid],
                   **{k: trip[k] for k in ("track_x", "track_y", "track_t", "nodes_x", "nodes_y", "arcs")},
                   "frame": train.iloc[idx][CANDIDATE_COLUMNS].reset_index(drop=True),
                   "probs": probs[idx],
                   "route": read_reference_route(data_dir, tid)}  # evaluation only
        path = cache_dir / f"{tid}.pkl"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# Evaluating one decoder config
# --------------------------------------------------------------------------- #
def trip_counts(path: Path, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Decode one cached trip (``config=None`` = classifier-only) and return summable counts."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    graph = RoadGraph(d["nodes_x"], d["nodes_y"], d["arcs"])
    t0 = perf_counter()
    if config is None:
        res = top1_predictions(len(d["track_x"]), graph.n_arcs, d["frame"], d["probs"])
    else:
        res = predict_route(d["track_x"], d["track_y"], d["track_t"], graph, d["frame"], d["probs"],
                            HMMConfig.from_dict(config), d["trip_id"])
    seconds = perf_counter() - t0
    pm = point_metrics(res.arc_ids, d["frame"])
    filled, breaks = fill_route(res.route, graph)
    sm = sequence_metrics(filled, d["route"])
    a = res.arc_ids
    pair = (a[:-1] != NO_ARC) & (a[1:] != NO_ARC)
    direct = (a[:-1] == a[1:]) | (graph.v[np.maximum(a[:-1], 0)] == graph.u[np.maximum(a[1:], 0)])
    return {"trip_id": d["trip_id"], "fold": d["fold"], "n_eval_points": pm["n_eval_points"],
            "n_correct": pm["n_correct"], "n_pairs": int(pair.sum()), "n_direct": int((pair & direct).sum()),
            "n_seq_match": sm["n_seq_match"], "len_pred": sm["len_pred"], "len_ref": sm["len_ref"],
            "edit_distance": sm["edit_distance"], "longest": max(sm["len_pred"], sm["len_ref"]),
            "n_fill_breaks": breaks, "seconds_decode": seconds}


def summarise(counts: pd.DataFrame) -> Dict[str, float]:
    s = counts.sum(numeric_only=True)
    p = s["n_seq_match"] / s["len_pred"] if s["len_pred"] else 0.0
    r = s["n_seq_match"] / s["len_ref"] if s["len_ref"] else 0.0
    return {"top1_accuracy": s["n_correct"] / s["n_eval_points"],
            "continuity_direct": s["n_direct"] / s["n_pairs"] if s["n_pairs"] else float("nan"),
            "filled_seq_precision": p, "filled_seq_recall": r,
            "filled_seq_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "filled_edit_similarity": 1 - s["edit_distance"] / s["longest"],
            "n_fill_breaks": int(s["n_fill_breaks"]), "seconds_decode": float(s["seconds_decode"])}


class Evaluator:
    """Runs a config over the cached trips in parallel (largest trips first for load balance)."""

    def __init__(self, paths: Sequence[Path], n_jobs: int) -> None:
        self.paths = sorted(paths, key=lambda p: -p.stat().st_size)
        self.parallel = Parallel(n_jobs=n_jobs, backend="loky")

    def __call__(self, config: Optional[HMMConfig]) -> pd.DataFrame:
        cfg = None if config is None else config.to_dict()
        return pd.DataFrame(self.parallel(delayed(trip_counts)(p, cfg) for p in self.paths))


def scores(counts: pd.DataFrame) -> Dict[str, float]:
    """Objective-trip summary, per-fold Top-1, and the excluded trips reported separately."""
    objective = counts[~counts.trip_id.isin(EXCLUDED_TRIPS)]
    out = summarise(objective)
    folds = [summarise(g)["top1_accuracy"] for _, g in objective.groupby("fold", sort=True)]
    out.update({f"fold{k + 1}_top1": v for k, v in enumerate(folds)})
    out.update({"fold_top1_mean": float(np.mean(folds)), "fold_top1_std": float(np.std(folds))})
    excluded = counts[counts.trip_id.isin(EXCLUDED_TRIPS)]
    if len(excluded):
        out.update({f"excluded_{k}": v for k, v in summarise(excluded).items()
                    if k in ("top1_accuracy", "filled_seq_f1")})
    return out


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
BASELINE_PARAMS = {k: getattr(HMMConfig(), k) for k in (
    "emission_power", "beta_m", "beta_gap_m_per_s", "log_disconnect_penalty", "uturn_log_penalty",
    "backward_log_penalty", "distance_weight", "gps_sigma_m", "normalize_transitions")}


def suggest(trial: Any) -> HMMConfig:
    return HMMConfig(
        emission_power=trial.suggest_float("emission_power", 0.2, 3.0, log=True),
        beta_m=trial.suggest_float("beta_m", 1.0, 60.0, log=True),
        beta_gap_m_per_s=trial.suggest_float("beta_gap_m_per_s", 0.0, 10.0),
        log_disconnect_penalty=trial.suggest_float("log_disconnect_penalty", -40.0, -5.0),
        uturn_log_penalty=trial.suggest_float("uturn_log_penalty", -10.0, 0.0),
        backward_log_penalty=trial.suggest_float("backward_log_penalty", -10.0, 0.0),
        distance_weight=trial.suggest_float("distance_weight", 0.0, 1.0),
        gps_sigma_m=trial.suggest_float("gps_sigma_m", 5.0, 50.0),
        normalize_transitions=trial.suggest_categorical("normalize_transitions", [False, True]),
    )


def ablations(best: HMMConfig) -> Dict[str, Optional[HMMConfig]]:
    """Each tuned feature switched back to its baseline value, one at a time."""
    base = HMMConfig()
    variants: Dict[str, Optional[HMMConfig]] = {"classifier_top1": None, "hmm_untuned": base, "hmm_tuned": best}
    for name, field in [("no_uturn_penalty", "uturn_log_penalty"), ("no_backward_penalty", "backward_log_penalty"),
                        ("no_distance_emission", "distance_weight"), ("emission_power_1", "emission_power"),
                        ("untuned_beta", "beta_m"), ("untuned_gap_beta", "beta_gap_m_per_s"),
                        ("untuned_disconnect_penalty", "log_disconnect_penalty")]:
        if getattr(best, field) != getattr(base, field):
            variants[name] = HMMConfig.from_dict({**best.to_dict(), field: getattr(base, field)})
    variants["flip_normalize_transitions"] = HMMConfig.from_dict(
        {**best.to_dict(), "normalize_transitions": not best.normalize_transitions})
    return variants


def main(argv: Optional[Sequence[str]] = None) -> None:
    import optuna

    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--trials", type=int, default=60)
    args.add_argument("--n-jobs", type=int, default=12)
    args.add_argument("--model", type=Path, default=ROOT / "artifacts" / "final_model.joblib")
    args.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "pathsnap_hmm_cache")
    args.add_argument("--trials-out", type=Path, default=ROOT / "results" / "hmm_tuning_trials.csv")
    args.add_argument("--ablations-out", type=Path, default=ROOT / "results" / "hmm_ablations.csv")
    args.add_argument("--config-out", type=Path, default=ROOT / "artifacts" / "hmm_config.json")
    a = args.parse_args(argv)
    started = perf_counter()

    bundle = load_classifier(a.model)
    train = pd.read_parquet(ROOT / "results" / "features_train.parquet")
    train["trip_id"] = train["trip_id"].astype(str)
    train = train.sort_values(["trip_id", "point_idx", "arc_id"]).reset_index(drop=True)
    print(f"train: {train.trip_id.nunique()} trips, {len(train):,} candidates; out-of-fold probabilities ...",
          flush=True)
    probs, trip_fold = out_of_fold_probabilities(train, bundle, n_jobs=a.n_jobs)
    paths = build_trip_cache(train, probs, trip_fold, ROOT / "data" / "train", a.cache_dir)
    evaluate = Evaluator(paths, a.n_jobs)
    print(f"cached {len(paths)} trips in {a.cache_dir} ({perf_counter() - started:.0f}s so far)", flush=True)

    records: List[Dict[str, Any]] = []

    def objective(trial: Any) -> float:
        cfg = suggest(trial)
        t0 = perf_counter()
        s = scores(evaluate(cfg))
        records.append({"trial": trial.number, **trial.params, **s, "seconds_trial": perf_counter() - t0})
        pd.DataFrame(records).to_csv(a.trials_out, index=False)  # saved after every trial
        print(f"trial {trial.number:3d}: top1 {s['top1_accuracy']:.5f} (folds {s['fold_top1_mean']:.4f} "
              f"+- {s['fold_top1_std']:.4f}) filled F1 {s['filled_seq_f1']:.4f} [{perf_counter() - t0:.0f}s]",
              flush=True)
        return s["top1_accuracy"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(BASELINE_PARAMS)  # trial 0 = untuned baseline
    study.optimize(objective, n_trials=a.trials)

    best = HMMConfig.from_dict({**HMMConfig().to_dict(), **study.best_params})
    print(f"\nbest trial {study.best_trial.number}: top1 {study.best_value:.5f} vs baseline "
          f"{records[0]['top1_accuracy']:.5f}", flush=True)

    rows = []
    for name, cfg in ablations(best).items():
        s = scores(evaluate(cfg))
        rows.append({"variant": name, **s})
        print(f"  {name:28s} top1 {s['top1_accuracy']:.5f}  filled F1 {s['filled_seq_f1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(a.ablations_out, index=False)

    best.save(a.config_out,
              objective="pooled observation-level Top-1 accuracy, training trips, out-of-fold probabilities",
              best_trial=study.best_trial.number, best_value=study.best_value,
              baseline_value=records[0]["top1_accuracy"], n_trials=a.trials, sampler="optuna TPE", seed=SEED,
              n_train_trips=len(paths), excluded_from_objective=list(EXCLUDED_TRIPS),
              classifier=str(bundle.get("model_name")), folds=f"GroupKFold({N_FOLDS}) by trip",
              created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print(f"\nwrote {a.trials_out}, {a.ablations_out}, {a.config_out} ({perf_counter() - started:.0f}s total)")


if __name__ == "__main__":
    main()
