"""Compare binary candidate scorers, grouped by GPS trip."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
POINT = ["trip_id", "point_idx"]
IDS = ["arc_id", "reference_arc_id", "from_node_id", "to_node_id"]
METADATA = set(POINT + IDS + ["label", "partition"])

MODELS = {
    "LogisticRegression": ("Linear", LogisticRegression(max_iter=1000, random_state=SEED), True),
    "GaussianNB": ("Probabilistic", GaussianNB(), False),
    "KNN": ("Instance-based", KNeighborsClassifier(n_neighbors=7, weights="distance"), True),
    "RandomForest": ("Bagging ensemble", RandomForestClassifier(
        n_estimators=100, min_samples_leaf=3, random_state=SEED, n_jobs=2), False),
    "HistGradientBoosting": ("Gradient boosting", HistGradientBoostingClassifier(
        max_iter=100, early_stopping=False, random_state=SEED), False),
    "MLP": ("Neural network", MLPClassifier(
        hidden_layer_sizes=(32,), max_iter=100, random_state=SEED), True),
}


def parser(description):
    args = argparse.ArgumentParser(description=description)
    args.add_argument("--train", type=Path, required=True)
    args.add_argument("--student-test", type=Path, required=True)
    args.add_argument("--features", nargs="+", required=True)
    args.add_argument("--output", type=Path, default=ROOT)
    return args


def read_candidates(path, partition, features):
    if len(set(features)) != len(features) or METADATA.intersection(features):
        raise ValueError("Use unique inference-time features and exclude metadata and labels.")
    frame = pd.read_csv(path, dtype={"trip_id": str})
    if frame.empty or frame[POINT + IDS].isna().any().any():
        raise ValueError(f"{path} needs candidates with complete point and arc metadata.")
    permitted = {p.name for p in (ROOT / "Student folder" / partition).iterdir() if p.is_dir()}
    if unexpected := set(frame["trip_id"]) - permitted:
        raise ValueError(f"Unexpected trip IDs in {partition}: {sorted(unexpected)}")
    for column in ["point_idx", *IDS]:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any():
            raise ValueError(f"{column} must contain nonnegative integer IDs.")
        frame[column] = values.astype("int64")
    if frame.duplicated(POINT + ["arc_id"]).any():
        raise ValueError("Duplicate candidate arc for a GPS point.")
    if frame.groupby(POINT)["reference_arc_id"].nunique().gt(1).any():
        raise ValueError("Each GPS point needs exactly one aligned reference arc ID.")
    frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame[features]).any().any():
        raise ValueError("Features may contain NaN but may not contain infinities.")
    frame["label"] = frame["arc_id"].eq(frame["reference_arc_id"]).astype(int)
    return frame.sort_values(POINT + ["arc_id"]).reset_index(drop=True)


def pipeline(name, params=None):
    _, classifier, scale = MODELS[name]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler() if scale else "passthrough"),
        ("model", clone(classifier).set_params(**(params or {}))),
    ])


def grouped_folds(train):
    if train["trip_id"].nunique() < 5:
        raise ValueError("GroupKFold(5) requires at least five training trips.")
    folds = list(GroupKFold(5).split(train, train["label"], train["trip_id"]))
    for fit, valid in folds:
        if any(train["label"].iloc[part].nunique() != 2 for part in (fit, valid)):
            raise ValueError("Each CV training and validation partition needs both candidate classes.")
    return folds


def cv_scores(model, train, features, folds):
    rows = []
    for index, (fit, valid) in enumerate(folds, start=1):
        fitted = clone(model).fit(train[features].iloc[fit], train["label"].iloc[fit])
        rows.append({
            "fold": index,
            "train_macro_f1": f1_score(train["label"].iloc[fit], fitted.predict(train[features].iloc[fit]),
                                       average="macro", zero_division=0),
            "validation_macro_f1": f1_score(train["label"].iloc[valid],
                                            fitted.predict(train[features].iloc[valid]),
                                            average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


def evaluate(model, frame, features):
    inputs = frame[features]
    started = perf_counter()
    probabilities = model.predict_proba(inputs)
    inference_time = perf_counter() - started
    labels = frame["label"]
    predicted = model.classes_[probabilities.argmax(axis=1)]
    positive = probabilities[:, list(model.classes_).index(1)]
    candidates = frame.assign(probability=positive)
    top = candidates.loc[candidates.groupby(POINT, sort=False)["probability"].idxmax()]
    previous = top.groupby("trip_id")[["arc_id", "to_node_id", "point_idx"]].shift()
    consecutive = top["point_idx"].eq(previous["point_idx"] + 1)
    connected = top["arc_id"].eq(previous["arc_id"]) | top["from_node_id"].eq(previous["to_node_id"])
    metrics = {
        "Top-1 Acc": top["label"].mean(),
        "Route Continuity": connected[consecutive].mean(),
        "Student Macro F1": f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "Macro Precision": precision_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "Macro Recall": recall_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "Log Loss": log_loss(labels, probabilities, labels=[0, 1]),
        "Inference Time": inference_time,
        "Candidate Miss Rate": 1 - frame.groupby(POINT)["label"].max().mean(),
        "Student Observations": len(top),
    }
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return metrics, matrix


def fit_and_evaluate(model, train, student, features):
    started = perf_counter()
    model.fit(train[features], train["label"])
    elapsed = perf_counter() - started
    metrics, matrix = evaluate(model, student, features)
    return {"Train Time": elapsed, **metrics}, matrix


def save_model(model, name, features, artifacts):
    joblib.dump({"pipeline": model, "features": features, "model_name": name}, artifacts)


def output_dirs(output):
    results, artifacts = output / "results", output / "artifacts"
    for path in (results, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    return results, artifacts


def plot_bars(table, columns, filename, label, errors=None):
    ax = table.set_index("Model")[columns].plot.bar(yerr=errors, ylabel=label, rot=25)
    ax.figure.tight_layout()
    ax.figure.savefig(filename, dpi=150)
    plt.close(ax.figure)


def compare(args):
    train = read_candidates(args.train, "train", args.features)
    student = read_candidates(args.student_test, "student_test", args.features)
    if set(train["trip_id"]) & set(student["trip_id"]):
        raise ValueError("Train and student-test trips must be disjoint.")
    folds = grouped_folds(train)
    results, artifacts = output_dirs(args.output)
    rows = []
    for name, (family, _, _) in MODELS.items():
        model = pipeline(name)
        started = perf_counter()
        scores = cv_scores(model, train, args.features, folds)
        cv_time = perf_counter() - started
        metrics, matrix = fit_and_evaluate(model, train, student, args.features)
        rows.append({
            "Model": name, "Family": family,
            "Key Config": json.dumps(model["model"].get_params(), sort_keys=True),
            "CV Mean": scores["validation_macro_f1"].mean(),
            "CV Std": scores["validation_macro_f1"].std(ddof=0),
            "CV Time": cv_time, **metrics,
        })
        scores.to_csv(results / f"{name}_folds.csv", index=False)
        matrix_frame = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
        matrix_frame.to_csv(results / f"{name}_confusion.csv", index_label="actual / predicted")
        save_model(model, name, args.features, artifacts / f"{name}.joblib")
        if name == "MLP":
            loss = pd.Series(model["model"].loss_curve_, name="training_loss")
            loss.index = range(1, len(loss) + 1)
            loss.to_csv(results / "mlp_loss.csv", index_label="epoch")
            ax = loss.plot(xlabel="Epoch", ylabel="Training loss")
            ax.figure.tight_layout()
            ax.figure.savefig(results / "mlp_loss.png", dpi=150)
            plt.close(ax.figure)

    table = pd.DataFrame(rows)
    table["CV Macro F1"] = [f"{mean:.6f} ± {std:.6f}"
                           for mean, std in zip(table["CV Mean"], table["CV Std"])]
    columns = ["Model", "Family", "Key Config", "CV Macro F1", "Top-1 Acc", "Route Continuity",
               "Student Macro F1", "Log Loss", "Inference Time"]
    table[columns].to_csv(results / "model_comparison.csv", index=False)
    table.to_csv(results / "model_comparison_details.csv", index=False)
    plot_bars(table, ["CV Mean"], results / "cv_f1.png", "CV macro F1",
              errors=table["CV Std"].to_numpy())
    plot_bars(table, ["Top-1 Acc", "Student Macro F1"], results / "student_scores.png", "Score")
    plot_bars(table, ["Log Loss"], results / "log_loss.png", "Candidate log loss")

    ranked = table.assign(selection_score=table["CV Mean"] + table["Top-1 Acc"]).sort_values(
        ["selection_score", "Log Loss", "Inference Time", "Model"],
        ascending=[False, True, True, True],
    )
    best, second = (ranked.iloc[index] for index in (0, 1))
    (results / "selection.json").write_text(json.dumps({"model": best["Model"]}, indent=2) + "\n")
    (results / "selection.md").write_text(
        f"# Selected model\n\n**{best['Model']}** ranked first. Selection gives equal weight to "
        "grouped validation macro F1 and student-test observation top-1 accuracy. Ties use "
        "lower student-test candidate log loss, then lower batch inference time, then model name.\n\n"
        "| Model | CV macro F1 | Student top-1 | Sum | Log loss | Inference (s) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "".join(
            f"| {r['Model']} | {r['CV Mean']:.6f} | {r['Top-1 Acc']:.6f} | "
            f"{r['selection_score']:.6f} | {r['Log Loss']:.6f} | {r['Inference Time']:.6f} |\n"
            for r in (best, second)
        ), encoding="utf-8",
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        compare(parser(__doc__).parse_args())
