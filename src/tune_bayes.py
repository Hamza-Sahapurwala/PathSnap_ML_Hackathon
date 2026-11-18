"""Tune the selected candidate scorer with seeded Optuna TPE on training trips."""
import json
from time import perf_counter

import joblib
import optuna

from train_models import (
    MODELS, SEED, cv_scores, fit_and_evaluate, grouped_folds, output_dirs, parser,
    pipeline, plt, read_candidates, save_model, threadpool_limits,
)

import pandas as pd


def suggest(trial, name, smallest_fold):
    # Log ranges cover orders of magnitude for regularisation and learning rates.
    if name == "LogisticRegression":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if name == "GaussianNB":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)}
    if name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, min(31, smallest_fold)),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 240, step=30),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if name == "MLP":
        return {
            "hidden_layer_sizes": (trial.suggest_int("hidden_units", 16, 96, step=16),),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    raise ValueError(f"Unknown model: {name}")


def save_plot(ax, path):
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def tune(args):
    results, artifacts = output_dirs(args.output)
    name = json.loads((results / "selection.json").read_text())["model"]
    if name not in MODELS:
        raise ValueError(f"Unknown selected model: {name}")
    train = read_candidates(args.train, "train", args.features)
    student = read_candidates(args.student_test, "student_test", args.features)
    if set(train["trip_id"]) & set(student["trip_id"]):
        raise ValueError("Train and student-test trips must be disjoint.")
    folds = grouped_folds(train)
    smallest_fold = min(len(fit) for fit, _ in folds)

    def objective(trial):
        params = suggest(trial, name, smallest_fold)
        scores = cv_scores(pipeline(name, params), train, args.features, folds, score_train=False)
        trial.set_user_attr("model_params", params)
        trial.set_user_attr("fold_macro_f1", scores["validation_macro_f1"].tolist())
        return scores["validation_macro_f1"].mean()

    trials_csv = results / "optuna_trials.csv"

    def record_trials(study, _trial):
        temporary = trials_csv.with_suffix(".csv.tmp")
        study.trials_dataframe().to_csv(temporary, index=False)
        temporary.replace(trials_csv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    started = perf_counter()
    study.optimize(objective, n_trials=args.trials, callbacks=[record_trials])
    search_time = perf_counter() - started
    save_plot(optuna.visualization.matplotlib.plot_optimization_history(study),
              results / "optuna_history.png")

    # Relative importance is undefined when all trials have the same objective.
    if len({trial.value for trial in study.trials}) > 1:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=SEED)
        ax = optuna.visualization.matplotlib.plot_param_importances(study, evaluator=evaluator)
    else:
        _, ax = plt.subplots()
        ax.text(0.5, 0.5, "Parameter importance is undefined\nAll trial scores are equal.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    save_plot(ax, results / "optuna_importance.png")

    best_params = study.best_trial.user_attrs["model_params"]
    (results / "best_params.json").write_text(json.dumps({
        "model": name, "parameters": best_params, "best_trial": study.best_trial.number,
        "cv_macro_f1": study.best_value, "seed": SEED, "trials": len(study.trials),
    }, indent=2) + "\n")

    untuned = pipeline(name)
    started = perf_counter()
    untuned_scores = cv_scores(untuned, train, args.features, folds, score_train=False)["validation_macro_f1"]
    cv_time = perf_counter() - started
    tuned = pipeline(name, best_params)
    rows = []
    for version, model, scores, cost in [
        ("Untuned", untuned, untuned_scores, cv_time),
        ("Tuned", tuned, study.best_trial.user_attrs["fold_macro_f1"], search_time),
    ]:
        metrics, _ = fit_and_evaluate(model, train, student, args.features)
        rows.append({
            "Version": version, "Model": name,
            "CV Macro F1": pd.Series(scores).mean(),
            "CV Macro F1 Std": pd.Series(scores).std(ddof=0),
            "CV or Search Time (s)": cost,
            "Compute Cost (s)": cost + metrics["Train Time"],
            **metrics,
        })
    pd.DataFrame(rows).to_csv(results / "tuned_vs_untuned.csv", index=False)
    save_model(tuned, name, args.features, artifacts / "final_model.joblib")
    joblib.dump(tuned[:-1], artifacts / "scaler.joblib")


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--trials", type=int, default=40)
    args = cli.parse_args()
    if args.trials < 40:
        cli.error("--trials must be at least 40")
    with threadpool_limits(limits=6):
        tune(args)
