# PathSnap candidate scoring

Run from the repository root with Python 3.10 or newer.

```bash
python -m pip install -r requirements-models.txt
python src/train_models.py --train data/train_candidates.csv --student-test data/student_test_candidates.csv --features distance_m heading_delta_deg
python src/tune_bayes.py --train data/train_candidates.csv --student-test data/student_test_candidates.csv --features distance_m heading_delta_deg
```

Replace the example feature names with Member A's inference-time columns. Run both
commands with the same inputs and feature order. Include all 80 training trips
and all ten student-testing trips in the report experiments.

## Run against the team’s committed feature tables

The team’s feature export on main at commit `7c08aa4` provides
`results/features_train.parquet` and `results/features_student.parquet`.
Check out those two files from main or pass absolute paths to copies of them.
They use the same candidate schema as the CSV interface and include rows for
points whose aligned arc is missing from the candidate set.

```bash
python src/train_models.py --train results/features_train.parquet --student-test results/features_student.parquet --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist
python src/tune_bayes.py --train results/features_train.parquet --student-test results/features_student.parquet --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist
```

## Candidate CSV contract

Each row is a candidate arc for one observed GPS point. Both CSVs need
`trip_id` (the original zero-padded directory name), `point_idx` (original
zero-based GPS index), `arc_id`, `reference_arc_id` (the single aligned target
for that point), `from_node_id`, `to_node_id`, and the named numeric features.
Trip IDs are checked against the repository's train and student-test folders.
The scripts derive the candidate label from exact arc equality. Keep rows for
points whose reference arc was missed, so their all-negative candidate sets
count against Top-1 accuracy and contribute to the candidate miss rate. Member
A's export must provide at least one candidate per aligned point, including any
points requiring a nearest-arc fallback. The scripts cannot measure points
that are absent from the export. Record alignment and fallback coverage before
interpreting the metrics.

CV macro F1 and the student-test macro precision, recall, F1, confusion matrix
and log loss measure binary candidate classification. Top-1 measures whether
the strongest candidate matches the aligned arc for each point. Equal
probabilities resolve to the lowest arc ID. Continuity measures adjacent
observation pairs within each trip whose predicted arcs repeat or whose
directed endpoints meet. An empty denominator produces a blank continuity
value. Inference time is wall-clock seconds for all student-test candidate
probabilities, including preprocessing. Fit time and per-fold validation F1 are also recorded. The baseline
comparison records training-fold F1 for all models except KNN, whose
training predictions would include each candidate as its own neighbor.

## Optuna search

The comparison selects the highest sum of train CV macro F1 and student-test
Top-1, then resolves ties by lower log loss and lower inference time.
TPE maximises mean GroupKFold(5) validation macro F1 using only training trips.
The selected algorithm's search ranges are defined in `src/tune_bayes.py`.
The linear model searches regularisation strength, Naive Bayes searches
variance smoothing, and KNN searches locality, weighting, and L1/L2 distance.
The forest searches diversity and tree capacity. Boosting searches leaf
capacity, regularisation, and the learning-rate/iteration tradeoff. The MLP
searches hidden width, weight regularisation, and the initial learning rate.
The 40-trial default and seed 42 apply to a serial search. Compute cost
combines baseline CV or TPE search time with the corresponding full-data refit.

## Decoder and API handoff

`artifacts/final_model.joblib` bundles the tuned sklearn pipeline, feature
names in input order, and model name. Call the pipeline's `predict_proba`
on those columns and pass its class-1 probabilities and the corresponding
candidate arc IDs to the decoder. The pipeline already applies imputation and
any needed scaling. `artifacts/scaler.joblib` is the fitted preprocessing
pipeline alone, which includes the scaler or a passthrough for unscaled
models. Applying it separately before the full pipeline would double-preprocess
the input. Record package versions from the training environment and match
them on the API laptop.
