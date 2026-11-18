#!/usr/bin/env bash
# =============================================================================
# PathSnap – End-to-End Execution Pipeline with Fixed Random Seeds
# Stages: eda -> preprocess/features -> train -> tune -> hmm -> maps -> tests
# =============================================================================
set -euo pipefail

SEED=42
export PYTHONHASHSEED=$SEED

echo "=========================================================="
echo "  [Stage 1/7] Exploratory Data Analysis (EDA)            "
echo "=========================================================="
python -m src.eda

echo "=========================================================="
echo "  [Stage 2/7] Preprocessing & Feature Extraction         "
echo "=========================================================="
python -m src.features

echo "=========================================================="
echo "  [Stage 3/7] Baseline Models Training & Comparison      "
echo "=========================================================="
python src/train_models.py \
  --train results/features_train.parquet \
  --student-test results/features_student.parquet \
  --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist \
  --seed $SEED

echo "=========================================================="
echo "  [Stage 4/7] Bayesian Hyperparameter Optimisation       "
echo "=========================================================="
python src/tune_bayes.py \
  --train results/features_train.parquet \
  --student-test results/features_student.parquet \
  --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist \
  --seed $SEED

echo "=========================================================="
echo "  [Stage 5/7] HMM Viterbi Decoder Tuning & Evaluation    "
echo "=========================================================="
python -m src.hmm_tuning --trials 60 --seed $SEED
python -m src.hmm_experiments --partition student_test --config artifacts/hmm_config.json

echo "=========================================================="
echo "  [Stage 6/7] Interactive & Static Folium Map Generation  "
echo "=========================================================="
python -m src.make_map --trip 00000077 00000065 00000052 --png

echo "=========================================================="
echo "  [Stage 7/7] Automated Integration & API Testing        "
echo "=========================================================="
python -m pytest tests/ -v

echo "=========================================================="
echo "  Pipeline execution completed successfully!              "
echo "=========================================================="
