# PathSnap – High-Precision Map Matching System

PathSnap is an end-to-end, privacy-preserving map-matching engine designed to reconstruct vehicle journeys from sparse, noisy GPS trajectories onto directed OpenStreetMap road networks. It combines spatial KD-Tree candidate pruning, topological and kinematic feature engineering, a calibrated machine learning classifier, and a directed network-constrained Hidden Markov Model (HMM) solved via Viterbi decoding.

---

## 1. Project Directory Structure

```text
PathSnap_ML_Hackathon/
├── api/                        # FastAPI production backend
│   ├── __init__.py
│   └── main.py                 # REST endpoints: /health, /trips, /predict
├── ui/                         # Streamlit evaluator frontend
│   ├── app.py                  # Folium interactive visualization & metrics UI
│   ├── requirements.txt
│   └── run.bat
├── src/                        # Core algorithmic modules
│   ├── eda.py                  # Exploratory Data Analysis & anomaly profiling
│   ├── preprocess.py           # Coordinate projection, KD-tree, DP ground-truth labeler
│   ├── features.py             # 14 kinematic/topological features & caching
│   ├── train_models.py         # 5 baseline classifiers training & comparison
│   ├── tune_bayes.py           # Optuna Bayesian hyperparameter optimisation
│   ├── hmm.py                  # Directed graph Dijkstra, emission/transition, Viterbi
│   ├── hmm_tuning.py           # Bayesian optimization of HMM decoding parameters
│   ├── hmm_experiments.py      # Classifier-only vs HMM benchmarking
│   ├── make_map.py             # Folium HTML & PNG map generator
│   └── pipeline.py             # Isolated, label-free inference pipeline
├── artifacts/                  # Model weights and configuration files
│   ├── final_model.joblib      # Calibrated RandomForest classifier pipeline
│   ├── scaler.pkl              # Train-fitted feature scaler
│   └── hmm_config.json         # Tuned HMM hyperparameter configuration
├── results/                    # Experimental results, metrics, plots, and logs
│   ├── model_comparison.csv    # 5-model evaluation across CV and student-test
│   ├── hmm_comparison.csv      # Per-trip & pooled comparison: Classifier vs HMM
│   ├── hmm_ablations.csv       # Parameter sensitivity ablations
│   ├── hmm_tuning_trials.csv   # 60 Optuna HMM tuning trials
│   ├── cv_f1.png, log_loss.png # Comparative diagnostic plots
│   └── eda/                    # Dataset summaries, anomaly logs, distribution plots
├── tests/                      # Automated test suite
│   ├── test_pipeline.py        # Pipeline integrity and route leakage tests
│   └── test_api.py             # End-to-end API integration tests (57 assertions)
├── scripts/
│   └── run_all.sh              # Sequential reproduction script with fixed seeds
├── requirements.txt            # Pinned production dependencies
├── pytest.ini                  # Pytest configuration
└── README.md                   # System documentation and deployment guide
```

---

## 2. Environment Installation

Ensure Python 3.10+ (tested on Python 3.13) is installed:

```bash
# 1. Clone repository and navigate inside
git clone https://github.com/Hamza-Sahapurwala/PathSnap_ML_Hackathon.git
cd PathSnap_ML_Hackathon

# 2. Create and activate virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. How to Run Each Pipeline Stage in Order

Every stage is deterministic and pinned with seed `42`:

```bash
# Stage 1: Exploratory Data Analysis (EDA)
python -m src.eda

# Stage 2: Preprocessing, Candidate Generation, and Feature Caching
python -m src.features

# Stage 3: Train and Compare the 5 Baseline Classifiers
python src/train_models.py \
  --train results/features_train.parquet \
  --student-test results/features_student.parquet \
  --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist \
  --seed 42

# Stage 4: Bayesian Hyperparameter Optimisation for Classifier
python src/tune_bayes.py \
  --train results/features_train.parquet \
  --student-test results/features_student.parquet \
  --features distance proj_frac arc_length arc_bearing gps_heading angle_diff speed dt cand_rank dist_diff_best from_node_degree to_node_degree connects_to_prev_top rolling_mean_dist \
  --seed 42

# Stage 5: HMM Viterbi Decoder Tuning & Evaluation
python -m src.hmm_tuning --trials 60 --seed 42
python -m src.hmm_experiments --partition student_test --config artifacts/hmm_config.json

# Stage 6: Generate Interactive and Static Folium Evaluation Maps
python -m src.make_map --trip 00000077 00000065 00000052 --png

# Stage 7: Run Integration & Unit Tests
python -m pytest tests/ -v
```

Alternatively, run the unified automated script:
```bash
bash scripts/run_all.sh
```

---

## 4. Live Multi-Laptop Demo Setup

The system is decoupled into an API backend and a Streamlit UI, allowing deployment across two laptops on the same Wi-Fi / LAN network.

### Laptop 1: API Server (Backend)

1. **Find Laptop 1's LAN IP address:**
   ```cmd
   ipconfig | findstr /i "IPv4"
   ```
   *Example output: `IPv4 Address. . . . . . . . . . . : 192.168.1.45`*

2. **Configure Windows Firewall (Allow inbound TCP port 8000):**
   ```cmd
   netsh advfirewall firewall add rule name="FastAPI 8000" dir=in action=allow protocol=TCP localport=8000
   ```

3. **Start the FastAPI server:**
   ```cmd
   set EVAL_DATA_DIR=./data/student_test
   set MODEL_DIR=./artifacts
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *(Note: Binding to `0.0.0.0` allows outside requests. Do not browse to `0.0.0.0`; verify locally using `http://127.0.0.1:8000/health`).*

---

### Laptop 2: Evaluator UI (Frontend)

1. Ensure Laptop 2 is connected to the same Wi-Fi network as Laptop 1.
2. In the repository folder on Laptop 2, start the Streamlit client pointing to Laptop 1's IP:
   ```cmd
   set API_URL=http://192.168.1.45:8000
   streamlit run ui/app.py
   ```
3. Open `http://localhost:8501` in the browser. You can also dynamically update the API URL via the left sidebar input.
4. Select any student-test trip (e.g. `00000077`) to visualize the GPS points, predicted route, classifier-only route, and ground truth route on Folium with live metrics.

---

## 5. Experimental Results Summary

### 5.1 Classifier Baseline Comparison (`results/model_comparison.csv`)

| Model | Family | CV Macro F1 | Student Top-1 Acc | Route Continuity | Student Macro F1 | Log Loss | Inference Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | **Bagging Ensemble** | **0.8872 ± 0.046** | **0.8598** | **0.9293** | **0.9198** | **0.0713** | **0.467** |
| MLP | Neural Network | 0.8809 ± 0.042 | 0.8481 | 0.9314 | 0.9120 | 0.0801 | 0.041 |
| KNN (k=7) | Instance-based | 0.8623 ± 0.045 | 0.8376 | 0.9137 | 0.9055 | 0.3989 | 252.769 |
| LogisticRegression | Linear | 0.8538 ± 0.045 | 0.8373 | 0.9289 | 0.8919 | 0.1130 | 0.034 |
| GaussianNB | Probabilistic | 0.7309 ± 0.016 | 0.7896 | 0.9302 | 0.7285 | 0.5991 | 0.073 |

*Decision: **RandomForest** achieved the highest validation Macro F1 (0.8872) and held-out student-test Top-1 accuracy (0.8598) with well-calibrated probabilities (log loss 0.0713).*

---

### 5.2 Sequence Decoding: Classifier vs. HMM (`results/hmm_comparison.csv`)

Evaluated across all 10 held-out student test trips (19,160 GPS points, 3,477 reference arcs):

| Metric | Classifier Only (Top-1) | HMM Baseline (Untuned) | **HMM Tuned (Trial 51)** | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Observation Top-1 Accuracy** | 0.8598 | 0.9284 | **0.9390** | **+7.92%** |
| **Per-Trip Mean Top-1 Accuracy** | 0.8736 ± 0.071 | 0.9361 ± 0.048 | **0.9407 ± 0.051** | **+6.71%** |
| **Adjacent Connected Continuity** | 0.9558 | 0.9988 | **0.9994** | **+4.36%** |
| **Disconnected Transitions** | 846 | 23 | **12** | **-98.6%** |
| **Filled Route Precision** | 0.2326 | 0.7930 | **0.8590** | **+269.3%** |
| **Filled Route Recall** | 0.9888 | 0.9914 | **0.9914** | **+0.26%** |
| **Sequence Route F1-Score** | 0.3766 | 0.8811 | **0.9204** | **+144.4%** |
| **Sequence Edit Similarity** | 0.2318 | 0.7908 | **0.8565** | **+269.5%** |
| **Mean Decode Time per Trip** | **0.010 s** | 0.491 s | 0.521 s | Real-time capable |

*Key finding: Raw classifier decisions exhibit high ping-pong oscillations between antiparallel arcs (producing 846 disconnected transitions). The tuned HMM enforces directed graph continuity, lifting sequence Route F1 from 0.3766 to **0.9204**.*
