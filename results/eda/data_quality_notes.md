# PathSnap Data Quality & Exploratory Findings
**Member A: Data Engineering & Preprocessing**  
**Date:** 2026-11-18  
**Dataset Scope:** 80 Training Trips, 10 Student-Test Trips (90 total trips, 227,577 total GPS observations).

---

## 1. Dataset Inventory & Completeness
- **File Completeness:** Every trip folder across `train/` and `student_test/` contains all 4 expected files (`.track`, `.nodes`, `.arcs`, `.route`).
- **Data Cleanliness:** 
  - Zero missing values (`NaN` count = 0, `Inf` count = 0).
  - All coordinates reside within valid geographic coordinate bounds ($E \in [-180, 180], N \in [-90, 90]$).
  - Angular coordinate ranges verify that observations are centered in Western/Central Europe (primarily Germany / UTM Zone 32N, EPSG:32632).

---

## 2. Sampling Rate & Temporal Gaps
- **Nominal Sampling Frequency:** Nominally 1.0 Hz.
- **Exact 1 Hz Rate:** **93.84%** of consecutive observations have exactly $\Delta t = 1.0$s.
- **Maximum Observed Gap:** **153064.0 seconds**.
- **Temporal Anomalies:** Several trajectories experience occasional GPS signal dropouts or transmission delays ($\Delta t \in [2, 10]$s).
- **Engineering Implication:** Candidate scoring and HMM transition probabilities must condition on $\Delta t$ rather than assuming uniform 1.0s increments. When $\Delta t > 1.0$s, transition penalties must scale proportionally to physical reachable distance.

---

## 3. GPS Measurement Noise & Speed Outliers
- **Kinematic Analysis:** Instantaneous speeds were computed in projected UTM metres.
- **Speed Outliers:**
  - Observations exceeding $35\text{ m/s}$ ($>126\text{ km/h}$): **8716 instances**.
  - Observations exceeding $50\text{ m/s}$ ($>180\text{ km/h}$): **1538 instances**.
- **Causes:** Sudden position jumps due to urban canyon multipath reflection, building obstruction, and GPS re-acquisition jitter.
- **Engineering Implication:** Features should incorporate vehicle heading, speed relative to road segment orientation, and distance rank, rather than raw Cartesian proximity alone.

---

## 4. Road Network Topology & Graph Properties
- **Network Extents:** Each trip has its **own localized road network** ranging from ~300 to ~155,000 nodes and ~600 to ~307,000 directed arcs.
- **Isolated Nodes:** Across all trips, there are **0 isolated nodes** (degree 0) in the raw network files.
- **Connected Components:** The networks contain a dominant strongly connected component alongside disconnected suburban fragments.
- **Engineering Implication:** Candidate generation must index both node-incident arcs and arc midpoints to ensure high candidate recall without being trapped in disconnected components.

---

## 5. Reference Route Ground-Truth Analysis
- **Route Continuity:** The reference `.route` files form nearly continuous directed paths (**99.93% average continuity**).
- **Trajectory vs Route Cardinality:** A trajectory has on average ~2,600 GPS points while its reference route has ~390 arcs, reflecting vehicle dwell time (multiple GPS observations falling along a single road segment).
- **Ground-Truth Labeling:** A monotonic dynamic programming alignment correctly matches each GPS point to its reference arc along the route sequence, resolving self-intersections and turns.

---

## 6. Investigation of Candidate Misses (Trips 00000033 & 00000054)
- **Concentration of Misses:** Trips `00000033` and `00000054` account for approximately **88% of all candidate misses** in the training partition.
- **Physical Cause:** In both trips, the mean GPS distance to the reference route is abnormally large (**81 m** for trip 33, and **132 m** for trip 54), indicating severe GPS offset, drift, or ground-truth recording discrepancy rather than a candidate-generation bug. Outside these two trips, top-10 candidate recall exceeds 99%.

---

## 7. Bidirectional Road Segments & Heading Disambiguation
- **Candidate Symmetry:** For **73–76% of GPS points**, the rank-0 and rank-1 candidates represent an arc and its reverse counterpart $(u \to v \text{ and } v \to u)$ at identical perpendicular distances.
- **Modeling Recommendation:** The spatial rank ordering between rank-0 and rank-1 is arbitrary. Downstream classifiers must rely primarily on `angle_diff` and kinematic directional alignment rather than `cand_rank` or `dist_diff_best` alone to disambiguate the true direction of travel.

