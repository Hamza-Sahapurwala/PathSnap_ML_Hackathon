# HMM / Viterbi route reconstruction: experiments and results (Member C)

All numbers below come from the files named next to them; nothing is estimated or copied by hand from elsewhere.
Re-run with:

```bash
python -m src.hmm_tuning --trials 60 --n-jobs 12                          # train trips only (~25 min)
python -m src.hmm_experiments --partition student_test --config artifacts/hmm_config.json
python -m src.make_map --trip 00000077 00000065 00000052 --png
```

## 1. Setup

| item | value |
|---|---|
| classifier | RandomForest selected by Member B (`results/selection.md`), untuned. B's run saved no model file, so Member C rebuilt it with B's own `pipeline("RandomForest")`, seed 42, A's training parquet. Its student-test Top-1, continuity, macro F1 and log loss equal B's `results/model_comparison.csv` to 10 digits. Saved as `artifacts/final_model.joblib`. |
| candidates / features | Member A's export (`results/features_*.parquet`), 10 candidates per GPS point, 14 features |
| point-level reference | A's aligned `reference_arc_id` per GPS point (same reference as B's Top-1) |
| sequence-level reference | the trip's `.route` file (read only by the evaluation code, after prediction) |
| decoder | `src/hmm.py`; tuned config `artifacts/hmm_config.json` |
| comparison | classifier-only (highest probability per point) vs classifier + HMM on **the same candidates and probabilities** |

## 2. Model

* **Hidden states:** the candidate arcs of each GPS observation, keeping the original `arc_id`.
* **Emission:** `emission_power * log p` where `p` is the classifier's positive-class probability (clipped at 1e-6),
  plus an optional Gaussian term on the perpendicular GPS distance (`distance_weight`, `gps_sigma_m`).
* **Transition:** Newson-Krumm style. `d_net` = directed network distance between the two projected positions
  (same arc, direct, or bounded Dijkstra through intermediate arcs); `log T = -|d_net - d_gps| / beta - log(2 beta)` with
  `beta = beta_m + beta_gap_m_per_s * (dt - 1)`. Unreachable or faster-than-60 m/s pairs get a bounded floor
  (`log_disconnect_penalty`) instead of `-inf`, so one bad observation cannot break the decode. Optional penalties for a
  U-turn onto the reverse arc and for moving backwards along the same arc.
* **Decoding:** log-space Viterbi with back-pointers; variable candidate counts; points without candidates are skipped
  and bridged; a time gap above 60 s starts a new independent segment.
* **Route output:** per-point arcs, consecutive repeats collapsed, then `fill_route` inserts the shortest directed
  connecting arcs between consecutive route arcs (the reference `.route` lists every driven arc; GPS points are sparser).
* **Efficiency:** no all-pairs matrix. Bounded, cached Dijkstra over a CSR adjacency (radius `2 * d_gps + 100 m`,
  capped at 2 km, 60 hops, 4,000 settled nodes). Decode time 0.07 s to 1.2 s per student-test trip; graph search is a small
  share of it (cache hit rate 79-97 %), and no search hit its caps.

## 3. Decoder tuning (training trips only)

* **Probabilities:** out-of-fold. GroupKFold(5) by trip; each fold is scored by a clone of the final pipeline fitted on the
  other four folds. Using in-sample probabilities from the final model would make the decoder over-trust the classifier.
* **Search:** Optuna TPE, seed 42, 60 trials, trial 0 = untuned baseline. Objective = pooled observation-level Top-1
  accuracy over 78 training trips. Every trial: `results/hmm_tuning_trials.csv`.
* **Excluded from the objective:** trips 00000033 and 00000054. Their GPS lies far from the reference route (most of the
  training candidate misses); every method scores about 0.10-0.12 Top-1 on them (table below), so they would only add noise.
* **Student-test was not used** for any tuning decision; it was scored once with the chosen config.

Result (`results/hmm_ablations.csv`, 78 objective trips; fold = mean +- std over the 5 trip folds):

| variant | Top-1 | fold Top-1 | filled route F1 | edit similarity | Top-1 on 033 + 054 |
|---|---|---|---|---|---|
| classifier only | 0.8356 | 0.834 +- 0.052 | 0.357 | 0.217 | 0.102 |
| HMM untuned | 0.9117 | 0.910 +- 0.041 | 0.845 | 0.737 | 0.113 |
| **HMM tuned (trial 51)** | **0.9295** | 0.928 +- 0.029 | **0.870** | **0.777** | 0.110 |

Tuned values: `emission_power` 0.50, `distance_weight` 0.84 with `gps_sigma_m` 12.2 m, `beta_m` 3.0 m,
`beta_gap_m_per_s` 9.6, `log_disconnect_penalty` -38.0, `uturn_log_penalty` -4.3, `backward_log_penalty` -0.24,
no transition normalisation.

**Ablation** (tuned config with one setting switched back to its untuned value; Top-1 change vs tuned 0.9295):

| setting reverted | Top-1 | change | reading |
|---|---|---|---|
| transition normalisation switched on | 0.9035 | -0.026 | harmful: normalising each row throws away how *unlikely* every transition was |
| no U-turn penalty | 0.9243 | -0.005 | helps: noisy points otherwise flip to the reverse arc |
| disconnect penalty -15 instead of -38 | 0.9243 | -0.005 | helps: disconnected jumps must be rarer |
| no distance emission | 0.9267 | -0.003 | helps a little: geometry adds information the classifier already partly has |
| emission power 1 instead of 0.5 | 0.9276 | -0.002 | flattening the RF probabilities helps a little (RF scores are over-confident) |
| no backward penalty | 0.9289 | -0.001 | negligible |
| `beta_m` 10 instead of 3 | 0.9300 | +0.0005 | flat: no real effect |
| `beta_gap_m_per_s` 2 instead of 9.6 | 0.9305 | +0.001 | flat: no real effect |

The two beta settings are not well determined: their untuned values score marginally *higher* than the tuned
ones, by far less than the fold spread (+-0.03). We kept the Optuna winner as the pre-declared protocol choice rather
than re-selecting from the ablation.

## 4. Student-test results (held out; `results/hmm_comparison.csv`)

Pooled over the 10 trips (19,160 GPS points; reference route 3,477 arcs):

| metric | classifier only | HMM untuned | **HMM tuned** |
|---|---|---|---|
| observation-level Top-1 accuracy | 0.8598 | 0.9284 | **0.9390** |
| Top-1, mean +- std over trips | 0.874 +- 0.071 | 0.936 +- 0.048 | 0.941 +- 0.051 |
| candidate miss rate (same for all) | 0.0030 | 0.0030 | 0.0030 |
| continuity: same arc or directly connected | 0.929 | 0.981 | 0.980 |
| continuity: connected incl. multi-hop paths | 0.956 | 0.999 | 0.999 |
| adjacent pairs with no plausible connection | 846 | 23 | 12 |
| route precision (filled) | 0.233 | 0.793 | 0.859 |
| route recall (filled) | 0.989 | 0.991 | 0.991 |
| **route F1 (filled)** | 0.377 | 0.881 | **0.920** |
| edit similarity to `.route` | 0.232 | 0.791 | **0.857** |
| decode time, mean per trip | 0.01 s | 0.49 s | 0.52 s |

Classifier-only Top-1 0.8598 equals Member B's reported student-test Top-1 for the RandomForest, which confirms that both
evaluations use the same points and reference.

Per trip (Top-1; points fixed / broken by the tuned HMM relative to classifier-only):

| trip | points | classifier only | HMM untuned | HMM tuned | fixed | broken |
|---|---|---|---|---|---|---|
| 00000000 | 2,503 | 0.939 | 0.962 | 0.970 | 84 | 4 |
| 00000007 | 1,354 | 0.929 | 0.993 | 0.995 | 89 | 0 |
| 00000009 | 1,049 | 0.861 | 0.914 | 0.909 | 66 | 16 |
| 00000040 | 2,995 | 0.934 | 0.964 | 0.978 | 158 | 26 |
| 00000052 | 262 | 0.958 | 1.000 | 1.000 | 11 | 0 |
| 00000056 | 3,958 | 0.883 | 0.946 | 0.952 | 321 | 48 |
| 00000065 | 768 | 0.811 | 0.858 | 0.831 | 50 | 35 |
| 00000076 | 1,486 | 0.867 | 0.933 | 0.949 | 138 | 16 |
| 00000077 | 3,911 | 0.708 | 0.848 | 0.876 | 820 | 161 |
| 00000091 | 874 | 0.847 | 0.943 | 0.947 | 97 | 9 |

## 5. Where the HMM helps, and where it does not

**Helps (every trip):**
* Classifier-only flips between an arc and its reverse twin (in Member C's data profiling, 73-76 % of GPS points have
  both directions of a road among their candidates). Its route has 5,593 arcs for a 3,477-arc reference, and filling the
  gaps between the flips adds detours (14,780 arcs, precision 0.23). The tuned HMM's route (3,274 arcs, 4,013 after
  filling) is close to the reference, and pairs with no plausible connection drop from 846 to 12.
* The largest gain is the hardest trip, **00000077** (0.708 -> 0.876, 820 points fixed): a long urban trip with many
  parallel streets, where per-point decisions are ambiguous but the sequence is not (`maps/hmm_student_test_00000077.html`).
* On **00000052** the HMM is perfect (1.000) and its route matches the reference almost exactly
  (`maps/hmm_student_test_00000052.html`).

**Does not help / hurts:**
* The HMM still breaks some points the classifier had right (321 fixed vs 48 broken on 00000056; 161 broken on 00000077).
  These are mostly where the HMM commits to one of two near-parallel candidates for a run of points and picks the wrong one.
* **Tuning hurt two trips** relative to the untuned HMM: 00000065 (0.858 -> 0.831) and 00000009 (0.914 -> 0.909). On
  00000065 the tuned decoder lost 36 points and gained 15. The losses cluster just before the 210 s stop (points 100-110)
  and around a 21 s gap (points 530-536). 11 of the 36 lost points are on the *reverse* of the reference arc: the U-turn
  penalty that helps on average resists a real turnaround (`maps/hmm_student_test_00000065.html`). Route F1 on the
  same trip still improved (0.812 -> 0.921).
* **It cannot repair what is not in the candidate set.** 0.3 % of student-test points have no correct candidate. On the two
  noisy training trips (00000033, 00000054) every method stays near 0.11 Top-1, because the GPS is far from the reference.
* Point accuracy is measured against A's *alignment* of the reference route to GPS points, which is itself an estimate.
  The reference route is "reference information", not verified ground truth (Dataset Exploration guide, section 6).

## 6. Limitations

* The final classifier is a rebuild of B's untuned RandomForest (identical metrics), not a file B produced, and it was
  not Optuna-tuned (team decision, time).
* The decoder objective is point accuracy only; route metrics were recorded but not optimised.
* `fill_route` uses the shortest path between predicted arcs; when the vehicle took a longer way between two sparse GPS
  points, the filled route is wrong there (route recall is already 0.991, precision 0.859 is the weaker side).
* Gap handling is a fixed 60 s reset; trip 00000065 shows that behaviour around long stops could be improved.
* Timing: the tuned decode in the comparison re-used the graph cache warmed by the untuned decode of the same trip,
  so its `seconds_graph_search` is about zero; the untuned run (cold cache) gives the honest graph-search cost.

## 7. Maps (`maps/`)

`hmm_student_test_00000077`, `_00000065`, `_00000052` (`.html` interactive, `.png` screenshot). Layers: faded
OpenStreetMap basemap, road network within 250 m of the track (grey), raw GPS (blue, hover = index, timestamp and every
method's arc with ✓/✗), reference route (green), classifier-only (orange), untuned HMM (purple, dashed), tuned HMM (red),
and mismatched observations per method (rings). The layer control (top right) toggles each layer. Generated by
`src/make_map.py`.

## 8. Viva preparation

* **Hidden states?** The candidate arcs of each GPS point (Member A's top-10), with their original arc ids.
* **Emission?** The RandomForest's probability that the candidate is the right arc, in log space, raised to 0.5
  (tuned), plus a Gaussian term on GPS-to-arc distance.
* **Transition?** How well the driving distance between two candidates matches the straight-line GPS distance,
  using a Laplace distribution and a directed shortest path. One-way streets are respected.
* **Disconnected candidates?** Bounded penalty (-38 in log space), not `-inf`, so a single bad point cannot make the
  whole trip impossible. The ablation shows a softer penalty (-15) costs 0.5 points of accuracy.
* **Missing candidates / gaps?** Points without candidates are skipped and the decode bridges over them. Gaps over
  60 s start a new segment, because the vehicle could have gone anywhere.
* **Why out-of-fold probabilities for tuning?** The final RF has seen every training trip. Its in-sample scores are
  too confident, so a decoder tuned on them would learn to ignore the road network.
* **Did the HMM help?** Yes: Top-1 0.860 -> 0.939, route F1 0.377 -> 0.920 on the held-out student-test trips, better on all
  10 trips. It does not help when the right arc is not among the candidates, and tuning slightly hurt 2 of 10 trips.
* **Complexity?** Viterbi is O(T K^2) with K <= 10, so about 100 transition scores per step. The cost is the
  shortest-path search, which is bounded by radius, hops and nodes, and cached per source node.
