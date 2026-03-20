# Overnight Implementation Plan — 2026-03-19

Goal: Break all three floors. Then synthesize the best architecture.

Status legend: [ ] = todo, [~] = in progress, [x] = done, [!] = tested, skip

---

## TIER 1 — Quick Wins (30–90 min each, high expected ROI)

### T1-A: Glitch-Aware Activity Correction
**Target**: SHA256 power: 48.9% → ???
**Idea**: rel_act=0.104 for SHA256 is 2× OOD. Correct it:
  `effective_activity = rel_act / (1 + 0.3 × (comb_per_ff - 1))`
**Test**: Does SHA256 effective_activity fall inside [0.035-0.051] training range?
**Implementation**: Replace rel_act with effective_activity in power features. Try coefficients 0.2, 0.3, 0.4.
**File**: `absolute_v18_glitch.py`
**Status**: [ ]

### T1-B: Wire Capacitance from DEF (SPEF-lite)
**Target**: AES power oracle floor: 20.4% → ???
**Idea**: AES oracle floor exists because wire cap varies per placement. Estimate it:
  For each net in DEF: C_wire_net ≈ HPWL_net × 0.2 fF/µm (sky130 M2)
  Sum → estimated_wire_cap per placement. Add as power feature.
**Implementation**: Parse net bounding boxes from DEF (already have parser). Compute per-net HPWL from pin xy coords. Sum → feature.
**File**: `absolute_v18_wirecap.py` (can combine with T1-A)
**Status**: [ ]

### T1-C: Power Delta Decomposition
**Target**: Power: 32% → ???
**Idea**: Don't predict total power. Decompose:
  P_total = P_cell + P_wire_est + P_clock_delta
  - P_cell: exact from Liberty internal power tables + SAIF toggle counts (no ML)
  - P_wire_est: HPWL × cap_per_um × V² × f (deterministic from DEF)
  - P_clock_delta: only this residual needs ML prediction
  The delta is much smaller and more design-invariant than total.
**Implementation**:
  1. Parse Liberty .lib for internal_power tables (or use driven_cap as proxy)
  2. Compute P_cell = Σ_cell(toggle × E_switch)  -- approximate with driven_cap_per_ff
  3. Compute P_wire_est from T1-B wire cap estimate × V_dd² × f
  4. Train model on residual: log(P_actual - P_cell - P_wire_est)
**File**: `absolute_v19_delta.py`
**Status**: [ ]

### T1-D: Systematic Skew Features from timing_paths.csv
**Target**: Skew: 0.237 → ???
**Idea**: Skew = systematic (tree asymmetry) + random (local noise).
  Systematic is predictable from FF spatial distribution:
  - Centroid offset of top-K critical FFs vs. all FFs
  - Gini coefficient of FF x/y distributions
  - Convex hull eccentricity
  - From timing_paths.csv: extract FF positions of worst-slack paths, compute spatial spread
**Implementation**:
  1. Load timing_paths.csv, extract launch/capture FF names
  2. Cross-reference with DEF to get FF xy positions
  3. Compute: critical_ff_centroid_offset, critical_ff_spread,
     asymmetry_x, asymmetry_y, frac_critical_on_boundary
  4. Add to skew feature set
**File**: `skew_v2_spatial.py`
**Status**: [ ]

---

## TIER 2 — Medium Effort (2–4 hours each, solid expected ROI)

### T2-A: Skip-Graph GNN for Skew (Top-K Critical Paths)
**Target**: Skew: 0.237 → sub-0.10
**Idea**: Build sparse graph of top-50 worst-slack launch-capture FF pairs.
  Run 2-layer GNN on this subgraph. Pool with max (not mean) to preserve worst-case.
  Key: topology of critical paths (chain vs star vs tree) is design-invariant.
**Implementation**:
  1. From timing_paths.csv: take top-50 paths by worst slack
  2. Build edge list: (launch_ff_node, capture_ff_node, path_slack, path_length)
  3. Node features: [x_norm, y_norm, log_area, drive_strength, is_critical]
  4. 2-layer GCN → max-pool → concat global features → skew head
  5. LODO evaluation
**File**: `skew_gnn_v1.py`
**Status**: [ ]

### T2-B: Design Embedding via Circuit Statistics
**Target**: All tasks — better cross-design generalization
**Idea**: Explicit design embedding from ~15 circuit statistics:
  - Logic depth histogram bins (depth 1-3, 4-6, 7-10, 10+)
  - Fanout distribution (p25, p50, p75, p90, max)
  - Cell type entropy (how diverse is the cell mix)
  - Rent's exponent proxy (pins vs. cells at multiple granularities)
  - FF clustering coefficient (how clustered are FFs spatially)
  PCA to 3-4 dims. Use as extra features in all models.
**File**: `design_embedding.py`
**Status**: [ ]

### T2-C: Multi-Task Joint Prediction (Shared Trunk)
**Target**: All three tasks simultaneously
**Idea**: Single neural net with shared 3-layer MLP trunk + 3 task heads.
  - Power and WL share most of trunk (ρ=0.915)
  - Skew head gets additional critical-path features
  - Per-placement normalization for targets
  - GradNorm balancing across 3 tasks
**File**: `multitask_v1.py`
**Status**: [ ]

### T2-D: Conformal Prediction for Uncertainty
**Target**: Not improving MAPE — but quantifying when to trust predictions
**Idea**: Split conformal prediction on LODO residuals → per-prediction intervals.
  Use nonconformity score = |y_pred - y_true| / (|y_true| + eps)
  Calibrate on held-out 20% of training designs.
  Output: power=0.05W ± 0.01W (90% coverage interval)
**File**: `conformal_pred.py`
**Status**: [ ]

### T2-F: k_WA from Placement Quality Metrics (RUDY congestion)
**Target**: WL — reduce within-design variance, improve per-placement WL prediction
**Idea**: k_WA varies 1.4× across designs AND varies within designs across placements.
  Placement quality metrics that predict k_WA:
  - RUDY (Rectangular Uniform Wire Density): divide die into grid, count nets crossing each cell
  - Pin density variance (hot spots = detour)
  - Net bounding box overlap ratio (congestion proxy)
  - Average net degree weighted by bounding box area
  These are computable from DEF nets section.
**File**: `absolute_v20_rudy.py`
**Status**: [ ]

### T2-E: RSMT Wirelength Estimation Per Net
**Target**: WL: 11% → sub-5%
**Idea**: Instead of Donath aggregate model, compute per-net Steiner estimate.
  For each net: RSMT_est ≈ HPWL × (1 + 0.1 × log(pin_count))  (Rent correction)
  Sum over all nets → per-placement WL estimate. Use as baseline + predict delta.
**Implementation**:
  1. Parse all nets from DEF (have parser)
  2. For each net: compute bounding box of all pins, HPWL = (x_max-x_min) + (y_max-y_min)
  3. Apply Rent correction for multi-pin nets
  4. Sum → rsmt_total. Use log(rsmt_total) as WL normalization baseline.
  5. Predict log(actual_wl / rsmt_total) — should be ~constant (~1.1-1.5)
**File**: `absolute_v20_rsmt.py`
**Status**: [ ]

---

## TIER 3 — High Effort / Experimental

### T3-A: Fine-tuning Last Layer (Transfer Learning)
**Target**: All tasks — K-shot for neural net
**Idea**: Train MLP on 3 designs, freeze all layers except last, fine-tune on K=1-3
  labeled points from test design. Neural net version of K-shot calibration.
**Status**: [ ]

### T3-B: Bayesian Optimization with Predictor as Surrogate
**Target**: Application paper contribution
**Idea**: Use predictor to guide CTS knob search via BO.
  Practically useful even with 10-15% MAPE.
**Status**: [ ]

### T3-C: Synthetic Design Augmentation
**Target**: More training designs → better generalization
**Idea**: Perturb existing designs (relocate 10-20% FFs, vary clock freq),
  run CTS, treat as new designs. Gets to 40-100 "designs".
**Note**: Requires running EDA tools — probably out of scope for tonight.
**Status**: [ ]

---

## EXECUTION ORDER (Tonight)

1. [T1-A] Glitch correction → test SHA256 effective_activity range → run LODO  (~45 min)
2. [T1-B] Wire cap from DEF → add to power features → run LODO  (~60 min)
3. [T1-C] Power delta decomposition → evaluate  (~90 min)
4. [x] T1-D Skew spatial features → 0.0769 MAE ← DONE
5. [T2-E] RSMT WL estimate → evaluate  (~90 min)
6. [T2-A] Skip-graph GNN for skew → LODO  (~2-3 hours)
7. [T2-B] Design embedding → plug into all models  (~60 min)
8. [T2-C] Multi-task joint model → full LODO  (~2-3 hours)
9. [T2-D] Conformal prediction  (~60 min)
10. SYNTHESIZE: combine best components → final architecture

---

## Results Tracker

| Approach | Power MAPE | WL MAPE | Skew MAE | Notes |
|----------|-----------|---------|----------|-------|
| baseline_best (v16_final) | 32.0% | 11.0% | 0.237 | zero-shot |
| kshot_best (v17, K=20) | 9.8% | 6.6% | — | 20 labeled runs |
| T1-A glitch correction | — | — | — | pending |
| T1-B wire cap feature | — | — | — | pending |
| T1-C power delta | — | — | — | pending |
| T1-D skew spatial | — | — | **0.0769** ✓ | XGB LODO, all 4 designs <0.10 |
| T2-E RSMT WL | — | — | — | pending |
| T2-A skip-graph GNN | — | — | — | pending |
| T2-B design embedding | — | — | — | pending |
| T2-C multi-task | — | — | — | pending |
| FINAL SYNTHESIS | — | — | — | pending |

---

## Synthesis Criteria (End of Night)

After all experiments, build final model using:
- Best power feature set (T1-A, T1-B, T1-C, T2-B findings)
- Best WL feature set (T2-E findings)
- Best skew approach (T1-D or T2-A, whichever works)
- Best normalization (keep v11 proven pw_norm)
- Best model architecture per task (XGB/LGB/GNN/MLP — pick winner)
- K-shot calibration layer on top (proven: K=20 → 9.8%)

Save final architecture in `final_synthesis/` folder.
