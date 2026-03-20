# CLAUDE.md — CTS Outcome Prediction: Generalization is the Only Truth

## Prime Directive

**Your only goal: achieve MAE < 0.10 on ALL THREE tasks (skew, power, wirelength) on UNSEEN designs.**

Leave-One-Design-Out (LODO) is the ONLY evaluation that matters. A model that fits training designs perfectly but fails on unseen ones is useless. Never report results unless they are on a held-out design that was never seen during training.

**Run quick feasibility tests first. ALWAYS. Never run 300-epoch training until you have verified that the approach is sensible on a 20-epoch smoke test.**

**Record everything.** Every approach tried, why it was tried, what happened, why it worked or failed. Write findings to `RESEARCH_LOG.md` continuously.

---

## Project Overview

CTS-Task-Aware-Clustering predicts Clock Tree Synthesis (CTS) outcomes — **skew**, **power**, and **wirelength** — for unseen VLSI circuit designs using physics-informed ML.

The core challenge: **zero-shot generalization**. Models trained on known designs must predict accurately on entirely new, unseen circuit families.

Dataset lives in `dataset_with_def/`.

---

## Data Paths

```
dataset_with_def/
  unified_manifest_normalized.csv     # Main CSV: all placements + CTS runs + targets
  placement_files/                    # Per-run folders:
    {design}_{run_id}/
      {design}.def                    # Physical placement (DEF format)
      {design}.saif                   # Switching activity (SAIF format)
      timing_paths.csv                # Timing path data

processed_graphs/                     # Pre-processed .pt graph files
  {placement_id}.pt                   # Contains: X[N,18], A_skip_csr, A_wire_csr,
                                      # p_indices, num_nodes, raw_areas
```

### CSV Columns

Key columns in `unified_manifest_normalized.csv`:
- `placement_id`, `design_name` — identifiers
- `cts_max_wire`, `cts_buf_dist`, `cts_cluster_size`, `cts_cluster_dia` — CTS knobs
- `z_cts_*` — z-scored versions of knobs
- `skew_setup` (ns), `power_total` (W), `wirelength` (µm) — RAW targets
- `z_skew_setup`, `z_power_total`, `z_wirelength` — globally z-scored (DO NOT USE for training — see normalization section)

### PT File Contents

```python
data = torch.load('processed_graphs/aes_run_xxx.pt')
data['X']           # [N, 18] node features (normalize_features output)
data['A_skip_csr']  # [N,N] sparse: timing path (skip) edges
data['A_wire_csr']  # [N,N] sparse: physical wire edges (1-hop)
data['p_indices']   # [2,E] 2-hop wire mask (build_X_hop_mask output)
data['num_nodes']   # int
```

### X Column Mapping

```
0:x_norm  1:y_norm  2-5:dist_boundaries  6:log1p(area)
7:avg_pin_cap*1000  8:total_pin_cap*1000  9:log2(drive)
10:is_sequential  11:is_buffer  12:toggle  13:sum_toggle
14:signal_prob  15:non_zero  16:log1p(fan_in)  17:log1p(fan_out)
```

---

## Designs in Dataset

- `aes`: 31 placements, ~2994 FFs
- `ethmac`: 47 placements, ~5000+ FFs
- `picorv32`: 31 placements, ~1597 FFs
- `sha256`: 31 placements, ~1807 FFs

**10 CTS runs per placement** (different knob configs), 140 placements total = 1400 data points.

---

## Target Normalization — CRITICAL

The globally z-scored targets (`z_skew_setup` etc.) in the CSV are WRONG for ML because they conflate design-type effects with CTS-parameter effects. An AES placement at 0.72ns gets z=-0.06 but an ethmac placement at 0.72ns gets a completely different z-score.

**Always use per-placement normalization:**

```python
# For each placement's 10 CTS runs:
vals = np.array([row['skew_setup'] for row in placement_runs])
mu, sig = vals.mean(), vals.std()
sig = max(sig, max(abs(mu)*0.01, 1e-4))  # floor to prevent explosion
z_target = (val - mu) / sig
```

This makes the task: "given this placement and these CTS knobs, how does skew compare to the average for THIS placement?" — which is directly learnable from knob-geometry interactions.

---

## Physics of Each Target

### SKEW

**Definition**: `max(clock_arrival_time) - min(clock_arrival_time)` across all FFs.

**Physical determinants** (from CTS literature):
1. **Wire length imbalance**: FFs far from the clock source get clock later → higher latency → imbalance = skew. Primary cause.
2. **Cluster diameter vs FF spacing**: CTS clusters FFs within `cluster_dia`. If `cluster_dia << kNN_spacing`, FFs can't be grouped → hard to balance → high skew.
3. **Buffer distance vs path length**: `buf_dist / max_skip_dist` = buffer stages per path. More stages = more equalization points = lower residual skew.
4. **Launch-capture spatial asymmetry**: If launch FFs are all on one side and capture FFs on the other, the tool must route long unbalanced paths.
5. **Drive strength variance**: Asymmetric RC delay across paths when drive strength varies.

**Key insight**: Skew is a MAX/WORST-CASE metric, not average. Any feature that averages over FFs (mean pooling, graph compression) destroys the signal. Must preserve tail statistics.

**What CTS-Bench proved** (your own paper):
- Raw GCN achieves skew MAE ~0.16, clustered graphs degrade immediately
- Generic graph coarsening "frequently results in negative R² scores under zero-shot evaluation"

### POWER

**Definition**: `P = α × C_total × V² × f`

**Physical determinants**:
1. **Total clock WL × wire capacitance**: C_wire ∝ WL. Power ≈ k1 × WL.
2. **Buffer count × buffer capacitance**: n_buffers ∝ n_ff / cluster_size. Power ≈ k2 × n_ff/cluster_size.
3. **Toggle activity**: dynamic power scales with switching frequency.
4. **cluster_size knob**: smaller cluster_size → more buffers → higher power.

**Key insight**: Power correlates tightly with WL (longer tree = more capacitance = more power). Shared representation works. Use `log(power / activity_sum)` to normalize for activity differences across designs.

### WIRELENGTH

**Definition**: Total length of all clock routing wires (µm).

**Physical determinants** (from Cong/Kahng/Robins 1993, DME algorithm):
1. **HPWL of FF bounding box**: Steiner minimum tree ≤ HPWL ≤ 1.5×SMT. Clock WL ≈ 1.1-1.5 × HPWL.
2. **cluster_dia × HPWL**: Larger clusters merge more FFs → shorter local routing.
3. **max_wire / HPWL**: Wire length budget relative to tree extent.
4. **FF spatial density**: Dense FF placement = shorter routing.
5. **cluster_size**: Larger cluster_size → fewer clusters → shorter inter-cluster routing.

**Key insight**: WL is the most predictable of the three tasks (global aggregate). HPWL alone gives ~80% of the signal.

---

## What Has Been Tried and Why It Failed

### Attempt 1: Task-Aware Graph Compression (AssignmentNet + trace moments)
**Approach**: Learn soft FF→cluster assignments C, compress adjacency A_c = C^T A C, use trace moments as graph features.
**Why it failed**:
- k=64 supernodes average ~47 FFs each — destroys skew signal
- CTS-Bench independently proved this: clustering causes negative R² for skew
- GradNorm oscillation: easy tasks (WL) dominated gradient, skew stagnated at 0.70-0.85 MAE
- Global target normalization conflated design-type effects

### Attempt 2: Global Z-score Targets
**Why it failed**: `z_skew` for AES at 0.72ns ≠ `z_skew` for ethmac at 0.72ns (different global means). Model must predict absolute baseline which requires knowing design type → fails on unseen designs.

### Attempt 3: 50-dim Global Feature MLP (no per-FF features)
**Why it failed**: Aggregate statistics (mean, std, percentiles) of FF positions cannot capture worst-case path signal. Skew MAE stayed at 0.80-0.93 while WL improved.

### Attempt 4: Order Statistics + Physics Interactions (current)
**Status**: Promising architecture but needs more evaluation. Uses per-FF SGC features with top-k ordering by timing path involvement.

---

## Approaches to Try (Priority Order)

### Priority 1: LightGBM/XGBoost on Physics Features (QUICK TEST FIRST)

**Rationale**: GAN-CTS (TCAD 2022) achieves 3% MAPE using ResNet50 features + simple regression. LightGBM with good features often beats neural networks on small datasets (140 placements). Trees handle nonlinearity, don't need gradient flow through graph structure.

**Features** (from cts_features.py, ~50 dims):
- HPWL, aspect ratio, centroid offset (Steiner proxy)
- Skip path distance percentiles (p50, p90, max, mean, std)
- Launch-capture spatial asymmetry
- kNN distances (k=4,8,16) — mean, std, max, p90
- Capacitive load features
- FF density at 3 radii
- Grid entropy/Gini
- Scale features (log n_ff, etc.)
- CTS knobs (all 4)
- **Interaction features** (knob × geometry products)

**Physics interactions for each task**:
```
Skew:  buf_dist/skip_max, cluster_dia/knn4, cluster_dia*centroid_dist
Power: n_ff/cluster_size, cluster_dia*HPWL, max_wire*HPWL
WL:    cluster_dia*HPWL, cluster_sz/HPWL, max_wire/HPWL
```

**Quick test**: 5-fold LODO on 4 designs = 4 models. Runs in minutes.

**If this works**: LightGBM generalizes much better than NNs on small datasets because it doesn't overfit feature interactions.

### Priority 2: Physics-Normalized Targets + Tree Models

**Per target**:
```python
# Skew: raw ns, per-placement z-score
# Power: log(power / sum_toggle) → design-invariant switching power
# WL: log(wl / n_ff) → per-FF wirelength (size-invariant)
```

### Priority 3: Neural Net with Per-FF Order Statistics

**Architecture** (no graph compression):
- Per-FF features → sorted by timing path involvement → top-k pool → skew head
- Global aggregate features → encoder → power/WL heads
- Interaction terms via LayerNorm

### Priority 4: Transfer Learning via ResNet50 on Placement Images

**As in GAN-CTS**: Render FF positions as 2D image → ResNet50 → 512-dim embedding → MLP with knobs.
**Quick test**: Use simple 8×8 or 16×16 grid image, small CNN (not full ResNet).

### Priority 5: Meta-features + Ridge Regression (Baseline)

**Dead simple**: HPWL, n_ff, knob values, 5 interaction terms → Ridge. Sets a baseline that must be beaten.

---

## Common Commands

```bash
# Quick feasibility test (always run first, 20 epochs max)
python quick_test.py

# Main predictor (LightGBM)
python cts_predictor.py --cv

# Advanced stacked ensemble
python advanced_predictor.py --cv

# Cross-design predictor (per-design z-scores)
python cross_design_predictor.py --cv

# Neural net (only run after quick test confirms feasibility)
python cts_model.py

# Run tests
python -m pytest test_cts_predictor.py -v
python -m pytest test_cts_predictor.py::TestZeroShotGeneralization -v
```

---

## MAE Targets

```
Skew:       < 0.10 (per-placement normalized units)
Power:      < 0.10 (per-placement normalized units)  
Wirelength: < 0.10 (per-placement normalized units)
```

All measured on LODO held-out design. If per-placement normalized, these correspond to predicting within ±10% of the within-placement variation range.

---

## Research Protocol

### Before Running Anything

1. **Read the research log** (`RESEARCH_LOG.md`) to see what's been tried
2. **Formulate hypothesis**: "I believe X will work because Y physics reason"
3. **Quick feasibility test**: 20 epochs max, 1 fold of LODO
4. **Evaluate**: Does the approach show convergence? Is train loss dropping? Is val loss following?
5. **Only then**: Run full 300-epoch LODO

### Recording Results

After EVERY experiment, append to `RESEARCH_LOG.md`:
```
## [DATE] Approach: [name]
**Hypothesis**: ...
**Implementation**: ...
**Result**: skew=X, power=Y, wl=Z (LODO, held_design=ethmac)
**Why it worked/failed**: ...
**Next step**: ...
```

### Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Train loss drops but val > 0.8 | Overfitting to seen designs | More regularization, simpler model, better features |
| One task improves, others worsen | GradNorm wrong direction | Check task weight update formula |
| WL/power oscillate, skew stuck | Mean pooling destroys skew signal | Order statistics, preserve tail |
| MAE > 1.0 | Normalization bug | Check per-placement z-score floor |
| Loss explodes at ep0 | Feature scale issue | Check global_feats mu/sig values |

---

## Architecture Decision Tree

```
Is the dataset small (< 500 samples per design)?
  YES → Try LightGBM first. Gradient boosted trees generalize better than NNs.
  NO  → Neural net with proper inductive bias

Does the target require worst-case reasoning?
  YES (skew) → Order statistics pooling, NOT mean pooling
  NO (power, WL) → Mean pooling OK

Is the target a global aggregate?
  YES (WL, power) → Global spatial features sufficient
  NO (skew) → Per-FF features with tail preservation needed

Are you evaluating generalization?
  ALWAYS use LODO, NEVER use random split or leave-one-placement-out
```

---

## Key Literature Findings

1. **GAN-CTS (Lu et al. TCAD 2022)**: 3% MAPE using ResNet50 + MLP. Key: shared layers for power/WL (correlated), separate for skew. Features from placement images.

2. **CTS-Bench (Khadka et al. 2026)**: Graph clustering destroys skew signal. Raw GCN: skew MAE 0.16, clustered: negative R². Power/WL survive clustering (global aggregates).

3. **Kahng UCSD (c300)**: Block aspect ratio, nonuniform sink placement, clock entry point are the three critical dimensions for CTS prediction. Floorplan context essential.

4. **iCTS (Li et al. TCAD 2025)**: Within-cluster skew ∝ cluster_dia × wire_cap_per_unit. Clustering quality determines skew more than individual buffer placement.

5. **DME algorithm / Cong et al.**: Clock WL is bounded by Steiner tree ≈ 1.1-1.5 × HPWL. WL is the most analytically predictable metric.

6. **Elmore delay model**: Delay = R × C_downstream. Skew = max(delay) - min(delay). To predict skew, must model path length variance, not just mean.

---

## Overnight Run Checklist

Before starting the overnight run, verify:
- [ ] RESEARCH_LOG.md exists and is current
- [ ] Quick test (20 epochs) passes without error
- [ ] Feature normalization gives mu ∈ [-3,3], sig ∈ [0.1, 3]
- [ ] Per-placement target normalization is applied (NOT global z-scores)
- [ ] LODO is the evaluation (NOT leave-one-placement-out)
- [ ] Results are written to file (not just stdout)
- [ ] Best model checkpoint is saved per design fold
- [ ] GPU memory < 6GB (leave headroom)

---

## Helper: Quick Sanity Check Script

```python
# Run this before any full training to check data looks right
import pandas as pd
import numpy as np

df = pd.read_csv('dataset_with_def/unified_manifest_normalized.csv')
print(f"Total rows: {len(df)}")
print(f"Designs: {df['design_name'].value_counts().to_dict()}")
print(f"\nRaw skew range: [{df['skew_setup'].min():.4f}, {df['skew_setup'].max():.4f}]")
print(f"Raw power range: [{df['power_total'].min():.6f}, {df['power_total'].max():.6f}]")
print(f"Raw WL range: [{df['wirelength'].min():.0f}, {df['wirelength'].max():.0f}]")

# Check per-placement variation
for design in df['design_name'].unique():
    d = df[df['design_name']==design]
    for pid, grp in d.groupby('placement_id'):
        skew_range = grp['skew_setup'].max() - grp['skew_setup'].min()
        print(f"  {pid}: skew_range={skew_range:.4f}ns  "
              f"wl_range={grp['wirelength'].max()-grp['wirelength'].min():.0f}µm")
        break  # just first one per design
```

