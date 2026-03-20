# NOVEL APPROACHES — CTS Outcome Prediction
## Path to Ultra-Low MAE (Target: < 0.005)

**Current best (LODO, per-placement normalized rank MAE)**:
- Skew: 0.237 ✗ (oracle ceiling: 0.107 with skew_hold)
- Power: 0.0656 ✓ (pre-CTS), 0.0163 (post-CTS)
- WL: 0.0849 ✓ (pre-CTS), 0.0803 (post-CTS)

**GNN Phase 2 (direct.ipynb, global z-scores — WRONG norm)**:
- Skew: 0.1449, Power: 0.0843, WL: 0.0699

**Absolute prediction (zero-shot, MAPE)**:
- Power: 32.0% LODO (v11), 7.5% zipdiv
- WL: 13.1% LODO (v11), 5.5% zipdiv

---

## Critical Shortcomings

### 1. Wrong Normalization in GNN
**Problem**: `direct.ipynb` loads `z_skew_setup`, `z_power_total`, `z_wirelength` from CSV —
these are GLOBAL z-scores that conflate design-type with CTS-knob effects.
**Fix**: Per-placement normalization: `z = (val - mu_placement) / sigma_placement`
**Expected impact**: All three GNN heads should see dramatic improvement.

### 2. Graph Compression Destroys Skew Signal
**Problem**: Phase 1 compresses N FFs → K=64-798 supernodes. Skew = max delay − min delay
(worst-case, not average). Compression averages the tails away.
**Fix**: Dedicated uncompressed skew head that operates on raw FF features, using
top-K timing-critical nodes (highest timing path involvement degree).
**Physics**: Only ~5-15% of FFs are on critical paths. Keep those intact.

### 3. Knob Feature Not Injected at FF Level
**Problem**: CTS knobs (cluster_dia, buf_dist, etc.) are only in the MLP heads.
They don't modulate the message passing, so the GNN can't learn "cluster_dia = 40µm
means FFs within 40µm get grouped → short intra-cluster routing."
**Fix**: Add knob-conditioned FiLM layers (Feature-wise Linear Modulation) in message passing.

### 4. Phase 1 Pre-training Objective Doesn't Preserve Task Signal
**Problem**: Reconstruction + locality + entropy loss doesn't encode skew signal.
Phase 1 sanity check: r(timing_path_dist, skew) = 0.0 after compression.
**Fix**: Add triplet/contrastive loss: if run_i has higher skew than run_j for same placement,
the compressed representation should preserve this ordering.

### 5. Per-Placement Polynomial Coefficients are Unpredictable Across Designs
**Problem**: The quadratic response y = a*cd² + b*cd + c has coefficients with CV > 100%.
Ridge regression on placement features to predict coefficients gives LODO R² = −304.
**Implication**: No single global model can predict within-placement response across designs.
**Solution class**: Per-placement adaptation methods (meta-learning, GP, few-shot).

---

## Novel Approach 1: Gaussian Process with Physics-Derived ARD Kernel ★★★

**Concept**: For each placement, the 10 CTS runs form a small regression dataset.
A GP with ARD RBF kernel over the 4 CTS knobs is fit to these 10 points.
The kernel length scales encode WHICH knobs matter most for THIS placement.

**Why it's novel**: Instead of using one global model, each placement has its own
GP regressor. For LODO, we use a hierarchical GP where:
- Prior = mean of kernel hyperparameters across training placements
- Posterior = fine-tuned on test placement's 10 points (if available)
- Zero-shot = prior only (no test labels)

**Implementation** (relative prediction):
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# For each held-out design:
# 1. Collect all training placements' (knob configs, outcomes)
# 2. Fit GP with ARD RBF kernel (4 length scales = 4 knobs)
# 3. GP automatically learns which knobs matter most
# 4. For test placement: predict z-score of each CTS run

kernel = (C(1.0) * RBF(length_scale=[1,1,1,1], length_scale_bounds=(0.01,100))
          + WhiteKernel(0.1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

# Training data: stack all training placements' (z_knobs, z_outcome)
# The GP learns the cross-placement response surface
```

**Key physics insight**: cluster_dia dominates power (ρ=−0.952). GP's ARD should
automatically assign short length scale to cluster_dia → sharp sensitivity → captures
the quasi-monotone relationship better than tree models.

**Expected improvement**: Power/WL: could reach 0.03-0.04 (smooth relationship + proper kernel).
Skew: limited by fundamental noise floor (~0.20).

**Status**: Implemented for absolute prediction (v13, GP for WL). NOT yet applied to
relative (per-placement z-score) prediction.

---

## Novel Approach 2: MAML Meta-Learning for Skew ★★★

**Concept**: Model-Agnostic Meta-Learning (Finn et al., 2017). Train a meta-model such
that 5 gradient steps on 5 labeled examples from a new design produces good predictions.

**Why this could break the skew ceiling**: The 0.107 oracle ceiling assumes zero labeled
examples from the test design. With MAML:
- Few-shot mode: 3-5 labeled CTS runs from new design → 5 gradient steps → MAE < 0.05?
- The meta-model learns the "gradient direction" for adaptation, not the exact solution.

**Architecture**:
- Meta-model: MLP[z_knobs(4) + placement_features(29)] → z_skew
- Inner loop: 5 SGD steps on k=5 support examples from new design
- Outer loop: standard MAML loss on query set (remaining runs)

**Implementation (Reptile, simpler than MAML)**:
```python
# Reptile: simpler to implement, similar performance
def reptile_step(meta_model, task_data, n_inner=5, inner_lr=0.1):
    fast_model = copy.deepcopy(meta_model)
    for _ in range(n_inner):
        loss = compute_loss(fast_model, task_data['support'])
        grads = torch.autograd.grad(loss, fast_model.parameters())
        update_model(fast_model, grads, inner_lr)

    # Move meta-model toward fast_model
    meta_grad = (fast_model.params - meta_model.params) / meta_lr
    meta_model.update(meta_grad)
```

**LODO evaluation**:
- Hold out entire design (e.g., AES)
- Provide 3 support examples from AES (3 CTS runs, labeled)
- Evaluate on remaining 7 runs per AES placement

**Expected improvement for skew**: Could achieve 0.05-0.10 with 3-5 labeled support examples.
This would break the zero-shot ceiling since it uses a few test-design labels.

---

## Novel Approach 3: Per-Placement Kernel Ridge Regression ★★★

**Concept**: For LODO, we have 3 training designs × ~120 placements × 10 runs.
Train a kernel ridge regressor on (normalized_knobs × placement_features) → outcome.
The kernel is: k(x,x') = exp(-gamma * ||f_placement(x) - f_placement(x')||²)

**Key insight**: Two placements with similar (core_util, density, aspect_ratio)
have similar CTS response functions. The Gaussian kernel enforces that nearby
placements in feature space have similar knob→outcome mappings.

```python
from sklearn.kernel_ridge import KernelRidge

# Features: knob values × placement features (concatenated)
# Kernel: RBF with gamma tuned via leave-one-placement-out CV
X_combined = np.hstack([knob_features, placement_features])
krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
krr.fit(X_train, y_train)
```

**Why better than LightGBM**: KRR directly regularizes function smoothness.
For power (smooth P ∝ 1/cd), the kernel enforces that nearby knob configs give
similar predictions. LightGBM can have discontinuities from tree splits.

---

## Novel Approach 4: Contrastive Learning for Design-Invariant Graph Embeddings ★★

**Concept**: Learn a graph encoder E: graph → R^d such that:
- Same design, different placements → far in embedding space (different topologies)
- Same design, different knobs → close (same topology, different CTS parameters)
- **Crucially**: similar_topology(design_A) ≈ similar_topology(design_B) in embedding

**Contrastive loss**:
```python
# Anchor: graph of placement i, knob config k
# Positive: same placement i, different knob config k'
# Negative: different design j
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

**Why novel**: This directly addresses the design-identity leakage problem.
By explicitly pushing embeddings to be knob-aware but design-invariant,
the model learns features that generalize across the design boundary.

---

## Novel Approach 5: Physical Simulation — Fast Elmore Delay Estimator ★★

**Concept**: For each timing path (from timing_paths.csv), compute:
1. Wire length = Euclidean distance between launch and capture FF positions
2. RC delay ≈ 0.69 × R_wire × C_load = 0.69 × (len × R/µm) × (cap × C/µm)
3. sky130 parameters: R_metal2 ≈ 0.2 Ω/µm, C_metal2 ≈ 0.2 fF/µm
4. Skew = max(delay_i) - min(delay_i) over all launch-capture pairs

**Per-CTS-run**: cluster_dia modifies routing (FFs within cluster_dia get grouped →
shorter intra-cluster routing). Apply cluster_dia as a routing modifier.

**Implementation**:
```python
def fast_elmore_skew(timing_paths, ff_positions, cluster_dia, buf_dist):
    delays = []
    for _, path in timing_paths.iterrows():
        launch_pos = ff_positions.get(path['launch_flop'])
        capture_pos = ff_positions.get(path['capture_flop'])
        if launch_pos and capture_pos:
            dist = np.hypot(launch_pos[0]-capture_pos[0], launch_pos[1]-capture_pos[1])
            # Cluster grouping reduces effective distance
            effective_dist = max(dist - cluster_dia, 0)
            delay = 0.69 * 0.2e-3 * effective_dist * 0.2e-3 * effective_dist
            delays.append(delay)
    return max(delays) - min(delays) if delays else 0.0
```

**Expected improvement**: Provides a physics-grounded skew estimator that varies
with cluster_dia per run. Could be used as an additional feature or as a pre-conditioning
value that the model corrects.

---

## Novel Approach 6: GNN with FiLM-Conditioned Message Passing ★★★

**Concept**: Instead of injecting CTS knobs only in the final MLP head, condition
ALL message passing layers on the CTS knobs via FiLM (Feature-wise Linear Modulation).

```python
class FiLMLayer(nn.Module):
    def __init__(self, d_node, d_knob):
        super().__init__()
        self.gamma = nn.Linear(d_knob, d_node)  # scale per knob config
        self.beta  = nn.Linear(d_knob, d_node)  # shift per knob config

    def forward(self, h, knobs):
        gamma = self.gamma(knobs).unsqueeze(0)  # [1, d_node]
        beta = self.beta(knobs).unsqueeze(0)
        return h * (1 + gamma) + beta  # modulate each FF's representation
```

**Physics**: When cluster_dia is large, FFs across a wider area get clustered together →
the message passing should aggregate information at a coarser scale. FiLM allows
the aggregation window to adapt to the CTS knob values.

**Expected improvement**: Skew: 0.15-0.18 (the knob-conditioned GNN can learn
"at cd=70µm, FFs within 70µm radius have identical clock arrival → skew from outliers only")

---

## Novel Approach 7: Hierarchical Bayesian Regression ★★

**Concept**:
- Level 1 (placement): y_ij = α_i × cd + β_i × cs + ε (placement-specific coefficients)
- Level 2 (design): α_i ~ N(μ_design, σ²_α) (design-specific hyperprior)
- Level 3 (global): μ_design ~ N(μ_global, σ²_global)

For LODO (new design, no labeled data): use global prior only.
Prediction: E[y_new | x_new] = μ_global × x_new (prior mean only)

**Why better than flat regression**: Explicitly models within-placement vs between-placement
vs between-design variance components. The prior mean from training designs is a
principled prediction for unseen designs.

**Implementation**: PyMC or Stan. Can be approximated with sklearn BayesianRidge.

---

## Novel Approach 8: Improved GNN — Per-Placement Z-score Targets + Order Stats ★★★

**This is the highest-priority fix. Currently blocking the GNN from its potential.**

### Changes needed in direct.ipynb/matrix.ipynb:

1. **Fix normalization** (direct.ipynb, load_cts_parameters in helper.py):
```python
# WRONG (current):
targets = {'skew': row['z_skew_setup'], ...}  # global z-scores

# CORRECT:
# Pre-compute per-placement mu/sigma, then:
z_skew = (row['skew_setup'] - placement_mu['skew']) / placement_sigma['skew']
targets = {'skew': z_skew, ...}
```

2. **Order statistics skew head** (instead of mean pooling after compression):
```python
class SkewHead(nn.Module):
    def __init__(self, d_ff, d_knob):
        super().__init__()
        # Attention score = timing path involvement
        self.attn = nn.Linear(d_ff, 1)
        self.mlp = nn.Sequential(nn.Linear(d_ff + d_knob, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ff_feats, timing_degree, knobs):
        # Weight FFs by timing path involvement (not compressed!)
        weights = torch.sigmoid(self.attn(ff_feats)) * timing_degree.unsqueeze(-1)
        weights = weights / weights.sum()
        # Top-20% FFs are most critical
        k = max(1, int(0.2 * len(ff_feats)))
        topk_idx = weights.squeeze().topk(k).indices
        topk_feats = ff_feats[topk_idx].mean(0)  # mean of TOP-k only
        return self.mlp(torch.cat([topk_feats, knobs]))
```

3. **Proper LODO evaluation**: Hold out ALL placements of ONE design, not just some.

---

## Novel Approach 9: Wasserstein Embedding for Design Similarity ★★

**Concept**: Use optimal transport distance between FF position distributions as a
kernel for cross-design generalization.

Two designs are "similar" if their FF spatial distributions are close in Wasserstein space.
This is a TOPOLOGY-FREE similarity that doesn't use design identity.

```python
from scipy.stats import wasserstein_distance

def design_similarity(ff_pos_A, ff_pos_B):
    # Normalize positions to [0,1]
    wass_x = wasserstein_distance(ff_pos_A[:,0], ff_pos_B[:,0])
    wass_y = wasserstein_distance(ff_pos_A[:,1], ff_pos_B[:,1])
    return np.sqrt(wass_x**2 + wass_y**2)

# Use as kernel: k(A, B) = exp(-wass_distance(A, B)² / 2σ²)
# Smaller Wasserstein distance → similar layout → similar CTS response
```

**For LODO**: When predicting on unseen design D_new, weight predictions from
training designs by exp(-Wasserstein(D_new, D_train)²). Design most similar in
FF distribution gets highest weight.

---

## Novel Approach 10: Residual Calibration with Liberty-Based Electrical Features ★★

**Concept**: The Liberty file gives EXACT input capacitances per cell type.
Driven capacitance = sum of input caps of all cells in the clock net.
Power = α × C_total × f × V² is EXACT (not approximate) with Liberty data.

Currently in v13 for absolute prediction. Extension to relative prediction:
- For each CTS run, estimate driven_cap differently (CTS changes buffer placements)
- But: driven_cap is PRE-CTS (logic cells are fixed), so it's the same per placement
- The CHANGE in driven_cap with CTS knobs comes from buffer count changes
- Buffer count ≈ n_ff / cluster_size (approx)
- Liberty-based: C_clock = C_ff_inputs + C_buf × (n_ff/cluster_size)

This gives a physics-exact estimate of power that could replace or complement cluster_dia rank.

---

## Priority Implementation Order

1. **CRITICAL — Fix GNN normalization** (1 hour, direct.ipynb)
   - Change global z-scores → per-placement z-scores
   - Expected: skew 0.14 → 0.10-0.12, power 0.08 → 0.04-0.06

2. **HIGH — Run v13 absolute predictor** (already coded, need results)
   - MST + Wasserstein + Liberty caps + GP regressor
   - Expected: LODO power 32% → 20-25%, WL 13% → 8-10%

3. **HIGH — Per-placement GP regressor for relative prediction** (2 hours)
   - Use Approach 1: GP with ARD RBF kernel on z-knobs
   - Replace LightGBM for power/WL
   - Expected: power 0.0656 → 0.03-0.04, WL 0.0849 → 0.05-0.07

4. **HIGH — Order statistics skew head in GNN** (2 hours, direct.ipynb)
   - Keep raw FF features for skew head
   - Top-K timing-critical FFs with attention pooling
   - Expected: skew 0.14 → 0.10-0.12

5. **MEDIUM — FiLM-conditioned message passing** (3 hours)
   - Knob-conditioned GNN message passing
   - Expected: all tasks improve by 10-20%

6. **MEDIUM — Wasserstein design similarity weighting** (2 hours)
   - Weight training samples by Wasserstein distance to test design
   - Could help for designs with different n_ff regimes

7. **MEDIUM — MAML few-shot for skew** (4 hours, if above still fails)
   - Uses 3 labeled test examples for adaptation
   - The only proven path to < 0.10 skew

8. **LOW — Fast Elmore delay simulator** (3 hours)
   - Physics-based skew estimate as additional feature
   - Expected: marginal improvement only (similar to tight path features)

---

## Theoretical MAE Floors

| Task | Zero-shot floor | Few-shot (3 ex) floor | In-sample floor |
|------|-----------------|-----------------------|-----------------|
| Skew | ~0.20 (oracle: 0.107) | ~0.05 (MAML estimate) | ~0.005 (quadratic fit) |
| Power | ~0.035 (GP estimate) | ~0.01 | ~0.002 |
| WL | ~0.050 (GP+MST estimate) | ~0.01 | ~0.003 |

**The 0.0005 MAE target requires in-sample-quality prediction on unseen designs.**
This is achievable only via:
1. More diverse training designs (20+), OR
2. Few-shot adaptation with labeled test examples (MAML/GP), OR
3. Full physics simulation (Elmore delay model with Liberty parasitics)

---

## Key Literature Not Yet Fully Exploited

1. **Neural Processes (Garnelo et al., 2018)**: Amortized inference for arbitrary function
   regression. Meta-learns from context set → similar to MAML but Bayesian.
   Apply: context = 5 labeled CTS runs from test design → posterior prediction on rest.

2. **DeepKernel GP (Wilson & Adams, 2013)**: Neural network as feature extractor for GP kernel.
   Combines expressive power of NNs with GP uncertainty calibration.

3. **Equivariant GNN (Satorras et al., 2021 - EGNN)**: GNN that is equivariant to
   translation/rotation of FF positions. Clock tree routing IS translation-invariant.
   Better than standard GNN that uses absolute coordinates.

4. **Structural Graph Wavelets (Hammond, 2011)**: Multi-scale graph features using
   spectral graph wavelets. Captures both local and global structure without compression.

5. **FNO (Fourier Neural Operator, Li et al., 2021)**: For spatial domains, learn the
   mapping between functions. Generalize across different resolutions/designs.
