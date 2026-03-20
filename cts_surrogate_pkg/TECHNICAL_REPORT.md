# CTS-Surrogate: Technical Deep-Dive Report

**File**: `cts_surrogate.py`
**Purpose**: Everything you need to understand what this model does, how it works, why every design choice was made, and what the physics behind each feature means.

---

## 1. The Problem Being Solved

Clock Tree Synthesis (CTS) is a step in chip design where the EDA tool routes a clock signal from one source to thousands of flip-flops (FFs) across the chip. The quality of this routing determines three outcomes that matter enormously:

- **Skew**: How much variation is there in when the clock arrives at different FFs? High skew → timing failures.
- **Power**: How much energy does the clock network consume? Clock trees typically consume 20–40% of total chip power.
- **Wirelength (WL)**: How many millimeters of wire does the clock tree use? Determines routing congestion and manufacturing cost.

The EDA tool (e.g., OpenROAD) accepts four **knobs** that control how it builds the clock tree:

| Knob | What it controls | Symbol |
|------|-----------------|--------|
| `cd` — cluster diameter | Maximum distance (µm) within which FFs are grouped into one cluster | 20–70 µm |
| `cs` — cluster size | Maximum number of FFs in one cluster | 10–30 |
| `mw` — max wire | Maximum single wire segment length (µm) | 100–280 µm |
| `bd` — buffer distance | Maximum distance (µm) between buffer insertion points | 60–150 µm |

**Each actual CTS run takes 10–30 minutes.** There are ~5000 reasonable knob combinations. Evaluating all of them is physically impossible. This surrogate predicts all four outcomes in **< 5 ms** — 4–5 orders of magnitude faster — allowing real-time Pareto optimization.

---

## 2. High-Level Architecture

```
Physical design files                     CTS knobs
(DEF + SAIF + timing_paths.csv)          (cd, cs, mw, bd)
        │                                      │
        ▼                                      │
  FeatureEngine                                │
  ─────────────                                │
  Parses files once,               ┌───────────┘
  caches in memory.                │
  Builds feature                   │
  vectors on demand.               │
        │                          │
        └────────────┬─────────────┘
                     │
            ┌────────▼────────┐
            │  Shared context │  29 dims, knob-free
            │  (placement     │  geometry + activity
            │   fingerprint)  │  + timing slack
            └────────┬────────┘
                     │ + 8 knob dims (log + raw)
           ┌─────────┼──────────────────────┐
           │         │                      │
    ┌──────▼──┐ ┌────▼───┐ ┌──────▼──┐ ┌───▼──────┐
    │  Power  │ │  WL    │ │  Skew  │ │  Hold    │
    │  XGB    │ │LGB+Rdg │ │  XGB   │ │  LGB     │
    │  76 dim │ │ 84 dim │ │ 63 dim │ │  66 dim  │
    └────┬────┘ └────┬───┘ └───┬────┘ └────┬─────┘
         │           │         │            │
         └───────────┴─────────┴────────────┘
                           │
                  CTSPrediction(power_mW, wl_mm,
                                skew_z, hold_vio)
                           │
                  ParetoOptimizer (NSGA-II)
```

The key architectural insight: **parse once, predict millions of times.** The DEF/SAIF/timing files are parsed once and their features cached. Building a feature vector for a new (knob) configuration takes microseconds because it just patches the cached placement context with new knob values.

---

## 3. The Three Input Files and What We Extract

### 3.1 DEF File — Physical Placement

A DEF (Design Exchange Format) file describes where every cell is placed on the chip die. We parse it with regex in `_parse_def()`.

**What we extract:**

**Die geometry:**
- `die_area` — total chip area in µm² (denominator for density features)
- `die_w`, `die_h` — die width and height
- `die_aspect` — width/height ratio. Non-unity aspect ratios create asymmetric routing distances, which matters for clock balance.

**Flip-flop positions:**
- We find all cells matching `sky130_fd_sc_hd__df*` (D-type flip-flops in the SkyWater 130nm library)
- Extract their (x, y) coordinates in µm
- Compute: `ff_hpwl` = bounding box half-perimeter of all FFs (the best single proxy for clock tree length — the Steiner minimum tree length is bounded by HPWL), `ff_spacing` = √(bounding_box_area / n_ff) ≈ average inter-FF distance, `ff_cx`, `ff_cy` = centroid (normalized by die size), `ff_x_std`, `ff_y_std` = spatial spread

**Cell population:**
The cell mix tells us about the design's logic style, which affects switching activity:
- `n_ff` — flip-flop count (primary size metric)
- `n_buf`, `n_inv` — buffer and inverter count
- `n_xor_xnor` — XOR/XNOR count → critical for power (these toggle at full rate in AES, SHA-256)
- `n_mux` — MUX count
- `frac_xor`, `frac_mux` — fractions of active cells that are XOR and MUX
- `comb_per_ff` — combinational cells per FF (logic depth proxy)

**Drive strength:**
Each cell name ends in a drive strength number (e.g., `buf_2`, `inv_4`, `dfxtp_1`). We extract these and compute:
- `avg_ds` — mean drive strength across all placed cells
- `std_ds`, `p90_ds` — spread and tail of drive strength distribution
- `frac_ds4plus` — fraction of cells with drive strength ≥ 4

Drive strength matters for power because `P ∝ C_load × V² × f`, and drive strength determines the output capacitance.

### 3.2 SAIF File — Switching Activity

A SAIF (Switching Activity Interchange Format) file records how often each net toggles during a simulation. We parse it in `_parse_saif()`.

**What we extract:**
- `TC` values (toggle counts) for every net
- `rel_act = mean_toggles / max_toggles` — relative activity. This is the most important activity feature. A design with `rel_act = 0.8` has most of its logic switching near maximum frequency. SHA-256 has very high `rel_act` because its XOR-heavy datapath is fully active during hashing.
- `tc_std_norm` — normalized standard deviation of toggle counts. High variance means some nets switch much more than others.
- `frac_zero` — fraction of nets that never toggle (inactive logic)
- `frac_high_act` — fraction of nets toggling more than 2× average

### 3.3 timing_paths.csv — Timing Slack Distribution

This file lists the timing slack for every setup-timing path in the design. Slack = (required arrival time) − (actual arrival time). Negative slack = timing violation.

**What we extract:**
- `slack_mean`, `slack_std` — distribution center and spread
- `slack_min`, `slack_p10` — worst-case slack (most critical paths)
- `frac_neg`, `frac_tight`, `frac_critical` — fractions with slack < 0, < 0.5ns, < 0.1ns

**Why timing slack predicts CTS quality:** Tight timing paths constrain buffer insertion. If the worst timing path has almost no slack, the CTS tool cannot freely insert buffers to balance the clock tree without potentially creating hold violations.

### 3.4 Critical-Path Spatial Features (for Skew only)

Computed by `_parse_skew_spatial()`, these are the most important features for skew prediction. The function finds the **worst 50 timing paths** and locates the launch/capture FFs of each path in physical space.

**Features extracted:**
- `crit_max_dist` — normalized maximum spatial distance between any launch-capture FF pair on a critical path (units of grid cells)
- `crit_mean_dist`, `crit_p90_dist` — distribution of these distances
- `crit_ff_hpwl` — bounding box half-perimeter of all FFs on critical paths only (vs. all FFs)
- `crit_cx_offset`, `crit_cy_offset` — how far the critical-path FF centroid is from the overall FF centroid. Non-zero = asymmetric clock tree needed.
- `crit_star_degree` — whether critical paths form a "star" topology (many paths from one FF) vs. chain
- `crit_chain_frac` — fraction of critical paths that are sequential (chains)
- `crit_asymmetry` — spatial asymmetry of critical-path FFs
- `crit_eccentricity` — elongation of the critical-path FF cluster
- `crit_density_ratio` — density of critical-path FFs relative to all FFs

**Why this matters for skew:** Skew is a worst-case metric: `skew = max(clock_arrival) - min(clock_arrival)`. The worst-case is almost always determined by the FFs that appear in the slowest timing paths — exactly the critical-path FFs. By computing spatial statistics specifically on these FFs (not all FFs), we capture the signal that determines skew without averaging it away.

---

## 4. The Shared Context Vector (29 dims)

Built by `_shared_ctx()`, this is the placement "fingerprint" shared across the power, WL, and hold heads. It is **knob-independent** — the same for all 10 CTS runs on a given placement.

```
[0]  log(n_ff)              — design scale (log because 142 to 5000+ FFs)
[1]  log(die_area)          — physical footprint
[2]  log(ff_hpwl)           — Steiner tree length proxy
[3]  log(ff_spacing)        — average inter-FF distance
[4]  die_aspect             — die shape (non-square creates routing asymmetry)
[5]  1.0                    — bias term
[6]  ff_cx                  — normalized FF centroid x (0=left, 1=right)
[7]  ff_cy                  — normalized FF centroid y
[8]  ff_x_std               — FF horizontal spread
[9]  ff_y_std               — FF vertical spread
[10] frac_xor               — XOR/XNOR fraction (high → high toggle rate)
[11] frac_mux               — MUX fraction
[12] frac_and_or            — AND/OR fraction
[13] frac_nand_nor          — NAND/NOR fraction
[14] frac_ff_active         — flip-flop fraction of active cells
[15] frac_buf_inv           — buffer+inverter fraction (clock tree overhead)
[16] comb_per_ff            — logic depth per FF
[17] avg_ds                 — mean drive strength
[18] std_ds                 — drive strength spread
[19] p90_ds                 — 90th percentile drive strength
[20] frac_ds4plus           — fraction with high drive strength
[21] log(cap_proxy)         — n_active × avg_ds (total capacitive load proxy)
[22] rel_act                — relative switching activity
[23] mean_sig_prob          — mean signal probability
[24] tc_std_norm            — activity variability
[25] frac_zero              — fraction of inactive nets
[26] frac_high_act          — fraction of highly active nets
[27] log_n_nets             — total net count (log)
[28] n_nets / n_ff          — nets per FF (connectivity density)
```

The log transforms on [0-3, 21, 27] handle the large range of values across designs (142 to 5000+ FFs, 35,000 to 4,000,000+ µm² die area).

---

## 5. The Four Prediction Heads

Each head appends target-specific features to the shared context, then passes through a StandardScaler and a tree ensemble.

### 5.1 Power Head (76 dims, XGBoost)

**Physics**: `P_clock = α × C_total × V² × f`

Where `C_total = C_wire × WL + C_buf × n_buffers`. So power depends on:
1. Clock frequency `f` (f_ghz)
2. Total wire capacitance (∝ WL, which depends on placement)
3. Buffer count (∝ n_ff / cluster_size, determined by CTS knobs)
4. Toggle activity (α, from SAIF)

**Physics normalizer**: `pw_norm = n_ff × f_GHz × avg_ds`

This removes the first-order size and frequency dependence, so the model learns the relative power variation due to CTS knobs, not absolute scale.

**Feature breakdown (76 dims)**:
- [0–28]: Shared context
- [29–35]: Design params: f_GHz, t_clk, synth_delay, synth_leakage, synth_area, core_util, density
- [36–43]: Knob block: log(cd), log(cs), log(mw), log(bd), cd, cs, mw, bd
- [44–57]: 14 interaction terms (see below)
- [58–75]: 18 timing features

**Key interaction terms for power:**
- `frac_xor × comb_per_ff` — XOR-heavy logic with deep pipelines → high switching activity
- `rel_act × frac_xor` — activity amplified by XOR density
- `log(cd × n_ff / die_area)` — cluster density: large clusters in dense designs reduce buffer count more aggressively
- `log(cs × ff_spacing)` — cluster size relative to physical FF spacing: if cs is large but FFs are spread out, many clusters are underfilled
- `log(mw × ff_hpwl)` — max wire relative to tree extent: long wire budget on a large design allows fewer buffer stages
- `log(n_ff / cs)` — approximate number of clusters (buffer count proxy)
- `log(n_active × rel_act × f_ghz)` — total switching workload

**Why XGBoost?** The power response is nonlinear in the knobs (doubling cluster size doesn't halve buffer count — it depends on the spatial distribution of FFs). XGBoost naturally handles these piecewise nonlinearities without requiring hand-crafted basis functions.

**The SHA-256 floor**: SHA-256 achieves only 48.9% MAPE zero-shot. SHA-256 is a hash function whose datapath is dominated by XOR operations (`frac_xor >> 0.5`). Its `rel_act` is genuinely high — the design really does have most of its logic switching every cycle. Clipping `rel_act` to the training range makes this worse because SHA-256's activity is not an outlier — it's a different design regime (logic-switching dominated vs. clock-tree-dominated).

### 5.2 Wirelength Head (84 dims, LightGBM + Ridge blend)

**Physics**: From the DME (Deferred Merge Embedding) algorithm for clock trees:
`WL ≈ 1.1–1.5 × HPWL(FF bounding box)`

More precisely: `WL = k₁ × HPWL − k₂ × cluster_dia + k₃ × (n_ff / cluster_size)`

Where the first term is the Steiner tree, the second is the reduction from merging nearby FFs, and the third is inter-cluster routing overhead.

**Physics normalizer**: `wl_norm = √(n_ff × die_area)`

This is motivated by: if FFs are uniformly placed in a die of area A with density n_ff/A, the average nearest-neighbor distance scales as √(A/n_ff), and the total tree length scales as n_ff × (A/n_ff)^(1/2) = √(n_ff × A).

**Feature breakdown (84 dims)**:
- [0–28]: Shared context
- [29–32]: Design params: f_GHz, t_clk, core_util, density (no synth for WL)
- [33–40]: Knob block (log + raw)
- [41–52]: 12 interaction terms
- [53–71]: 19 gravity/graph features
- [72–74]: 3 extra scale features
- [75–83]: 9 net routing features (the breakthrough)

**Net routing features (9 dims) — the WL breakthrough:**

These were added in v21 and cut WL MAPE from 11.0% → 7.0%. They come from signal net geometry in the DEF file:
- `log(rsmt_total)` — total RSMT (Rectilinear Steiner Minimum Tree) length of all signal nets. The clock tree length correlates with signal routing density — dense signal routing means shorter average wire paths and denser FF placement, which means a shorter clock tree.
- `rsmt / (n_ff × √die_area)` — normalized RSMT (routing density per unit)
- `net_hpwl_mean`, `net_hpwl_p90` — signal net bounding box statistics
- `frac_high_fanout` — fraction of nets with > 4 fanout
- `rudy_mean`, `rudy_p90` — RUDY (Routing Utilization DensitY) congestion metrics
- `rsmt_total × cd / (n_ff × die_area)` — interaction: routing density with cluster diameter
- `rudy_mean × cd` — congestion with clustering

**Critical insight: RSMT as feature, not normalizer.** An early attempt (v19) divided WL by RSMT as a normalization. This made things worse (20.2% MAPE vs. 11.0%). The reason: RSMT is a *proxy for routing environment*, not a *predictor of clock tree structure*. Using it as a feature lets the model learn "dense signal routing → shorter clock tree" as a learned correlation, which is the correct direction of causality.

**Why LightGBM + Ridge blend (α=0.3)?**

`WL_pred = exp(0.3 × LGB_output + 0.7 × Ridge_output) × wl_norm`

Ridge regression is a linear model. On unseen designs, linear models extrapolate more gracefully than nonlinear ones. LightGBM captures the within-training-set nonlinearities. The 0.3/0.7 blend uses LGB for 30% of the signal (capturing knob×geometry interactions) and Ridge for 70% (ensuring stable extrapolation to new design families). This blend was found empirically to reduce WL MAPE from ~9% (LGB alone) to ~7%.

### 5.3 Skew Head (63 dims, XGBoost)

**Physics**: `skew = max(clock_arrival) − min(clock_arrival)`

Skew is a worst-case metric. It is determined by the *tail* of the path length distribution, not the mean. This is why skew prediction is fundamentally harder than power or WL prediction, and why features that work for power/WL (mean FF positions, average distances) fail for skew.

**Physics normalizer**: Per-placement z-score.

For each placement, the 10 CTS runs produce a distribution of skew values. We z-score each run's skew relative to that placement's distribution:
```
z_skew = (skew − μ_placement) / σ_placement
```

This is the correct normalization because:
1. It removes the design-level baseline (different circuits have different inherent skews)
2. It trains the model to predict **how sensitive skew is to CTS knobs** for this specific placement
3. It makes the target dimensionless and comparable across designs

**Feature breakdown (63 dims)**:
- [0–21]: Skew-specific context (22 dims, subset of shared ctx with timing slack and activity)
- [22–29]: Knob block (log + raw)
- [30–45]: 16 critical-path spatial features (from `_parse_skew_spatial`)
- [46–62]: 17 physics interaction terms

**The 17 physics interactions for skew** encode the Elmore delay model relationships:
- `cd / (ff_spacing + 1)` — cluster diameter relative to FF spacing. If cd >> spacing, most FFs get merged into a few large clusters → hard to balance → high skew.
- `bd / (crit_max_dist_um + 1)` — buffer distance relative to longest critical path. If bd << max critical path, many buffers inserted → more equalization → lower skew.
- `mw / (crit_max_dist_um + 1)` — max wire relative to critical path length.
- `crit_star × cd` — star topology (many paths from one hub FF) amplified by large cluster diameter → hard to equalize.
- `crit_asymmetry × mw` — spatially asymmetric critical paths with large wire budget → can equalize.
- `log(crit_max_um / (cd + 1))`, `log(crit_max_um / (bd + 1))`, `log(crit_max_um / (mw + 1))` — ratios of physical constraints.
- `crit_cx × cd`, `crit_cy × mw` — centroid offset interactions (off-center critical paths with specific knobs).

**Why the critical-path spatial features are the key breakthrough:**

Before adding these features, skew MAE was ~0.237. After: 0.074 (3.2× improvement). The reason is direct: the EDA tool builds the clock tree to minimize skew on the *critical timing paths*. A model without knowledge of where those paths are in physical space cannot reason about clock tree balance for those specific paths.

### 5.4 Hold Violation Head (66 dims, LightGBM)

**Physics**: Hold violations occur when the clock arrives at a capture FF **too early**, before data from the launch FF has propagated. This is the opposite problem from setup timing.

`skew ↔ hold_vio correlation: r = −0.96` (near-perfect anti-correlation across all designs)

The reason: increasing `mw` allows longer individual wire segments, which gives the tool more flexibility to balance paths and reduce skew. But longer wires create higher clock latency variance, increasing the probability that a capture FF sees the clock before the data arrives.

**Feature breakdown (66 dims)**:
- Same base as WL head: shared ctx + design params + knob block + 12 interactions
- 9 hold-specific features that encode the skew↔hold coupling:
  - `log(n_ff / cs)` — number of clusters (more clusters = more potential hold paths)
  - `bd / (crit_max_dist + 1)` — buffer spacing relative to critical path length
  - `crit_star × cs` — star topology with cluster granularity
  - `crit_chain × bd` — chain topology with buffer distance
  - `crit_asymmetry × cd` — asymmetric critical paths with cluster diameter
- 4 net routing features (RUDY congestion affects hold via detours)

**Why LightGBM for hold?** Hold violations are counts (integers 0, 1, 2, ...), often zero. LightGBM handles zero-inflated regression better than XGBoost for this distribution because of its leaf-wise tree growth.

---

## 6. Target Transformations (What the Model Actually Predicts)

The models don't predict raw power, WL, or skew. They predict transformed targets, which is critical for generalization:

### Power
```python
y_train = log(power_W / pw_norm)
y_pred  = exp(model.predict(X)) × pw_norm
```
Log-transforming the normalized power converts multiplicative errors to additive ones. The model learns to predict the log-ratio of actual power to the physics-predicted baseline.

### Wirelength
```python
y_train = log(wl_um / wl_norm)
y_pred  = exp(0.3 × lgb_out + 0.7 × ridge_out) × wl_norm
```
Same log-ratio structure. The blend `exp(α × lgb + (1-α) × ridge)` means the two models predict log-WL, and the blend is a geometric mean.

### Skew
```python
y_train = (skew - μ_placement) / σ_placement
y_pred  = model.predict(X)   # directly a z-score
```
No exp transform because skew is already on a reasonable scale. The z-score is directly predicted.

### Hold violations
```python
y_train = log1p(hold_vio_count)
y_pred  = expm1(clip(model.predict(X), 0, 20))
```
`log1p` handles the zero-inflation (log(1+0) = 0). `expm1` inverts it. The clip to [0, 20] prevents `exp(20) ≈ 5×10⁸` from exploding.

---

## 7. Batch Prediction and the _patch() Trick

The `batch_build()` method enables evaluating 5000 knob configurations in a single call. The key insight: for a fixed placement, only the knob-dependent features change across different configurations. Everything else is constant.

```python
def _patch(x0, head):
    X = np.tile(x0, (N, 1))        # N copies of the base vector
    # Update log-knob positions
    for li, v in zip(_KNOB_LOG[head], [cd_arr, cs_arr, mw_arr, bd_arr]):
        X[:, li] = np.log1p(v)
    # Update raw-knob positions
    for ri, v in zip(_KNOB_RAW[head], [cd_arr, cs_arr, mw_arr, bd_arr]):
        X[:, ri] = v
    # Update knob×placement interaction terms
    for ii, kind in _KNOB_INTER[head]:
        if kind == 'cd':       X[:,ii] = log1p(cd_arr × n_ff / die_area)
        elif kind == 'cs':     X[:,ii] = log1p(cs_arr × ff_spacing)
        elif kind == 'mw':     X[:,ii] = log1p(mw_arr × ff_hpwl)
        elif kind == 'cs_inv': X[:,ii] = log1p(n_ff / cs_arr)
    return X
```

Instead of calling `build()` 5000 times (which would recompute all placement features 5000 times), `batch_build()`:
1. Calls `build()` once with median knobs to get the base vector `x0`
2. Tiles it N times
3. Patches only the knob-dependent positions

This achieves ~100× speedup over naive looping. The critical correctness requirement is that `_KNOB_LOG`, `_KNOB_RAW`, and `_KNOB_INTER` accurately encode which positions in each feature vector are knob-dependent.

---

## 8. The Pareto Optimizer

### Why multi-objective?

The four CTS outcomes trade off against each other:
- Power ↔ Skew: `cd↑` reduces power (fewer buffers) but has mixed skew effects
- Skew ↔ Hold: `mw↑` reduces skew but increases hold violations (r = −0.96)
- Power ↔ WL: coupled in clock-dominated designs (AES r=0.61) but independent in logic-dominated (PicoRV32 r=0.05)

No single knob configuration minimizes all four objectives. The Pareto front is the set of configurations where improving any one objective requires worsening at least one other.

### NSGA-II (the new default)

NSGA-II (Non-dominated Sorting Genetic Algorithm II) works by:
1. Initialize population of N=100 random knob configurations
2. Evaluate all four objectives for each
3. Assign "rank" by non-domination: rank 1 = Pareto front, rank 2 = Pareto front after removing rank 1, etc.
4. Assign "crowding distance" within each rank: solutions in sparse regions of objective space get higher crowding distance (diversity preservation)
5. Select parents by tournament selection on (rank, crowding distance)
6. Generate offspring via SBX crossover and polynomial mutation
7. Combine parents + offspring, re-rank, keep best N
8. Repeat for 50 generations

**Why NSGA-II beats random search:**
- Random search at N=5000 finds 27 Pareto solutions for AES
- NSGA-II at N=5000 finds 158 Pareto solutions (6× more)
- NSGA-II at N=500 already finds 41 solutions, matching random-5000's count
- ETH MAC best skew: random −0.795z → NSGA-II −0.960z (+20.7%)
- SHA-256 best skew: random −0.742z → NSGA-II −0.952z (+28.4%)

The reason: evolutionary pressure guides the search toward regions of objective space that random sampling rarely reaches. For skew in particular, the optimal region (large `mw`, specific `cd`/`cs` balance) is a small corner of the 4D space that random sampling hits infrequently.

### Bayesian search (Optuna NSGA-II sampler)

Optuna's NSGA-II sampler builds a probabilistic model of the objective function surface and proposes configurations that balance exploitation (near known good solutions) and exploration. It achieves better skew extremes with 10× fewer evaluations:
- AES best skew_z: random −0.896 → Bayesian@500 −1.014 (13.2% better)
- PicoRV32: random −0.641 → Bayesian@500 −0.722 (12.6% better)

Best choice: NSGA-II is the default (same speed, much better coverage). Use Bayesian when evaluations are expensive or you specifically want to minimize one objective deeply.

---

## 9. K-Shot Calibration: Handling Unseen Design Regimes

When the surrogate encounters a design outside its training regime (e.g., zipdiv with 142 FFs vs. 1597–5000 in training), the physics normalizers extrapolate incorrectly:

- `pw_norm = n_ff × f_GHz × avg_ds` → underestimates by **2.3×** for zipdiv
- `wl_norm = √(n_ff × die_area)` → underestimates by **605×** for zipdiv

The systematic scale error is corrected by **multiplicative K-shot calibration**:

```python
# K support samples from the new design
k_hat_pw = median(actual_power[0:K] / predicted_power[0:K])
k_hat_wl = median(actual_wl[0:K]    / predicted_wl[0:K])

# Apply to all predictions
power_calibrated = k_hat_pw × power_raw
wl_calibrated    = k_hat_wl × wl_raw
```

**Why median instead of mean?** At small K, one outlier ratio (e.g., a placement where the clock tree was routed unusually) inflates the mean. The median is robust to this.

**Results on zipdiv:**

| K | Power MAPE | WL MAPE |
|---|-----------|---------|
| 0 (zero-shot) | 55.5% | 99.8% |
| 1 | **5.0% ±1.4** | **4.1% ±0.6** |
| 5 | **4.1% ±0.7** | **3.7% ±0.2** |

With just one CTS run from zipdiv (10–30 min), both targets drop below 5%.

**Skew doesn't need calibration** (already 3.93 ps = 0.77% of true 0.51ns) because per-placement z-scoring is scale-invariant by construction.

---

## 10. Cross-Target Physics: What the Model Implicitly Learns

### Skew ↔ Hold Anti-Correlation (r = −0.96)

Both skew and hold violations respond to `mw` (max wire length), but in opposite directions:
- Large `mw` → tool can use longer individual segments → more equalization → **lower skew**
- Large `mw` → longer paths create higher clock latency variance → some capture FFs see clock late → **more hold violations**

This is the Elmore delay tradeoff: the same mechanism that reduces skew (longer, more balanced routing) also increases hold risk.

### Power ↔ WL Coupling Depends on Design Regime

| Design | cor(power, WL) | Reason |
|--------|---------------|--------|
| AES | 0.61 | **Clock-dominated**: 2994 FFs, large die, clock tree dominates total capacitance |
| ETH MAC | 0.14 | Mixed |
| PicoRV32 | 0.05 | **Logic-dominated**: 1597 FFs, compact die, XOR datapath dominates switching |
| SHA-256 | 0.05 | Logic-dominated |

In AES: `P ≈ k × WL × C_per_unit × V² × f`. Longer clock tree → more capacitance → more power. Direct proportionality.

In PicoRV32/SHA-256: The CTS knobs barely change WL within a placement (CV ≈ 0.8%). Most power variation comes from the combinational logic activity, not the clock tree.

**Consequence for prediction**: The model cannot use predicted WL as a feature for power (cascading errors). It must predict them independently and let the Pareto optimizer reveal the tradeoff.

---

## 11. What the Model Cannot Do

**AES power floor (20.2% oracle MAPE):**
Even with perfect calibration (using all training placements as support), AES power achieves only 20.2% MAPE. The reason: power varies 2.7× across AES placements due to different routing layer utilization (wire capacitance from actual routing paths). The DEF file shows cell placement but not the actual routed wires — that requires a SPEF (Standard Parasitic Exchange Format) file. Without SPEF, the remaining 20% error is physically irreducible.

**True zero-shot on extreme OOD:**
For designs far outside the training size regime (like zipdiv), the normalizers fail on zero-shot but K=1 fixes it. This is an inherent limitation of normalizer-based regression.

**Absolute skew in ns without per-placement statistics:**
The model predicts skew as a z-score. Converting to nanoseconds requires knowing μ and σ for the specific placement (from historical runs or K=1).

---

## 12. Summary: Why Each Design Choice Was Made

| Decision | What was tried instead | Why this works better |
|----------|----------------------|----------------------|
| Per-placement z-score for skew | Global z-score across all designs | Global z-score conflates design-type effect with CTS-parameter effect; model can't generalize |
| Critical-path spatial features for skew | Mean/std of all FF positions | Skew is worst-case; averaging destroys the tail signal |
| XGBoost for power/skew | Neural networks | NN power catastrophically failed (333% MAPE); small dataset (1400 rows) favors trees |
| LGB + Ridge blend for WL | LGB alone | Ridge's linearity stabilizes OOD extrapolation; LGB captures within-distribution nonlinearity |
| RSMT as feature (not normalizer) for WL | Dividing WL by RSMT | Using RSMT as normalizer made things worse (20.2% vs 11.0%); correct direction is correlation, not normalization |
| NSGA-II as default optimizer | Random 5000-sample sweep | 6× more Pareto solutions, 20-28% better minimum skew, same wall time |
| Median k-hat for K-shot | Mean k-hat | Robust to outlier ratios at small K; achieves ≤10% power at K=10 vs. K=20 with mean |
| Multiplicative calibration | Additive bias correction | The error is a scale mismatch (normalizer 2-600× off); multiplicative correction fixes it exactly |
