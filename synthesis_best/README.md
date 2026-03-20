# Synthesis Best — Unified CTS Outcome Predictor

**Locked**: 2026-03-19 (Session 11)
**DO NOT MODIFY** — frozen reference point.

---

## Results (LODO, 4 designs)

### Zero-Shot (no labeled data from test design)

| Design  | Power MAPE | WL MAPE | Skew MAE |
|---------|-----------|---------|----------|
| AES     | 36.6%     | 12.8%   | 0.0859 ✓ |
| ETH MAC | 12.3%     |  7.2%   | 0.0787 ✓ |
| PicoRV  | 30.1%     |  4.3%   | 0.0631 ✓ |
| SHA-256 | 48.9%     |  3.7%   | 0.0675 ✓ |
| **Mean**| **32.0%** | **7.0% ✓**| **0.0738 ✓** |

All 4 designs hit skew target (<0.10). Previous skew best: 0.237 → **3.2× improvement**.
**v21 (2026-03-20)**: Net routing features (RSMT, RUDY) cut WL from 11.0% → **7.0%** (AES: 24.9%→12.8%).

### With K=10 Shot Calibration (power + WL, median k_hat — v20/v21)

| Design  | Power MAPE | WL MAPE | Skew MAE |
|---------|-----------|---------|----------|
| AES     | 19.8% (oracle 20.2%)| 9.8%  | 0.0859 ✓ |
| ETH MAC |  4.3%     |  5.3%   | 0.0787 ✓ |
| PicoRV  |  5.4%     |  3.3%   | 0.0631 ✓ |
| SHA-256 |  9.9%     |  2.7%   | 0.0675 ✓ |
| **Mean**| **9.8% ✓**| **4.8% ✓**| **0.0738 ✓** |

*v20: median k_hat reduces K from 20→10. v21: net features cut WL 11%→7% zero-shot.*

---

## Architecture

### Skew Model (NEW — Session 11)
**LGBMRegressor(n=300, num_leaves=31, lr=0.03)**
- Features (64 dims): placement geometry + CTS knobs + **critical-path spatial features**
  - From DEF: FF positions by name
  - From timing_paths.csv: worst-50 launch-capture FF pairs
  - Computed: crit_mean/max/p90 distance, crit_ff_hpwl, centroid offset,
    spread, boundary fraction, star/chain topology, asymmetry, eccentricity
  - Physics interactions: cd/(ff_spacing), bd/(crit_max_um), mw/(crit_max_um),
    crit_star×cd, crit_asym×mw
- Target: per-placement z-scored skew_setup (normalized per placement's 10 CTS runs)
- Key insight: critical path topology (chain vs star) × CTS knob ratios is the
  design-invariant signal that generalizes across designs

### Power Model (Session 10 baseline, unchanged)
**XGBRegressor(n=300, max_depth=4, lr=0.05, subsample=0.8, colsample_btree=0.8)**
- Features: base_pw(58d, WITH synth) + timing(18d) = 76 dims
- Normalization: n_ff × f_ghz × avg_ds
- K-shot: k_hat = **median**(actual/pred) on K support samples (v20: K=10 → 9.8%)

### WL Model (v21 — Session 12, net features added)
**LGB(300) + Ridge(1000) blend, α=0.3**
- Features: base_wl(53d, NO synth) + gravity(19d) + extra_scale(3d) + **net(9d)** = **84 dims**
- Normalization: sqrt(n_ff × die_area)
- Net features (from DEF NETS section via net_features_cache.pkl):
  - log(rsmt_total), rsmt_total/(n_ff×√die_area), net_hpwl_mean, log(net_hpwl_p90)
  - frac_high_fanout, rudy_mean, rudy_p90, rsmt×cd interaction, rudy×cd interaction
- Key insight: Signal net RSMT (r=0.997 for AES) proxies clock routing density;
  generalization works because physics is the same: dense routing → longer clock tree

---

## Key Discoveries (Session 11)

1. **Critical-path spatial features break the skew floor**: 0.237 → 0.0738 (3.2×)
   - The topology of worst-slack paths (star vs chain, spatial spread) × CTS knob ratios
   - Design-invariant: a star pattern with cluster_dia < crit_max_dist = high skew in any design
   - Implementation: parse DEF for FF positions by name, cross-reference timing_paths.csv

2. **Glitch correction is a dead end** (T1-A):
   - eff_act = rel_act/(1+0.3×(comb_per_ff-1)) pulls SHA256 closer to training range
   - But disrupts ETH/PicoRV which use different logic types
   - Net: marginal at best (coef=0.1 gives 31.4% vs 32.0%)

3. **P_wire_est is only 2-9% of total power** (T1-C):
   - Power delta decomposition doesn't help: the wire component is too small
   - The CTS clock tree dominates, not signal routing

4. **RSMT normalization for WL fails** (T2-E):
   - RSMT of signal nets ≠ clock tree WL (different topologies)
   - Donath sqrt(n_ff×die_area) remains best WL normalizer

5. **RUDY/net features don't help power** (T1-B, T2-F):
   - Correlated with features already present, adds noise for OOD SHA256

---

## Remaining Bottlenecks

| Target | Current | Oracle | Gap | Root Cause |
|--------|---------|--------|-----|------------|
| AES Power | 19.5% (K=20) | 20.2% | ~0% | Oracle-limited: within-design wire cap variation |
| SHA256 Power | 10.5% (K=20) | 10.8% | ~0% | Oracle-limited: SAIF glitch contamination |
| AES WL | 24.9% | 15.2% | 10% | Steiner routing proxy inadequate for dense AES |
| Power ZS | 32.0% | — | — | k_PA OOD for SHA256/PicoRV without calibration |

**The only remaining practical ceiling that can be broken:**
- AES WL (24.9% → 15.2% oracle): needs better Steiner routing proxies or SPEF
- For ICCAD: need more designs (currently only 4) and skew comparison to baselines

---

## How to Run

```bash
cd /home/rain/CTS-Task-Aware-Clustering

# Build caches (one-time, ~5 min)
python3 synthesis_best/build_skew_cache.py
python3 synthesis_best/build_net_cache.py   # optional

# Run full synthesis evaluation
python3 synthesis_best/final_synthesis.py

# Run just skew evaluation
python3 synthesis_best/skew_v2_spatial.py
```

## What ICCAD Still Needs

1. **More designs**: 4 is not enough. Need 8-10+ for credible generalization claim.
2. **Skew baseline comparison**: CTS-Bench gets 0.16 within-design, 0.237+ cross-design.
   Our 0.074 is a major improvement but needs explicit comparison.
3. **AES WL floor**: 24.9% vs 15.2% oracle. Probably requires SPEF or better net routing proxy.
4. **Power zero-shot**: 32% is high for ICCAD claim. K-shot (9.8%) is practically realistic.
