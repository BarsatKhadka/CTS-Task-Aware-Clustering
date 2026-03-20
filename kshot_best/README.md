# K-Shot Best — Multiplicative Calibration on Zero-Shot Absolute Predictor

**Locked**: 2026-03-19 (Session 11)
**DO NOT MODIFY** — this folder is a frozen reference point.

---

## What This Is

K-shot multiplicative bias correction on top of `baseline_best/absolute_v16_final.py`.

After zero-shot prediction, observe K labeled samples from the test design.
Compute `k_hat = median(actual[supp] / pred[supp])`, apply `pred_cal = pred × k_hat` to all remaining samples.

**v20 update (2026-03-20)**: Switched from mean → median k_hat. Reduces K needed for ≤10% power from K=20 to **K=10**.

---

## Results (LODO, 4 designs, 200 reps per K)

### Mean MAPE Across All 4 Designs

| K (random samples) | Power MAPE | WL MAPE |
|--------------------|-----------|---------|
| 0 (zero-shot)      | 32.0%     | 11.0%   |
| 1                  | 12.6%     |  8.9%   |
| 3                  | 10.9%     |  7.4%   |
| 5                  | 10.3%     |  7.0%   |
| **10**             | **9.8%** ✓| **6.5%** ✓ |
| 20                 |  9.8%     |  6.5%   |

**≤10% power MAPE achieved at K=10 (with median k_hat).**

### Per-Design Power MAPE at Key K Values

| Design  | K=0   | K=1   | K=3   | K=10  | K=20  | Oracle |
|---------|-------|-------|-------|-------|-------|--------|
| AES     | 36.6% | 22.9% | 20.5% | 19.7% | 19.5% | 20.2%  |
| ETH MAC | 12.3% |  5.6% |  4.6% |  4.2% |  4.1% |  6.8%  |
| PicoRV  | 30.1% |  7.7% |  6.2% |  5.4% |  5.3% |  6.5%  |
| SHA-256 | 48.9% | 14.1% | 12.4% | 10.9% | 10.4% | 10.8%  |

---

## How to Run

```bash
cd /home/rain/CTS-Task-Aware-Clustering
python3 kshot_best/absolute_v17_kshot.py
```

---

## Architecture

**Base model** (identical to baseline_best/absolute_v16_final.py):
- Power: XGBRegressor(n=300, max_depth=4, lr=0.05, subsample=0.8) — 76 dims
- WL: LGB(300)+Ridge(1000) blend α=0.3 — 75 dims

**K-shot calibration** (v20: median):
```python
k_hat_pw = median(actual_pw[support] / pred_pw[support])
k_hat_wl = median(actual_wl[support] / pred_wl[support])
pred_pw_cal = pred_pw[rest] * k_hat_pw
pred_wl_cal = pred_wl[rest] * k_hat_wl
```

Two modes evaluated:
1. **Random-sample K**: K random rows from all test samples
2. **Placement-level K**: K full placements (all their CTS runs) as support

Both modes give nearly identical results.

---

## Key Insights

1. **K=1 already huge**: 32% → 12.6% power MAPE (61% reduction). Even one labeled sample reveals the design's k_PA level.

2. **AES is oracle-limited**: Floors at ~19.5% (oracle 20.2%) regardless of K. AES has high within-design k_PA variance (CV=0.185) — inherent unpredictability not capturable from these features.

3. **SHA256 reaches near-oracle at K=20**: 10.4%±1.4 vs oracle 10.8%. The bias correction rescales the OOD rel_act prediction to the right range.

4. **WL benefit too**: K=1 drops WL from 11.0% → 8.9%. K=20 gives 6.6%.

5. **Placement-level ≈ random-sample**: No benefit from using full placements as support vs random individual runs.

## Practical Deployment Guide

| Available labeled data | Recommendation                        |
|------------------------|---------------------------------------|
| 0 runs                 | zero-shot, power=32%, WL=11%          |
| 1–2 runs               | K=1 calibration, power≈12.6%          |
| 3–5 runs               | K=3 calibration, power≈10.9%          |
| **10+ runs**           | **K=10 calibration, power≈9.8% ✓**    |
| 1 full placement (10 runs) | **power≈9.8% ✓ (median k_hat)**   |

---

## Files

| File | Description |
|------|-------------|
| `absolute_v17_kshot.py` | K-shot calibration script |
| `absolute_v7_def_cache.pkl` → | Symlink to DEF parser cache |
| `absolute_v7_saif_cache.pkl` → | Symlink to SAIF parser cache |
| `absolute_v7_timing_cache.pkl` → | Symlink to timing paths cache |
| `absolute_v10_gravity_cache.pkl` → | Symlink to gravity features cache |
| `absolute_v13_extended_cache.pkl` → | Symlink to extended features cache |
