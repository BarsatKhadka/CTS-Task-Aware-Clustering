# Baseline Best — Zero-Shot Absolute Power+WL Predictor

**Locked**: 2026-03-19 (Session 10)
**DO NOT MODIFY** — this folder is a frozen reference point.

---

## Results (LODO, 4 designs)

| Design  | Power MAPE | WL MAPE | Power Oracle | WL Oracle |
|---------|-----------|---------|--------------|-----------|
| AES     | 36.6%     | 24.9%   | 20.2%        | 15.2%     |
| ETH MAC | 12.3%     |  8.2%   |  6.8%        |  8.8%     |
| PicoRV  | 30.1%     |  5.7%   |  6.5%        |  6.4%     |
| SHA-256 | 48.9%     |  5.1%   | 10.8%        |  8.1%     |
| **MEAN**| **32.0%** | **11.0%**|             |           |

## Architecture

**Power model**: XGBRegressor(n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8)
- Features: base_pw(58d, WITH synth sd/sl/sa) + timing(18d) = **76 dims**
- Normalization: `n_ff × f_ghz × avg_ds`

**WL model**: LGB(300) + Ridge(1000) blend, α=0.3
- Features: base_wl(53d, NO synth) + gravity(19d) + extra_scale(3d) = **75 dims**
- Normalization: `sqrt(n_ff × die_area)`

## Files

| File | Description |
|------|-------------|
| `absolute_v16_final.py` | Main predictor script |
| `absolute_v7_def_cache.pkl` | DEF parser cache (539 entries) |
| `absolute_v7_saif_cache.pkl` | SAIF parser cache (539 entries) |
| `absolute_v7_timing_cache.pkl` | Timing paths cache (539 entries) |
| `absolute_v10_gravity_cache.pkl` | Gravity vector features (541 entries) |
| `absolute_v13_extended_cache.pkl` | Extended features: driven_cap, MST, density (541 entries) |

## How to Run

```bash
cd /home/rain/CTS-Task-Aware-Clustering
python3 baseline_best/absolute_v16_final.py
```

The script auto-discovers all caches via hardcoded paths (absolute_v7_*, absolute_v10_*, absolute_v13_*).

## Key Engineering Decisions

1. **Synth features in power base**: WITHOUT synth (sd, sl, sa), SHA256 degrades from 48.9% → 66%
2. **XGB over LGB for power**: XGB max_depth=4 generalizes better across design styles
3. **NO synth in WL base**: WL generalizes better to OOD (zipdiv) without synth fingerprints
4. **extra_scale WL-only**: `log(area/ff)`, `log(n_comb)`, `comb_per_ff×log(n_ff)` — help WL, hurt power
5. **WL α=0.3**: LGB+Ridge blend. Use α=0.0 (Ridge-only) for truly OOD designs like zipdiv

## Known Ceilings

- **SHA256 power 48.9%** is the practical zero-shot floor: SHA256 rel_act=0.104 is 2× training max → tree OOD
- **AES power 36.6%** floor: oracle 20.2%; AES has highest k_PA variance (CV=0.185)
- **AES WL 24.9%** floor: oracle 15.2%; needs better Steiner routing proxies
- All ceilings require SPEF data or 1-shot calibration to break

## What NOT to Do

- Do NOT add driven_cap_per_ff / fanout_proxy / ra_corrected to power features → hurts SHA256 (57-66%)
- Do NOT use rel_act in power normalization denominator → amplifies SHA256 glitch contamination
- Do NOT use phys_pw = rel_act × n_nets × f as normalization → SHA256 becomes 349%+
- Do NOT switch to LGB for power without synth → SHA256 degrades to 66%
