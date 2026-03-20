"""
Example 03 — K-shot calibration on an unseen design (zipdiv)
==============================================================
Demonstrates how multiplicative K-shot calibration corrects the scale
mismatch when applying the surrogate to a design outside the training
size regime (zipdiv: 142 FFs vs training 1597-5000 FFs).

Zero-shot:  Power 55.5%, WL 99.8%  (normalizers extrapolate badly)
K=1 shot:   Power  ~5%,  WL  ~4%   (scale corrected from 1 CTS run)

Run:
    python3 examples/03_kshot_calibration.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cts_surrogate import CTSSurrogate
import pandas as pd
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = CTSSurrogate.from_package()

# ── Load zipdiv ground truth ───────────────────────────────────────────────────
gt = pd.read_csv(os.path.join(PKG, 'data', 'zipdiv_gt.csv'))
gt['power_mW'] = gt['power_total'] * 1000.0   # W → mW
gt['wl_mm']    = gt['wirelength']  / 1000.0   # µm → mm
gt = gt[gt['design_name'] == 'zipdiv'].copy()

print(f"Zipdiv ground truth: {len(gt)} CTS runs, "
      f"{gt['placement_id'].nunique()} placements")
print(f"  power: {gt['power_mW'].min():.3f}–{gt['power_mW'].max():.3f} mW")
print(f"  WL:    {gt['wl_mm'].min():.2f}–{gt['wl_mm'].max():.2f} mm")
print(f"  skew:  {gt['skew_setup'].min():.4f}–{gt['skew_setup'].max():.4f} ns")
print()

# ── Get raw (uncalibrated) predictions for all 20 runs ────────────────────────
# Compute per-placement skew stats for absolute ns conversion
sk_stats = {pid: (grp['skew_setup'].mean(),
                  max(grp['skew_setup'].std(), 1e-4))
            for pid, grp in gt.groupby('placement_id')}

preds = []
for _, row in gt.iterrows():
    pid  = row['placement_id']
    sk_mu, sk_sig = sk_stats[pid]
    pred = model.predict(pid,
                         cd=int(row['cts_cluster_dia']),
                         cs=int(row['cts_cluster_size']),
                         mw=int(row['cts_max_wire']),
                         bd=int(row['cts_buf_dist']),
                         sk_mu=sk_mu, sk_sig=sk_sig)
    preds.append(dict(
        true_pw=row['power_mW'], pred_pw=pred.power_mW,
        true_wl=row['wl_mm'],    pred_wl=pred.wl_mm,
        true_sk=row['skew_setup'],
        pred_sk=pred.skew_ns if pred.skew_ns is not None
                else sk_mu + pred.skew_z * sk_sig,
    ))

df = pd.DataFrame(preds)

def mape(t, p): return np.mean(np.abs(t-p)/np.abs(t))*100
def mae(t, p):  return np.mean(np.abs(t-p))

pw0 = mape(df['true_pw'].values, df['pred_pw'].values)
wl0 = mape(df['true_wl'].values, df['pred_wl'].values)
sk0 = mae(df['true_sk'].values, df['pred_sk'].values)

print(f"{'─'*55}")
print(f"ZERO-SHOT (K=0):")
print(f"  Power MAPE:  {pw0:.1f}%  ← normalizer 2.3× off (142 vs 1597-5000 FFs)")
print(f"  WL MAPE:     {wl0:.1f}%  ← normalizer 605× off (extreme OOD size)")
print(f"  Skew MAE:    {sk0*1000:.2f} ps  ← already excellent (0.77% of true)")
print()

# ── K-shot multiplicative calibration ─────────────────────────────────────────
# k_hat = median(actual / pred) on K support samples
# Then: calibrated = k_hat × raw_pred
print(f"K-SHOT CALIBRATION  (k_hat = median(actual/pred) over K samples)")
print(f"{'─'*55}")
print(f"{'K':>4}  {'Power MAPE':>12}  {'WL MAPE':>10}  {'Skew MAE':>10}")
print(f"{'0':>4}  {pw0:>10.1f}%    {wl0:>8.1f}%    {sk0*1000:>7.2f} ps")

np.random.seed(42)
N_TRIALS = 300
for K in [1, 2, 3, 5, 10, 20]:
    pw_trials, wl_trials = [], []
    for _ in range(N_TRIALS if K < 20 else 1):
        idx = np.random.choice(len(df), size=K, replace=False)
        sup = df.iloc[idx]
        k_pw = np.median(sup['true_pw'].values / sup['pred_pw'].values)
        k_wl = np.median(sup['true_wl'].values / sup['pred_wl'].values)
        pw_trials.append(mape(df['true_pw'].values, df['pred_pw'].values * k_pw))
        wl_trials.append(mape(df['true_wl'].values, df['pred_wl'].values * k_wl))

    pm, ps = np.mean(pw_trials), np.std(pw_trials)
    wm, ws = np.mean(wl_trials), np.std(wl_trials)
    print(f"{K:>4}  {pm:>9.1f}%±{ps:>3.1f}   {wm:>7.1f}%±{ws:>3.1f}   "
          f"{sk0*1000:>7.2f} ps  (unchanged)")

# ── Oracle calibration with all 20 ────────────────────────────────────────────
k20_pw = np.median(df['true_pw'].values / df['pred_pw'].values)
k20_wl = np.median(df['true_wl'].values / df['pred_wl'].values)
print(f"\nOracle scale factors (K=20):")
print(f"  k_pw = {k20_pw:.3f}  (model underestimates by 2.3×)")
print(f"  k_wl = {k20_wl:.3f}  (model underestimates by 605×)")
print(f"\nPhysical explanation:")
print(f"  The power normalizer (n_ff × f_GHz × avg_ds) and WL normalizer")
print(f"  (√(n_ff × die_area)) are calibrated to 1597-5000 FF designs.")
print(f"  Zipdiv (142 FFs, 215×165µm) is outside that regime.")
print(f"  K=1 CTS run (10-30 min EDA time) corrects both to <5% error.")
