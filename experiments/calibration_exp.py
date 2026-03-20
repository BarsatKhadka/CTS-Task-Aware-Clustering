"""
calibration_exp.py - Few-run placement-level calibration

Key finding: z_power ≈ β_cd × z_cluster_dia where β_cd = -0.92 ± 0.09 is
nearly universal. The ±0.09 variance in β_cd limits direct z-score MAE to ~0.15+.

Solution: If we run just 2 CTS configs for a new placement, we can estimate
the placement-specific β_cd and extrapolate to remaining runs.

This is the practical "few-shot" scenario:
  - Production: run 2+ configs → calibrate → predict best config without exploring all 10

Also tests:
  - How good is 1-run / 2-run / 3-run calibration?
  - Does it actually break the 0.10 MAE barrier?
  - For zero-shot: can β_cd be predicted from circuit features alone?
"""

import pickle, time, warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Placement-Level Calibration Experiment")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

# ── Build X29T for ML baseline ─────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xraw = df[knob_cols].values.astype(np.float32)
Xkz = X_cache[:, 72:76]
raw_max = Xraw.max(0) + 1e-6
Xrank = np.zeros((n, 4), np.float32); Xcent = np.zeros_like(Xrank)
Xrng = np.zeros_like(Xrank); Xmn = np.zeros_like(Xrank)
for pid in np.unique(pids):
    m = pids == pid; idx = np.where(m)[0]
    for j in range(4):
        v = Xraw[idx, j]; Xrank[idx, j] = rank_within(v)
        Xcent[idx, j] = (v - v.mean()) / raw_max[j]
        Xrng[idx, j] = v.std() / raw_max[j]; Xmn[idx, j] = v.mean() / raw_max[j]
Xplc = df[['core_util', 'density', 'aspect_ratio']].values.astype(np.float32)
Xplc_n = Xplc / (Xplc.max(0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
util = Xplc[:, 0]/100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp,
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T = np.hstack([X29, X_tight/tp_std])

y_pw = Y_cache[:, 1]; y_wl = Y_cache[:, 2]

# ── Zero-shot baselines ────────────────────────────────────────────────────
print(f"\n{T()} === ZERO-SHOT BASELINES (no calibration runs) ===")

def lodo_zero_shot(X, y, label):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        maes.append(mean_absolute_error(y[te], pred))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}")
    return np.mean(maes)

lodo_zero_shot(X29, y_pw, "X29 LGB (power)")
lodo_zero_shot(X29, y_wl, "X29 LGB (WL)")
lodo_zero_shot(X29T, y_pw, "X29T LGB (power)")
lodo_zero_shot(X29T, y_wl, "X29T LGB (WL)")

# ── Few-run placement-level calibration ───────────────────────────────────
print(f"\n{T()} === FEW-RUN PLACEMENT CALIBRATION ===")
print("Strategy: run n_calib CTS configs per NEW placement, fit per-placement slope,")
print("predict remaining runs using calibrated slope.")
print()

def placement_calibrate(X_base, y, n_calib, design_level=False):
    """
    For each held-out design (LODO):
    1. Train base model on training designs
    2. For each test placement: use first n_calib runs to estimate per-placement slope
    3. Combine base prediction + calibrated slope for remaining runs
    """
    dl = sorted(np.unique(designs)); all_maes = []

    for held in dl:
        tr = designs!=held; te = designs==held
        sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X_base[tr]), y[tr])

        # Process each test placement individually
        test_pids = np.unique(pids[te]); placement_maes = []

        for pid in test_pids:
            idx = np.where((pids==pid) & te)[0]
            if len(idx) < n_calib + 1: continue

            # Get base model predictions for ALL runs of this placement
            pred_all = m.predict(sc.transform(X_base[idx]))
            y_all = y[idx]

            # Calibration runs: first n_calib (in original order)
            calib_idx = np.arange(n_calib)
            test_idx = np.arange(n_calib, len(idx))

            if n_calib == 0:
                # Zero-shot: use base predictions directly
                placement_maes.append(mean_absolute_error(y_all, pred_all))
                continue

            # Compute per-placement bias from calibration runs
            bias = (y_all[calib_idx] - pred_all[calib_idx]).mean()

            # Also try slope calibration using cluster_dia
            cd_all = Xraw[idx, 3]  # raw cluster_dia
            cd_z = (cd_all - cd_all.mean()) / max(cd_all.std(), 1e-8)  # per-placement z-scored

            if n_calib >= 2:
                # Fit: correction = α + β × cd_z
                calib_X = np.column_stack([cd_z[calib_idx], np.ones(n_calib)])
                resid_calib = y_all[calib_idx] - pred_all[calib_idx]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(calib_X, resid_calib, rcond=None)
                    correction = np.column_stack([cd_z[test_idx], np.ones(len(test_idx))]) @ coeffs
                    pred_corrected = pred_all[test_idx] + correction
                except:
                    pred_corrected = pred_all[test_idx] + bias
            else:
                # 1-run: just bias correction
                pred_corrected = pred_all[test_idx] + bias

            placement_maes.append(mean_absolute_error(y_all[test_idx], pred_corrected))

        all_maes.append(np.mean(placement_maes))

    return all_maes

print("Power (per-placement z-score MAE):")
for nc in [0, 1, 2, 3, 5]:
    maes = placement_calibrate(X29T, y_pw, nc)
    tag = ' ✓' if np.mean(maes) < 0.10 else (' ✓✓' if np.mean(maes) < 0.05 else '')
    print(f"  n_calib={nc}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  "
          f"mean={np.mean(maes):.4f}{tag}")

print("\nWirelength (per-placement z-score MAE):")
for nc in [0, 1, 2, 3, 5]:
    maes = placement_calibrate(X29T, y_wl, nc)
    tag = ' ✓' if np.mean(maes) < 0.10 else (' ✓✓' if np.mean(maes) < 0.05 else '')
    print(f"  n_calib={nc}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  "
          f"mean={np.mean(maes):.4f}{tag}")

# ── Upper bound: in-distribution calibration ─────────────────────────────
print(f"\n{T()} === UPPER BOUND (oracle β_cd per placement) ===")
print("What if we perfectly know β_cd for each placement?")

for target_name, y in [('Power', y_pw), ('WL', y_wl)]:
    maes = []
    for held in sorted(np.unique(designs)):
        te = designs==held; tr = designs!=held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X29T[tr]), y[tr])

        test_pids = np.unique(pids[te]); placement_maes = []
        for pid in test_pids:
            idx = np.where((pids==pid) & te)[0]
            if len(idx) < 3: continue
            pred_all = m.predict(sc.transform(X29T[idx]))
            y_all = y[idx]
            cd_z = (Xraw[idx,3] - Xraw[idx,3].mean()) / max(Xraw[idx,3].std(), 1e-8)

            # Oracle: fit correction using ALL runs (upper bound)
            full_X = np.column_stack([cd_z, np.ones(len(idx))])
            resid = y_all - pred_all
            try:
                coeffs, _, _, _ = np.linalg.lstsq(full_X, resid, rcond=None)
                correction = full_X @ coeffs
                pred_corrected = pred_all + correction
            except:
                pred_corrected = pred_all
            placement_maes.append(mean_absolute_error(y_all, pred_corrected))

        maes.append(np.mean(placement_maes))
    print(f"  {target_name} (oracle β_cd): {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} "
          f"mean={np.mean(maes):.4f}")

# ── Per-placement β_cd prediction ─────────────────────────────────────────
print(f"\n{T()} === PREDICTING β_cd FROM CIRCUIT FEATURES ===")
print("Can we predict the per-placement β_cd zero-shot?")

# Compute per-placement β_cd for all placements
beta_cds = {}
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    if len(idx) < 3: continue
    cd_all = Xraw[idx, 3]
    cd_z = (cd_all - cd_all.mean()) / max(cd_all.std(), 1e-8)
    if cd_z.std() < 1e-6: continue

    # Fit β_cd for this placement
    for target_name, y in [('pw', y_pw), ('wl', y_wl)]:
        A = np.column_stack([cd_z, np.ones(len(idx))])
        coeffs, _, _, _ = np.linalg.lstsq(A, y[idx], rcond=None)
        if pid not in beta_cds: beta_cds[pid] = {}
        beta_cds[pid][target_name] = coeffs[0]

n_beta = len(beta_cds)
print(f"  Computed β_cd for {n_beta} placements")

# What context features predict β_cd?
# Use X_cache constant features (cols 0-67) as predictors of β_cd
beta_arr_pw = np.array([beta_cds[pid]['pw'] for pid in np.unique(pids) if pid in beta_cds])
beta_arr_wl = np.array([beta_cds[pid]['wl'] for pid in np.unique(pids) if pid in beta_cds])
pid_with_beta = [pid for pid in np.unique(pids) if pid in beta_cds]
design_with_beta = np.array([designs[pids==pid][0] for pid in pid_with_beta])

from scipy.stats import spearmanr
print("\n  Top correlates of β_cd_power with X_cache cols 0-67:")
# Get one X_cache row per pid
Xcache_per_pid = np.array([X_cache[pids==pid][0, :68] for pid in pid_with_beta])
corrs = []
for c in range(68):
    rho, _ = spearmanr(Xcache_per_pid[:, c], beta_arr_pw)
    corrs.append((abs(rho), rho, c))
corrs.sort(reverse=True)
for _, rho, c in corrs[:8]:
    print(f"    col{c}: rho={rho:.4f}")

# LODO to predict β_cd
print("\n  LODO MAE for predicting β_cd_power:")
dl = sorted(np.unique(design_with_beta)); maes_beta = []
for held in dl:
    tr = design_with_beta!=held; te = design_with_beta==held
    sc = StandardScaler()
    m = LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                      min_child_samples=10, verbose=-1)
    m.fit(sc.fit_transform(Xcache_per_pid[tr]), beta_arr_pw[tr])
    pred_beta = m.predict(sc.transform(Xcache_per_pid[te]))
    maes_beta.append(mean_absolute_error(beta_arr_pw[te], pred_beta))
print(f"    {maes_beta[0]:.4f}/{maes_beta[1]:.4f}/{maes_beta[2]:.4f}/{maes_beta[3]:.4f}  "
      f"mean={np.mean(maes_beta):.4f}  (std β_cd = {beta_arr_pw.std():.4f})")

print(f"\n{T()} DONE")
