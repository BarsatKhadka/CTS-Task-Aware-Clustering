"""
corrected_sim_exp.py - Corrected CTS simulation using min(cluster_size, FFs_in_radius)

Key physics insight: The CTS tool constrains clusters by BOTH:
  - cluster_size: max FFs per cluster (from knob)
  - cluster_dia: routing budget → limits FFs per cluster to those within radius
    effective_cs = min(cluster_size, λ × π × (cluster_dia/2)²)
    where λ = n_ff / die_area = FF spatial density

This gives rho=0.80 with z_total_clk vs rho=0.14 from naive n_ff/cluster_size.

Also explores:
  1. per-design transition point where cluster_dia vs cluster_size binds
  2. z(effective_cs) as a physics feature
  3. Physics formula + LGB with β_cd-weighted correction
  4. CORRECTED Ripley: per-cluster Z-score (within-run normalization)
"""

import pickle, time, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Corrected CTS Simulation: min(cluster_size, FFs_in_radius)")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/absolute_v5_def_cache.pkl', 'rb') as f:
    def_cache = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(df)
y_pw, y_wl = Y_cache[:, 1], Y_cache[:, 2]

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

# ── Build X29T ─────────────────────────────────────────────────────────────
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

# ── Physics features ───────────────────────────────────────────────────────
print(f"\n{T()} === CORRECTED PHYSICS SIMULATION ===")
n_ff_arr = np.array([def_cache.get(p, {}).get('n_ff', 2500) for p in pids], dtype=float)
die_area_arr = np.array([def_cache.get(p, {}).get('die_area', 400000) for p in pids], dtype=float)
ff_hpwl_arr = np.array([def_cache.get(p, {}).get('ff_hpwl', 1000) for p in pids], dtype=float)

# FF density (FFs per µm²)
lambda_arr = n_ff_arr / die_area_arr

# FFs within cluster_dia radius (uniform distribution model)
n_in_radius = lambda_arr * np.pi * (cd/2)**2
# Effective cluster size
effective_cs = np.minimum(cs, n_in_radius)
effective_cs = np.maximum(effective_cs, 1.0)
# Predicted cluster count
n_clusters_pred = n_ff_arr / effective_cs

# Indicator: which constraint is binding?
cs_limited = cs < n_in_radius  # cluster_size is binding (large cd)
cd_limited = ~cs_limited        # cluster_dia is binding (small cd)

print(f"  {cs_limited.mean():.3f} fraction where cluster_size is binding (large cd)")
print(f"  {cd_limited.mean():.3f} fraction where cluster_dia is binding (small cd)")

# Per-design breakdown
for design in ['aes','ethmac','picorv32','sha256']:
    mask = designs==design
    print(f"    {design}: {cs_limited[mask].mean():.2f} cs-limited, "
          f"{cd_limited[mask].mean():.2f} cd-limited, "
          f"avg n_in_radius={n_in_radius[mask].mean():.1f}")

# Correlation analysis
z_ncl = np.zeros(n); z_log_ncl = np.zeros(n)
n_total = df['clock_buffers'].values + df['clock_inverters'].values
z_ntot = np.zeros(n)

for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = n_clusters_pred[idx]; z_ncl[idx] = (v-v.mean())/max(v.std(),1e-8)
    v2 = np.log(n_clusters_pred[idx]+1); z_log_ncl[idx] = (v2-v2.mean())/max(v2.std(),1e-8)
    v3 = n_total[idx].astype(float); z_ntot[idx] = (v3-v3.mean())/max(v3.std(),1e-8)

rho_ncl_clk, _ = spearmanr(z_ncl, z_ntot)
rho_ncl_pw, _ = spearmanr(z_ncl, y_pw)
rho_logncl_pw, _ = spearmanr(z_log_ncl, y_pw)
print(f"\n  z_nclusters_pred vs z_total_clk: rho={rho_ncl_clk:.4f}")
print(f"  z_nclusters_pred vs z_power: rho={rho_ncl_pw:.4f}")
print(f"  z_log_nclusters_pred vs z_power: rho={rho_logncl_pw:.4f}")

# Transition point feature: (λπ(cd/2)² - cs) / cs → positive when cs binds, negative when cd binds
slack_arr = (n_in_radius - cs) / (cs + 1e-8)  # positive = cs is binding
z_slack = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = slack_arr[idx]; z_slack[idx] = (v-v.mean())/max(v.std(),1e-8)

rho_slack_pw, _ = spearmanr(z_slack, y_pw)
print(f"  z_slack (n_in_rad-cs)/cs vs z_power: rho={rho_slack_pw:.4f}")

# ── Build comprehensive physics feature set ────────────────────────────────
print(f"\n{T()} Building physics feature matrix...")

# Feature 1: z(n_clusters_pred) using corrected formula
# Feature 2: indicator of which constraint is binding
# Feature 3: slack (how far we are from the transition)
# Feature 4: log(n_clusters_pred) z-scored
# Feature 5: n_in_radius normalized by cluster_size
# Feature 6: ff_hpwl / cluster_dia (WL budget ratio)
# Feature 7: ff_hpwl × n_clusters_pred (predicted total routing scale)

z_effcs = np.zeros(n)  # z-score of effective cluster size
z_nir_cs = np.zeros(n)  # n_in_radius / cluster_size
z_hpwl_cd = np.zeros(n)  # ff_hpwl / cluster_dia

for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    for arr, z in [(effective_cs, z_effcs),
                   (n_in_radius/cs, z_nir_cs),
                   (ff_hpwl_arr/cd, z_hpwl_cd)]:
        v = arr[idx]; z[idx] = (v-v.mean())/max(v.std(),1e-8)

# Check correlations
for name, z in [('z_effcs', z_effcs), ('z_nir_cs', z_nir_cs), ('z_hpwl_cd', z_hpwl_cd)]:
    rho, _ = spearmanr(z, y_pw); print(f"  {name} vs z_power: rho={rho:.4f}")

# Build physics feature vector
Xphys = np.column_stack([z_ncl, z_log_ncl, z_effcs, z_nir_cs, z_hpwl_cd, z_slack])
Xphys_std = Xphys.std(0); Xphys_std[Xphys_std<1e-9]=1.0
Xphys_n = Xphys / Xphys_std

# Also: β_cd-aware feature (predicted β_cd × z_cluster_dia)
# β_cd ≈ -0.92 universally, but varies ±0.09 with placement properties
# Use a simple estimate: β_cd ≈ -0.92 + correction(n_in_radius/cs, die_area)
# From prior analysis: β_cd is nearly universal, so use -0.92 as default
cd_z_pp = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = cd[idx]; cd_z_pp[idx] = (v - v.mean()) / max(v.std(), 1e-8)
physics_pw_feat = -0.92 * cd_z_pp  # universal β_cd = -0.92

# ── LODO evaluation ────────────────────────────────────────────────────────
def lodo_lgb(X, y, label, ne=300, nl=20, lr=0.03, mc=15):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=ne, num_leaves=nl, learning_rate=lr,
                          min_child_samples=mc, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={mean_mae:.4f}{tag}")
    return maes

# Build different feature combinations
X_with_phys = np.hstack([X29T, Xphys_n])
X_with_phys_pw = np.hstack([X29T, Xphys_n, physics_pw_feat.reshape(-1,1)])

for arr in [X29T, X_with_phys, X_with_phys_pw]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

print(f"\n{T()} === POWER z-score MAE ===")
lodo_lgb(X29T, y_pw, "X29T baseline (0.2163)")
lodo_lgb(X_with_phys, y_pw, "X29T + phys_sim")
lodo_lgb(X_with_phys_pw, y_pw, "X29T + phys_sim + β_cd×z_cd")

# Ridge with physics formula only (pure physics predictor)
from sklearn.linear_model import Ridge
def lodo_ridge(X, y, label, alpha=1.0):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    print(f"  {label}: {np.mean(maes):.4f}")

print(f"\n  Ridge baselines:")
lodo_ridge(cd_z_pp.reshape(-1,1), y_pw, "β_cd×z_cd_pp Ridge (pure physics)")
lodo_ridge(np.column_stack([cd_z_pp, z_ncl]), y_pw, "β_cd×z_cd + z_nclusters Ridge")
lodo_ridge(np.column_stack([physics_pw_feat, z_ncl, z_effcs]), y_pw, "physics features Ridge")

print(f"\n{T()} === WL z-score MAE ===")
lodo_lgb(X29T, y_wl, "X29T baseline (0.2338)")
lodo_lgb(X_with_phys, y_wl, "X29T + phys_sim")
lodo_lgb(X_with_phys_pw, y_wl, "X29T + phys_sim + β_cd×z_cd")

# ── Hyperparameter search: fewer leaves to prevent overfitting ────────────
print(f"\n{T()} === HYPERPARAMETER SEARCH ===")
print("  Power (trying fewer leaves for better generalization):")
for nl, mc in [(10, 20), (15, 20), (20, 25), (30, 20)]:
    lodo_lgb(X_with_phys_pw, y_pw, f"nl={nl},mc={mc}", nl=nl, mc=mc)

# ── Strategy flags (synth_strategy, io_mode, time_driven) ─────────────────
print(f"\n{T()} === STRATEGY FLAGS AS FEATURES ===")
# One-hot encode strategy flags
strats = df['synth_strategy'].values
td = df['time_driven'].values.astype(float)
rd = df['routability_driven'].values.astype(float)
io = df['io_mode'].values.astype(float)

# Encode synth_strategy
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
strat_enc = le.fit_transform(strats).astype(float)
print(f"  Strategy classes: {le.classes_}")

X_strat = np.column_stack([strat_enc, td, rd, io])
X_aug = np.hstack([X_with_phys_pw, X_strat])
for c in range(X_aug.shape[1]):
    bad = ~np.isfinite(X_aug[:,c])
    if bad.any(): X_aug[bad,c] = 0.0

print("  Power:")
lodo_lgb(X_aug, y_pw, "X29T+phys+strategy")
print("  WL:")
lodo_lgb(X_aug, y_wl, "X29T+phys+strategy")

# ── Per-design β_cd calibration experiment ────────────────────────────────
print(f"\n{T()} === PER-DESIGN β_cd CALIBRATION (LODO with universal β_cd) ===")
# Use universal β_cd = -0.92 to predict z_power, then compare to current model
print("  Prediction: z_power ≈ -0.92 × z_cluster_dia (universal physics formula)")
maes = []
for held in sorted(np.unique(designs)):
    te = designs==held
    pred = physics_pw_feat[te]  # = -0.92 × z_cd_pp
    maes.append(mean_absolute_error(y_pw[te], pred))
print(f"  Universal physics (β_cd=-0.92 × z_cd): "
      f"{maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={np.mean(maes):.4f}")

# Two-term: β_cd × z_cd + β_cs × z_cs
z_cs_pp = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = cs[idx]; z_cs_pp[idx] = (v-v.mean())/max(v.std(),1e-8)

# Learn β_cs from training designs only
for held in sorted(np.unique(designs)):
    tr = designs!=held; te = designs==held
    X_2term = np.column_stack([cd_z_pp[tr], z_cs_pp[tr]])
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(X_2term, y_pw[tr], rcond=None)
    pred = np.column_stack([cd_z_pp[te], z_cs_pp[te]]) @ coeffs
    mae = mean_absolute_error(y_pw[te], pred)

maes_2t = []
for held in sorted(np.unique(designs)):
    tr = designs!=held; te = designs==held
    X_2term = np.column_stack([cd_z_pp[tr], z_cs_pp[tr],
                                z_ncl[tr], z_effcs[tr]])
    from numpy.linalg import lstsq
    coeffs, _, _, _ = lstsq(X_2term, y_pw[tr], rcond=None)
    pred = np.column_stack([cd_z_pp[te], z_cs_pp[te],
                             z_ncl[te], z_effcs[te]]) @ coeffs
    maes_2t.append(mean_absolute_error(y_pw[te], pred))
print(f"  Physics Ridge (4 phys terms): "
      f"{maes_2t[0]:.4f}/{maes_2t[1]:.4f}/{maes_2t[2]:.4f}/{maes_2t[3]:.4f} mean={np.mean(maes_2t):.4f}")

print(f"\n{T()} DONE")
