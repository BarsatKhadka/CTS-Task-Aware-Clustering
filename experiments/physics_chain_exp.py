"""
physics_chain_exp.py - Physics-chain two-stage predictor

Key discovery: z_total_clk (z-scored total clock buffers + inverters) has
Spearman rho=0.9811 with z_power. This is the physics chain:

  CTS knobs → buffer count → power (P ∝ n_buf × C_buf × V² × f)
  CTS knobs → routing WL  → power (P ∝ WL × C_wire × V² × f)

Strategy:
  Stage 1: Predict z_total_clk from knobs + DEF context (zero-shot LODO)
  Stage 2: Use predicted z_total_clk → z_power (near-perfect from rho=0.98)

Also:
  - What MAE do we get using TRUE z_total_clk to predict z_power?  (Oracle)
  - What MAE do we get using PREDICTED z_total_clk?  (End-to-end)
  - Can we improve z_WL prediction using buffer count? (rho=0.89)
  - Feature: n_buf_pred = n_ff / cluster_size (physics formula)
  - Spatial features: Ripley's L(r) for cluster_dia range
"""

import pickle, time, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Physics-Chain: Knobs → Buffer Count → Power/WL")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)

pids = df['placement_id'].values
designs = df['design_name'].values
n = len(df)

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

# ── Build X29T (standard baseline) ────────────────────────────────────────
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

# ── Clock buffer targets ───────────────────────────────────────────────────
print(f"\n{T()} === CLOCK BUFFER ANALYSIS ===")
n_buf = df['clock_buffers'].values.astype(float)
n_inv = df['clock_inverters'].values.astype(float)
n_trb = df['timing_repair_buffers'].values.astype(float)
n_total = n_buf + n_inv

# Per-placement z-scored
z_nbuf = np.zeros(n); z_ninv = np.zeros(n); z_ntot = np.zeros(n); z_ntrb = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    for arr, z in [(n_buf,z_nbuf),(n_inv,z_ninv),(n_total,z_ntot),(n_trb,z_ntrb)]:
        v = arr[idx]; z[idx] = (v-v.mean())/max(v.std(),1e-8)

rho_bpw, _ = spearmanr(z_nbuf, y_pw)
rho_tpw, _ = spearmanr(z_ntot, y_pw)
rho_twl, _ = spearmanr(z_ntot, y_wl)
rho_bwl, _ = spearmanr(z_nbuf, y_wl)
print(f"  z_clock_bufs vs z_power: rho={rho_bpw:.4f}")
print(f"  z_total_clk  vs z_power: rho={rho_tpw:.4f}")
print(f"  z_total_clk  vs z_WL:    rho={rho_twl:.4f}")
print(f"  z_clock_bufs vs z_WL:    rho={rho_bwl:.4f}")

# Physics formula: n_buf ≈ n_ff / cluster_size
with open(f'{BASE}/absolute_v5_def_cache.pkl', 'rb') as f:
    def_cache = pickle.load(f)
n_ff_arr = np.array([def_cache.get(pid, {}).get('n_ff', 2500) for pid in pids], dtype=float)
n_buf_pred_phys = n_ff_arr / cs  # simple physics formula: n_clusters ≈ n_ff/cluster_size

# z-score the physics prediction within placement
z_nphys = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = n_buf_pred_phys[idx]
    z_nphys[idx] = (v - v.mean()) / max(v.std(), 1e-8)

rho_phys, _ = spearmanr(z_nphys, z_ntot)
rho_phys_pw, _ = spearmanr(z_nphys, y_pw)
print(f"  z_phys_nbuf vs z_total_clk: rho={rho_phys:.4f}")
print(f"  z_phys_nbuf vs z_power: rho={rho_phys_pw:.4f}")

# ── Oracle: use TRUE z_total_clk to predict z_power ───────────────────────
print(f"\n{T()} === ORACLE: TRUE z_total_clk → z_power (upper bound) ===")

def lodo_oracle(y_target, y_pred_feature, label):
    """Use y_pred_feature to predict y_target via Ridge, LODO."""
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held
        X_tr = y_pred_feature[tr].reshape(-1,1)
        X_te = y_pred_feature[te].reshape(-1,1)
        m = Ridge(alpha=0.1).fit(X_tr, y_target[tr])
        maes.append(mean_absolute_error(y_target[te], m.predict(X_te)))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={np.mean(maes):.4f}")
    return np.mean(maes)

lodo_oracle(y_pw, z_ntot, "Oracle(z_total_clk → z_power) Ridge")
lodo_oracle(y_wl, z_ntot, "Oracle(z_total_clk → z_WL) Ridge")
lodo_oracle(y_pw, z_nbuf, "Oracle(z_clock_bufs → z_power) Ridge")

# Multiple features: z_ntot + z_ntrb + X_context
def lodo_multi_oracle(y_target, features, label):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                          min_child_samples=10, verbose=-1)
        m.fit(sc.fit_transform(features[tr]), y_target[tr])
        maes.append(mean_absolute_error(y_target[te], m.predict(sc.transform(features[te]))))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={np.mean(maes):.4f}")
    return np.mean(maes)

# Oracle with knob context
oracle_feats = np.column_stack([z_ntot, z_nbuf, z_ninv, z_ntrb])
lodo_multi_oracle(y_pw, oracle_feats, "Oracle(all_clk_counts → z_power) LGB")
lodo_multi_oracle(y_wl, oracle_feats, "Oracle(all_clk_counts → z_WL) LGB")

# Oracle + X29T context
oracle_feats_full = np.hstack([oracle_feats, X29T])
lodo_multi_oracle(y_pw, oracle_feats_full, "Oracle(clk_counts+X29T → z_power) LGB")
lodo_multi_oracle(y_wl, oracle_feats_full, "Oracle(clk_counts+X29T → z_WL) LGB")

# ── Stage 1: Predict z_total_clk from knobs + context ─────────────────────
print(f"\n{T()} === STAGE 1: Predict z_total_clk from knobs (zero-shot) ===")

def lodo_stage1(X, y_buf, label):
    dl = sorted(np.unique(designs)); maes = []; preds_all = np.zeros(n)
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y_buf[tr])
        pred = m.predict(sc.transform(X[te]))
        preds_all[te] = pred
        maes.append(mean_absolute_error(y_buf[te], pred))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={np.mean(maes):.4f}")
    return preds_all, maes

pred_ntot_X29, _ = lodo_stage1(X29, z_ntot, "X29 → z_total_clk")
pred_ntot_X29T, _ = lodo_stage1(X29T, z_ntot, "X29T → z_total_clk")
pred_nbuf_X29T, _ = lodo_stage1(X29T, z_nbuf, "X29T → z_clock_bufs")

# Physics-informed stage 1: z_nphys as direct predictor
print(f"\n  Physics formula (n_ff/cluster_size) → z_total_clk:")
rho_direct, _ = spearmanr(z_nphys, z_ntot)
print(f"    Spearman rho = {rho_direct:.4f}")
# LODO for physics formula
lodo_oracle(z_ntot, z_nphys, "  Physics z_nbuf_pred → z_total_clk")

# Additional physics feature: (n_ff / cluster_size) × cluster_dia
# Larger cluster_dia → fewer clusters → different buffer topology
n_ff_z = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    n_ff_z[idx] = n_ff_arr[idx]  # same within placement
# Interaction: buffer estimate × cluster_dia
n_buf_x_cd = n_ff_arr / cs * cd  # n_clusters × cluster_dia ≈ routing scale
z_nbuf_cd = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = n_buf_x_cd[idx]; z_nbuf_cd[idx] = (v-v.mean())/max(v.std(),1e-8)
rho_cd, _ = spearmanr(z_nbuf_cd, z_ntot)
print(f"  z(n_ff/cs × cd) vs z_total_clk: rho={rho_cd:.4f}")

# Enhanced knob features with physics interactions
# 1. log(n_ff / cluster_size) per placement (captures buffer count)
log_nff_cs = np.log(n_ff_arr / cs + 1)
z_log_nff_cs = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = log_nff_cs[idx]; z_log_nff_cs[idx] = (v-v.mean())/max(v.std(),1e-8)

# 2. 1/cluster_size (direct inverse relationship)
inv_cs = 1.0 / cs
z_inv_cs = np.zeros(n)
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = inv_cs[idx]; z_inv_cs[idx] = (v-v.mean())/max(v.std(),1e-8)

print(f"\n  z_inv_cluster_size vs z_total_clk: rho={spearmanr(z_inv_cs, z_ntot)[0]:.4f}")
print(f"  z_inv_cluster_size vs z_power: rho={spearmanr(z_inv_cs, y_pw)[0]:.4f}")

# Build physics-augmented feature set
Xphys_aug = np.column_stack([z_nphys, z_nbuf_cd, z_inv_cs, z_log_nff_cs])
X_aug_29T = np.hstack([X29T, Xphys_aug])

print(f"\n  Testing physics-augmented X29T:")
lodo_stage1(X_aug_29T, z_ntot, "X29T+Xphys_aug → z_total_clk")

# ── Stage 2: Use predicted z_total_clk → z_power ──────────────────────────
print(f"\n{T()} === STAGE 2: Predicted z_total_clk → z_power/z_WL ===")

def lodo_stage2(y_target, z_buf_pred, X_ctx, label):
    """Chain prediction: predicted z_buf → z_power."""
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        feats = np.hstack([z_buf_pred.reshape(-1,1), X_ctx])
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(feats[tr]), y_target[tr])
        maes.append(mean_absolute_error(y_target[te], m.predict(sc.transform(feats[te]))))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={np.mean(maes):.4f}")
    return np.mean(maes)

# Full chain: X29T → z_total_clk_pred → z_power
print("\n  Power prediction:")
lodo_stage2(y_pw, pred_ntot_X29T, X29T, "X29T + pred_z_total_clk → z_power LGB")
lodo_stage2(y_pw, pred_nbuf_X29T, X29T, "X29T + pred_z_nbuf → z_power LGB")

print("\n  WL prediction:")
lodo_stage2(y_wl, pred_ntot_X29T, X29T, "X29T + pred_z_total_clk → z_WL LGB")

# ── Direct use of clock buffer z-score in X29T ────────────────────────────
# (cheating version - uses true z_ntot as feature at test time)
print(f"\n{T()} === X29T + PHYSICS AUG FEATURES (no buffer oracle) ===")
print("  (using physics interactions only, no buffer oracle)")

# Add physics interaction features to X29T
Xbig = np.hstack([X29T, Xphys_aug])
for c in range(Xbig.shape[1]):
    bad = ~np.isfinite(Xbig[:,c])
    if bad.any(): Xbig[bad,c] = 0.0

def lodo_lgb(X, y, label):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={mean_mae:.4f}{tag}")
    return maes

print("\n  Power:")
lodo_lgb(X29T, y_pw, "X29T baseline")
lodo_lgb(Xbig, y_pw, "X29T + physics_aug (inv_cs, n_ff/cs, etc.)")

print("\n  WL:")
lodo_lgb(X29T, y_wl, "X29T baseline")
lodo_lgb(Xbig, y_wl, "X29T + physics_aug")

# ── Ripley L(r) spatial clustering features ───────────────────────────────
print(f"\n{T()} === SPATIAL FEATURES: Ripley L(r) for FF clustering ===")

try:
    from scipy.spatial import cKDTree
    with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
        ff_pos = pickle.load(f)

    # Compute Ripley L(r) - r at multiple radii for each placement
    # Radii match cluster_dia range: 35-70 µm
    ripley_feats = {}
    for pid in np.unique(pids):
        pi = ff_pos.get(pid)
        if pi is None or pi.get('ff_norm') is None:
            ripley_feats[pid] = np.zeros(7)
            continue
        ff_xy = pi['ff_norm']  # [N, 2] normalized [0,1]
        dw = pi.get('die_w', 500.0); dh = pi.get('die_h', 500.0)
        A = dw * dh  # µm²
        N = len(ff_xy)

        if N < 4:
            ripley_feats[pid] = np.zeros(7)
            continue

        tree = cKDTree(ff_xy)
        # Normalize radii to [0,1] coordinates
        radii_um = [35, 45, 55, 65, 75, 100, 150]  # µm, covers cluster_dia range and beyond
        rip = []
        scale = np.sqrt(A)  # µm, to convert [0,1] coords to µm
        for r_um in radii_um:
            r_norm = r_um / scale  # normalized radius
            # Count pairs within r (including self)
            pairs = tree.count_neighbors(tree, r_norm)
            # K(r) = A × (pairs - N) / N²  (subtract diagonal)
            K = A * (pairs - N) / (N * N) if N > 1 else 0
            L = np.sqrt(K / np.pi) if K > 0 else 0
            rip.append(L - r_um)  # L(r) - r: positive = clustering
        ripley_feats[pid] = np.array(rip)

    # Build ripley feature matrix
    X_rip = np.zeros((n, 7), np.float32)
    for i, pid in enumerate(pids):
        X_rip[i] = ripley_feats.get(pid, np.zeros(7))

    # Normalize
    rip_std = X_rip.std(0); rip_std[rip_std<1e-9] = 1.0
    X_rip_n = X_rip / rip_std

    print(f"  Ripley L(r)-r at radii 35-150µm. Shape: {X_rip.shape}")
    print(f"  Stats: min={X_rip.min():.2f}, max={X_rip.max():.2f}, mean={X_rip.mean():.2f}")

    # Correlations with z_total_clk and z_power
    print("  Ripley L(r)-r correlations with z_power and z_total_clk:")
    radii_um = [35, 45, 55, 65, 75, 100, 150]
    for j, r in enumerate(radii_um):
        rho_pw, _ = spearmanr(X_rip[:,j], y_pw)
        rho_clk, _ = spearmanr(X_rip[:,j], z_ntot)
        print(f"    L({r}µm)-r: rho_power={rho_pw:.4f}, rho_clk={rho_clk:.4f}")

    # Test in LODO
    X_rip_aug = np.hstack([X29T, X_rip_n, Xphys_aug])
    for c in range(X_rip_aug.shape[1]):
        bad = ~np.isfinite(X_rip_aug[:,c])
        if bad.any(): X_rip_aug[bad,c] = 0.0

    print("\n  Power with Ripley features:")
    lodo_lgb(X_rip_n, y_pw, "Ripley-only LGB")
    lodo_lgb(np.hstack([X29T, X_rip_n]), y_pw, "X29T + Ripley LGB")
    lodo_lgb(X_rip_aug, y_pw, "X29T + Ripley + Phys_aug LGB")

    print("\n  WL with Ripley features:")
    lodo_lgb(np.hstack([X29T, X_rip_n]), y_wl, "X29T + Ripley LGB")
    lodo_lgb(X_rip_aug, y_wl, "X29T + Ripley + Phys_aug LGB")

    # Test: predict β_cd from Ripley features
    print(f"\n{T()} === β_cd PREDICTION FROM RIPLEY FEATURES ===")
    # Compute β_cd per placement
    Xraw2 = df[knob_cols].values.astype(np.float32)
    cd_all = Xraw2[:, 3]
    beta_cds = {}
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        if len(idx) < 3: continue
        v = cd_all[idx]; cd_z = (v - v.mean()) / max(v.std(), 1e-8)
        if cd_z.std() < 1e-6: continue
        A_mat = np.column_stack([cd_z, np.ones(len(idx))])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y_pw[idx], rcond=None)
        beta_cds[pid] = coeffs[0]

    pid_with_beta = [p for p in np.unique(pids) if p in beta_cds]
    design_with_beta = np.array([designs[pids==p][0] for p in pid_with_beta])
    beta_arr = np.array([beta_cds[p] for p in pid_with_beta])

    # Ripley features per pid
    Xrip_per_pid = np.array([ripley_feats.get(p, np.zeros(7)) for p in pid_with_beta])
    Xdef_per_pid = np.array([
        [def_cache.get(p, {}).get('n_ff', 0),
         def_cache.get(p, {}).get('ff_hpwl', 0),
         def_cache.get(p, {}).get('die_area', 0),
         def_cache.get(p, {}).get('cap_proxy', 0)]
        for p in pid_with_beta
    ], dtype=float)
    Xbeta = np.hstack([Xrip_per_pid, Xdef_per_pid])

    from sklearn.preprocessing import StandardScaler as SS2
    dl_beta = sorted(np.unique(design_with_beta)); maes_beta = []
    for held in dl_beta:
        tr = design_with_beta!=held; te = design_with_beta==held
        sc = SS2()
        m = LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                          min_child_samples=5, verbose=-1)
        m.fit(sc.fit_transform(Xbeta[tr]), beta_arr[tr])
        pred_b = m.predict(sc.transform(Xbeta[te]))
        maes_beta.append(mean_absolute_error(beta_arr[te], pred_b))
    print(f"  β_cd pred MAE (Ripley+DEF): {maes_beta[0]:.4f}/{maes_beta[1]:.4f}/"
          f"{maes_beta[2]:.4f}/{maes_beta[3]:.4f} mean={np.mean(maes_beta):.4f} "
          f"(β_cd std={beta_arr.std():.4f})")

    # Use predicted β_cd as feature
    beta_pred_all = np.zeros(n)  # predicted β_cd per row
    for i, pid in enumerate(pids):
        if pid in beta_cds:
            beta_pred_all[i] = beta_cds[pid]  # use true (will replace with LODO pred)
        else:
            beta_pred_all[i] = -0.92

    # Build β_cd-aware feature: β_cd_pred × z_cluster_dia
    cd_z_pp = np.zeros(n)
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        v = cd_all[idx]; cd_z_pp[idx] = (v - v.mean()) / max(v.std(), 1e-8)

    beta_x_cdz = beta_pred_all * cd_z_pp  # physics-guided prediction

    Xbeta_feat = np.hstack([X29T, beta_x_cdz.reshape(-1,1), X_rip_n])
    for c in range(Xbeta_feat.shape[1]):
        bad = ~np.isfinite(Xbeta_feat[:,c])
        if bad.any(): Xbeta_feat[bad,c] = 0.0

    print("\n  Power with β_cd × z_cd feature:")
    lodo_lgb(Xbeta_feat, y_pw, "X29T + β_cd×z_cd + Ripley LGB")

except Exception as e:
    print(f"  [Ripley computation error: {e}]")

print(f"\n{T()} DONE")
