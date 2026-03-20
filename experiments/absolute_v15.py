"""
absolute_v15.py — Physics-Direct Absolute Power/WL Prediction (2% MAPE target)

KEY FINDINGS from exploratory analysis:
  Power constant (k_PA = P / (rel_act × n_nets × f)) varies 5.1x across designs:
    AES=8.76e-14, picorv32=7.64e-14, sha256=1.73e-14, ethmac=6.53e-14
  → Power from SAIF alone insufficient; need C_avg per net (design-specific)

  WL routing factor (k_WA = WL / (n_active × cell_spacing)) varies only 1.4x:
    AES=9.67, picorv32=7.45, sha256=6.98, ethmac=6.71
  → WL is the more tractable problem!

Strategy:
  WL: WL ≈ k_WA × n_active × sqrt(die_area/n_active) = k_WA × sqrt(n_active × die_area)
      k_WA varies 1.4x → need to predict from design features
      Features: core_util, density, aspect_ratio, frac_ff, frac_buf_inv, avg_ds

  Power: P ≈ rel_act × n_nets × f × C_avg_per_net × V²/2
         C_avg_per_net varies 5x → captured by driven_cap_per_ff (from liberty+DEF)
         Additional features: frac_xor, comb_per_ff, avg_ds, cap_proxy

  Both: Use log(y / physics_proxy) as target → should be closer to constant
        Ridge regression generalizes better than LGB for 4-design LODO

Best previous: v11 = power 32% MAPE, WL 13.1% MAPE (absolute LODO)
Target:        power 2-10% MAPE, WL 2-10% MAPE
"""

import pickle, time, warnings, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Absolute v15: Physics-Direct Power/WL Prediction")
print("=" * 70)
sys.stdout.flush()

# ── Load caches ────────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['df']

with open(f'{BASE}/absolute_v7_saif_cache.pkl', 'rb') as f:
    saif_c = pickle.load(f)
with open(f'{BASE}/absolute_v7_def_cache.pkl', 'rb') as f:
    def_c = pickle.load(f)
with open(f'{BASE}/absolute_v13_extended_cache.pkl', 'rb') as f:
    ext_c = pickle.load(f)
with open(f'{BASE}/sim_ff_cache.pkl', 'rb') as f:
    sim_ff_c = pickle.load(f)

df_csv = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
df_csv = df_csv.set_index('run_id').reindex(df['run_id'].values).reset_index()

pids    = df['placement_id'].values
designs = df['design_name'].values
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xknobs  = df[knob_cols].values.astype(np.float32)
n       = len(pids)

y_pw_abs = df_csv['power_total'].values.astype(np.float64)
y_wl_abs = df_csv['wirelength'].values.astype(np.float64)

T_CLK = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0}
V2h   = 1.8**2 / 2  # V²/2 = 1.62

print(f"{T()} Data loaded: {n} rows, {len(np.unique(pids))} placements")
print(f"  Power: [{y_pw_abs.min():.4f}, {y_pw_abs.max():.4f}] W")
print(f"  WL:    [{y_wl_abs.min():.0f}, {y_wl_abs.max():.0f}] µm")
sys.stdout.flush()

# ── Build comprehensive physics feature matrix ─────────────────────────────
print(f"\n{T()} Building feature matrix...")

features_list = []
physics_pw_list = []  # physics proxy for power (to normalize)
physics_wl_list = []  # physics proxy for WL (to normalize)

for i, pid in enumerate(pids):
    design = designs[i]
    f_hz   = 1e9 / T_CLK[design]

    s = saif_c.get(pid, {})
    d = def_c.get(pid, {})
    e = ext_c.get(pid, {})
    ff = sim_ff_c.get(pid, {})
    cs, cd, mw, bd = Xknobs[i]

    # --- SAIF physics features ---
    n_nets      = s.get('n_nets', 0) or 0
    mean_tc     = s.get('mean_tc', 0) or 0
    rel_act     = s.get('rel_act', 0) or 0
    mean_sig_prob = s.get('mean_sig_prob', 0) or 0
    frac_zero   = s.get('frac_zero', 0) or 0
    frac_high   = s.get('frac_high_act', 0) or 0
    tc_std_norm = s.get('tc_std_norm', 0) or 0

    # --- DEF features ---
    n_ff        = d.get('n_ff', 0) or 0
    n_active    = d.get('n_active', 0) or 0
    die_area    = d.get('die_area', 1) or 1
    die_w       = d.get('die_w', 500) or 500
    die_h       = d.get('die_h', 500) or 500
    die_aspect  = d.get('die_aspect', 1) or 1
    ff_hpwl     = d.get('ff_hpwl', 0) or 0
    ff_spacing  = d.get('ff_spacing', 0) or 0
    ff_density  = d.get('ff_density', 0) or 0
    avg_ds      = d.get('avg_ds', 1) or 1
    p90_ds      = d.get('p90_ds', 1) or 1
    frac_xor    = d.get('frac_xor', 0) or 0
    frac_mux    = d.get('frac_mux', 0) or 0
    frac_and_or = d.get('frac_and_or', 0) or 0
    frac_nand_nor = d.get('frac_nand_nor', 0) or 0
    frac_ff_act = d.get('frac_ff_active', 0) or 0
    frac_buf_inv = d.get('frac_buf_inv', 0) or 0
    comb_per_ff = d.get('comb_per_ff', 0) or 0
    n_buf       = d.get('n_buf', 0) or 0
    n_inv       = d.get('n_inv', 0) or 0
    n_xor_xnor  = d.get('n_xor_xnor', 0) or 0
    cap_proxy   = d.get('cap_proxy', 0) or 0  # n_active × avg_ds

    # --- v13 extended features (MST, Wasserstein, liberty caps) ---
    mst_per_ff  = e.get('mst_per_ff', 0) or 0
    mst_norm    = e.get('mst_norm', 0) or 0
    dens_cv     = e.get('dens_cv', 0) or 0
    dens_gini   = e.get('dens_gini', 0) or 0
    dens_entropy = e.get('dens_entropy', 0) or 0
    wass_total  = e.get('wass_total', 0) or 0
    driven_cap_ff = e.get('driven_cap_per_ff', 0) or 0  # pF/FF
    driven_cap_cv = e.get('driven_cap_cv', 0) or 0

    # --- FF spatial features from sim_ff_cache (actual kNN) ---
    knn1_mean = 0.0; knn1_std = 0.0
    if ff and ff.get('xy') is not None and len(ff['xy']) > 1:
        xy_um = np.array(ff['xy'])
        tree = cKDTree(xy_um)
        dists, _ = tree.query(xy_um, k=min(2, len(xy_um)))
        if dists.shape[1] > 1:
            knn1 = dists[:, 1]
            knn1_mean = knn1.mean()
            knn1_std  = knn1.std()

    # === PHYSICS PROXIES ===
    # Power proxy: rel_act × n_nets × f [C_avg × V²/2 is constant ≈ k_PA]
    phys_pw = max(rel_act * n_nets * f_hz, 1e-10)

    # WL proxy: n_active × sqrt(die_area/n_active) = sqrt(n_active × die_area)
    phys_wl = max(np.sqrt(n_active * die_area), 1e-3)

    # === DESIGN-INVARIANT FEATURES (normalize out size effects) ===
    # Core power features
    rel_act_log  = np.log1p(rel_act)
    n_nets_log   = np.log1p(n_nets)
    f_clk_log    = np.log1p(f_hz / 1e6)  # log(MHz)

    # Per-cell features (size-invariant)
    avg_cap_per_net = driven_cap_ff / max(comb_per_ff + 1, 1)  # scale by comb density
    toggle_energy   = rel_act * driven_cap_ff * 1e-12 * f_hz * V2h  # W per FF (physics)
    total_toggle_energy = toggle_energy * n_ff  # total W estimate

    # WL features
    cell_spacing = np.sqrt(die_area / max(n_active, 1))
    wl_proxy_ratio_est = max(knn1_mean / (cell_spacing + 1e-6), 1e-6)  # kNN vs cell_spacing

    # Design-structure ratios (cross-design invariant)
    xor_density   = n_xor_xnor / max(n_active, 1)  # XOR fraction drives low k_PA
    logic_depth_proxy = comb_per_ff  # logic depth per FF
    cap_density   = avg_ds / (comb_per_ff + 1)  # drive strength per logic stage
    ff_area_frac  = n_ff / max(n_active, 1)  # FF fraction of design

    # CTS knob features (within-placement variation)
    n_buf_cts_proxy = n_ff / max(cs, 1)  # CTS buffer count proxy
    cd_vs_spacing   = cd / max(knn1_mean, 1)  # cluster dia vs FF spacing
    mw_vs_spacing   = mw / max(knn1_mean, 1)  # wire budget vs FF spacing

    feat = [
        # === LOG-SCALE ABSOLUTE FEATURES ===
        np.log1p(n_nets),       # total nets (size proxy)
        np.log1p(n_active),     # total active cells
        np.log1p(n_ff),         # flip-flop count
        np.log1p(die_area),     # die area
        np.log1p(f_hz / 1e6),  # clock frequency (MHz)
        np.log1p(rel_act * 1000),  # activity factor (amplified for log)
        np.log1p(mean_sig_prob * 100),  # signal probability

        # === DESIGN-STRUCTURE RATIOS (size-invariant) ===
        frac_xor,               # XOR/XNOR fraction (low cap → low power ratio)
        frac_mux,               # MUX fraction
        frac_and_or,            # AND/OR fraction
        frac_nand_nor,          # NAND/NOR fraction (high-speed → higher cap)
        frac_ff_act,            # FF fraction of total active cells
        frac_buf_inv,           # buffer/inverter fraction
        comb_per_ff,            # combinational cells per FF (logic depth proxy)
        avg_ds,                 # average drive strength (cap proxy)
        p90_ds,                 # 90th percentile drive strength
        np.log1p(avg_ds),       # log drive strength

        # === SAIF SWITCHING STATISTICS ===
        frac_zero,              # fraction of static nets (non-switching)
        frac_high,              # fraction of high-activity nets
        tc_std_norm,            # toggle count variation

        # === LIBERTY-BASED CAP FEATURES ===
        np.log1p(driven_cap_ff * 1000),  # driven cap per FF (pF × 1000)
        driven_cap_cv,          # variation in driven cap
        np.log1p(toggle_energy * 1e9),   # physics power estimate (nW per FF)
        np.log1p(total_toggle_energy * 1000),  # total physics estimate (mW)

        # === WL/SPATIAL FEATURES ===
        np.log1p(knn1_mean),    # average FF nearest-neighbor distance (µm)
        np.log1p(cell_spacing), # average cell spacing
        wl_proxy_ratio_est,     # kNN vs cell_spacing ratio
        mst_per_ff,             # MST per FF (µm) from v13
        mst_norm,               # MST normalized by die scale
        dens_cv,                # FF density CV (spatial heterogeneity)
        dens_gini,              # FF density Gini coefficient
        dens_entropy,           # FF density entropy
        np.log1p(wass_total),   # Wasserstein distance (logic-FF routing pressure)
        die_aspect,             # die aspect ratio
        np.log1p(ff_hpwl),     # FF bounding box HPWL

        # === CTS KNOBS ===
        np.log1p(cs), np.log1p(cd), np.log1p(mw), np.log1p(bd),
        cs / max(n_ff, 1),      # cluster_size relative to n_ff
        cd / max(knn1_mean, 1), # cluster_dia vs FF spacing
        mw / max(knn1_mean, 1), # wire budget vs FF spacing
        np.log1p(n_buf_cts_proxy),  # log(CTS buffer count proxy)

        # === POWER-SPECIFIC INTERACTIONS ===
        # k_PA predictor: need features that explain 5x variation in C_avg_per_net
        np.log1p(rel_act * driven_cap_ff * f_hz / 1e6),  # toggle energy rate
        frac_xor * rel_act,     # xor activity interaction (sha256 effect)
        comb_per_ff * avg_ds,   # logic complexity × drive strength
        np.log1p(cap_proxy / max(n_nets, 1)),  # cap per net
    ]

    features_list.append(feat)
    physics_pw_list.append(phys_pw)
    physics_wl_list.append(phys_wl)

X = np.array(features_list, dtype=np.float64)
phys_pw = np.array(physics_pw_list)
phys_wl = np.array(physics_wl_list)

# Fix NaN/inf
for c in range(X.shape[1]):
    bad = ~np.isfinite(X[:, c])
    if bad.any():
        med = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
        X[bad, c] = med

print(f"{T()} Feature matrix: {X.shape}")
sys.stdout.flush()

# ── Physics formula check ─────────────────────────────────────────────────
print(f"\n{T()} === PHYSICS FORMULA CHECK ===")
def mape(y_true, y_pred):
    return 100.0 * np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-12)))

def mape_by_design(y_true, y_pred, designs):
    for d in sorted(np.unique(designs)):
        m = designs == d
        mp = mape(y_true[m], y_pred[m])
        print(f"  {d}: MAPE={mp:.1f}%")

# Per-design: what's the best constant k that minimizes MAPE for phys_pw * k = P?
print("Power: find k* = argmin MAPE over P = k* × phys_pw")
for design in sorted(np.unique(designs)):
    m = designs == design
    # Optimal k for this design
    k_vals = y_pw_abs[m] / phys_pw[m]
    k_opt = k_vals.mean()  # best k (oracle)
    P_pred = phys_pw[m] * k_opt
    mp = mape(y_pw_abs[m], P_pred)
    print(f"  {design}: k*={k_opt:.3e}, MAPE_oracle={mp:.1f}%, k_CV={k_vals.std()/k_vals.mean():.4f}")

print()
print("WL: find k* = argmin MAPE over WL = k* × phys_wl")
for design in sorted(np.unique(designs)):
    m = designs == design
    k_vals = y_wl_abs[m] / phys_wl[m]
    k_opt = k_vals.mean()
    WL_pred = phys_wl[m] * k_opt
    mp = mape(y_wl_abs[m], WL_pred)
    print(f"  {design}: k*={k_opt:.3f}, MAPE_oracle={mp:.1f}%, k_CV={k_vals.std()/k_vals.mean():.4f}")

# ── Targets: log ratios ───────────────────────────────────────────────────
# Power: predict log(P / phys_pw) then exp(pred) × phys_pw
y_pw_ratio = np.log(y_pw_abs / phys_pw)
y_wl_ratio = np.log(y_wl_abs / phys_wl)

print(f"\nPower log-ratio range: [{y_pw_ratio.min():.3f}, {y_pw_ratio.max():.3f}]")
print(f"WL    log-ratio range: [{y_wl_ratio.min():.3f}, {y_wl_ratio.max():.3f}]")
print("  (small range = physics proxy is accurate)")

for design in sorted(np.unique(designs)):
    m = designs == design
    print(f"  {design}: pw_ratio=[{y_pw_ratio[m].min():.3f},{y_pw_ratio[m].max():.3f}]  "
          f"wl_ratio=[{y_wl_ratio[m].min():.3f},{y_wl_ratio[m].max():.3f}]")
sys.stdout.flush()

# ── LODO evaluation functions ─────────────────────────────────────────────
def lodo_mape(X, y_ratio, normalizer, label, model_cls, **kw):
    """LODO absolute MAPE: predict ratio, denormalize, compute MAPE."""
    dl = sorted(np.unique(designs))
    mapes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = model_cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y_ratio[tr])
        pred_ratio = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_ratio) * normalizer[te]
        true_abs = np.exp(y_ratio[te]) * normalizer[te]
        mapes.append(mape(true_abs, pred_abs))
    mean_m = np.mean(mapes)
    s = '✓' if mean_m < 5 else ('~' if mean_m < 15 else '')
    folds = '/'.join([f'{x:.1f}' for x in mapes])
    print(f"  {label}: {folds}%  mean={mean_m:.1f}% {s}")
    sys.stdout.flush()
    return mean_m, mapes

# Also direct log prediction (no physics normalizer)
def lodo_log_direct(X, y_log, label, model_cls, **kw):
    """LODO: predict log(y) directly, convert to absolute MAPE."""
    dl = sorted(np.unique(designs))
    mapes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = model_cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y_log[tr])
        pred_log = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_log)
        true_abs = np.exp(y_log[te])
        mapes.append(mape(true_abs, pred_abs))
    mean_m = np.mean(mapes)
    s = '✓' if mean_m < 5 else ('~' if mean_m < 15 else '')
    folds = '/'.join([f'{x:.1f}' for x in mapes])
    print(f"  {label}: {folds}%  mean={mean_m:.1f}% {s}")
    sys.stdout.flush()
    return mean_m, mapes

# ── POWER EXPERIMENTS ─────────────────────────────────────────────────────
print(f"\n{T()} === POWER ABSOLUTE MAPE (target: <2%) ===")
print(f"  Prior best: ~32% MAPE (v11)")

LGB_F = dict(n_estimators=300, num_leaves=20, learning_rate=0.03, min_child_samples=10, verbose=-1)
LGB_S = dict(n_estimators=500, num_leaves=15, learning_rate=0.02, min_child_samples=10, verbose=-1)

# Baseline: median prediction per design (oracle of design mean)
print(f"\n  --- Reference: design-median baseline ---")
for held in sorted(np.unique(designs)):
    tr = designs != held; te = designs == held
    pred = np.full(te.sum(), y_pw_abs[tr].median() if hasattr(y_pw_abs[tr], 'median')
                   else np.median(y_pw_abs[tr]))
    mp = mape(y_pw_abs[te], pred)
    print(f"  held={held}: naive_median MAPE={mp:.1f}%")

print(f"\n  --- Physics proxy only (k=1) ---")
mape_vals = []
for held in sorted(np.unique(designs)):
    te = designs == held
    mp = mape(y_pw_abs[te], phys_pw[te])
    mape_vals.append(mp)
    print(f"  held={held}: MAPE={mp:.1f}%")
print(f"  mean: {np.mean(mape_vals):.1f}%")

print(f"\n  --- Ridge regression on log-ratio ---")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lodo_mape(X, y_pw_ratio, phys_pw, f"Ridge(α={alpha}) pw_ratio",
              Ridge, alpha=alpha)

print(f"\n  --- Ridge on log(P) directly ---")
for alpha in [0.1, 1.0, 10.0]:
    lodo_log_direct(X, np.log(y_pw_abs), f"Ridge(α={alpha}) log(P)",
                    Ridge, alpha=alpha)

print(f"\n  --- LGB on log-ratio ---")
lodo_mape(X, y_pw_ratio, phys_pw, "LGB_300 pw_ratio", LGBMRegressor, **LGB_F)
lodo_mape(X, y_pw_ratio, phys_pw, "LGB_500 pw_ratio", LGBMRegressor, **LGB_S)

print(f"\n  --- LGB on log(P) directly ---")
lodo_log_direct(X, np.log(y_pw_abs), "LGB_300 log(P)", LGBMRegressor, **LGB_F)

# ── WL EXPERIMENTS ────────────────────────────────────────────────────────
print(f"\n{T()} === WL ABSOLUTE MAPE (target: <2%) ===")
print(f"  Prior best: ~13.1% MAPE (v11)")

print(f"\n  --- Physics proxy only (k=1) ---")
mape_vals = []
for held in sorted(np.unique(designs)):
    te = designs == held
    mp = mape(y_wl_abs[te], phys_wl[te])
    mape_vals.append(mp)
    print(f"  held={held}: MAPE={mp:.1f}%")
print(f"  mean: {np.mean(mape_vals):.1f}%")

print(f"\n  --- Ridge on log-ratio ---")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lodo_mape(X, y_wl_ratio, phys_wl, f"Ridge(α={alpha}) wl_ratio",
              Ridge, alpha=alpha)

print(f"\n  --- Ridge on log(WL) directly ---")
for alpha in [0.1, 1.0, 10.0]:
    lodo_log_direct(X, np.log(y_wl_abs), f"Ridge(α={alpha}) log(WL)",
                    Ridge, alpha=alpha)

print(f"\n  --- LGB on log-ratio ---")
lodo_mape(X, y_wl_ratio, phys_wl, "LGB_300 wl_ratio", LGBMRegressor, **LGB_F)
lodo_mape(X, y_wl_ratio, phys_wl, "LGB_500 wl_ratio", LGBMRegressor, **LGB_S)

print(f"\n  --- LGB on log(WL) directly ---")
lodo_log_direct(X, np.log(y_wl_abs), "LGB_300 log(WL)", LGBMRegressor, **LGB_F)

# ── Ablation: which features matter? ─────────────────────────────────────
print(f"\n{T()} === FEATURE ABLATION (WL, best model) ===")

# Minimal WL features (Donath's model: just geometry)
idx_geom = [0, 1, 2, 3, 4]  # n_nets, n_active, n_ff, die_area, f_clk
X_geom = X[:, idx_geom]
for alpha in [1.0, 10.0]:
    lodo_mape(X_geom, y_wl_ratio, phys_wl, f"Ridge(α={alpha}) geometry only",
              Ridge, alpha=alpha)

# Add spatial features
idx_spatial = list(range(5)) + list(range(24, 34))  # geometry + spatial
X_spatial = X[:, idx_spatial]
for alpha in [1.0, 10.0]:
    lodo_mape(X_spatial, y_wl_ratio, phys_wl, f"Ridge(α={alpha}) geom+spatial",
              Ridge, alpha=alpha)

# Full features minus CTS knobs
idx_no_cts = list(range(35))  # exclude CTS knob features
X_no_cts = X[:, idx_no_cts]
lodo_mape(X_no_cts, y_wl_ratio, phys_wl, "Ridge(1.0) no_CTS", Ridge, alpha=1.0)
lodo_mape(X_no_cts, y_wl_ratio, phys_wl, "Ridge(10) no_CTS", Ridge, alpha=10.0)

# ── Detailed per-design breakdown (best models) ────────────────────────────
print(f"\n{T()} === DETAILED BREAKDOWN (best configurations) ===")
def detailed_lodo(X, y_ratio, normalizer, label, model_cls, **kw):
    dl = sorted(np.unique(designs))
    print(f"\n  {label}:")
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = model_cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y_ratio[tr])
        pred_ratio = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_ratio) * normalizer[te]
        true_abs = np.exp(y_ratio[te]) * normalizer[te]
        mp = mape(true_abs, pred_abs)
        print(f"    held={held}: pred=[{pred_abs.min():.4f},{pred_abs.max():.4f}]  "
              f"true=[{true_abs.min():.4f},{true_abs.max():.4f}]  MAPE={mp:.1f}%")

detailed_lodo(X, y_pw_ratio, phys_pw, "POWER Ridge(α=1) full features", Ridge, alpha=1.0)
detailed_lodo(X, y_wl_ratio, phys_wl, "WL Ridge(α=1) full features", Ridge, alpha=1.0)

# ── Feature importance for LGB ────────────────────────────────────────────
print(f"\n{T()} === WL FEATURE IMPORTANCE (held=ethmac) ===")
feat_names = [
    'log_n_nets', 'log_n_active', 'log_n_ff', 'log_die_area', 'log_f_clk_MHz',
    'log_rel_act', 'log_mean_sig_prob',
    'frac_xor', 'frac_mux', 'frac_and_or', 'frac_nand_nor',
    'frac_ff_active', 'frac_buf_inv', 'comb_per_ff',
    'avg_ds', 'p90_ds', 'log_avg_ds',
    'frac_zero', 'frac_high_act', 'tc_std_norm',
    'log_driven_cap_pF', 'driven_cap_cv', 'log_toggle_energy_nW', 'log_total_toggle_mW',
    'log_knn1_mean', 'log_cell_spacing', 'knn1_vs_cell_spacing',
    'mst_per_ff', 'mst_norm', 'dens_cv', 'dens_gini', 'dens_entropy',
    'log_wass_total', 'die_aspect', 'log_ff_hpwl',
    'log_cs', 'log_cd', 'log_mw', 'log_bd',
    'cs_per_ff', 'cd_vs_knn1', 'mw_vs_knn1', 'log_n_buf_cts',
    'log_toggle_rate', 'xor_act_interact', 'comb_ds_interact', 'log_cap_per_net',
]

tr = designs != 'ethmac'; te = designs == 'ethmac'
sc = StandardScaler()
m = LGBMRegressor(**LGB_F)
m.fit(sc.fit_transform(X[tr]), y_wl_ratio[tr])
imps = m.feature_importances_
top = sorted(zip(feat_names[:len(imps)], imps), key=lambda x: -x[1])[:15]
for name, imp in top:
    print(f"  {name}: {imp}")

print(f"\n{T()} DONE")
