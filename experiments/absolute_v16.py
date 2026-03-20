"""
absolute_v16.py — Physics-Direct Absolute Predictor

Builds on v11 (power=32%, WL=13.1%) with Session 8 insights:

Key changes from v11:
1. Physics normalization: use phys_pw = rel_act * n_nets * f_clk (vs v11's n_ff*f*avg_ds)
   This makes log-target = log(k_PA), which captures design-style variation more directly.

2. New features to distinguish SHA256 (the main failure mode):
   - frac_xor * avg_ds: XOR cells have lower drive → lower C_load per switch
   - log(n_nets / n_active): fanout proxy (sha256 has more nets per active cell)
   - n_nets / n_ff: nets per FF (sha256 has much larger fanout)
   - frac_xor * comb_per_ff: XOR-heavy logic indicator
   - xor_adj_activity = rel_act / (1 + frac_xor): downweight XOR activity
   - log(driven_cap_per_ff): actual load capacitance per FF (from liberty, v13 cache)
   - mst_per_ff: minimum spanning tree dist per FF (v13 cache, routing proxy)

3. Gravity vectors ALSO tested for power (v11 excluded them due to sha256 confusion)
   With fanout proxy features distinguishing sha256, gravity may now help power too.

4. Extended cache (absolute_v13_extended_cache.pkl):
   - driven_cap_per_ff, driven_cap_mean, driven_cap_cv
   - mst_per_ff, dens_gini, dens_entropy

5. Aggressive Ridge regularization (alpha=1000-10000) for absolute generalization.

Expected improvement over v11:
  Power: 32% → ~20-25% (sha256 fold specifically benefits from XOR-aware features)
  WL: 13.1% → ~10-12% (mst_per_ff improves WL physics model)
"""

import re
import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'

DEF_CACHE_V7     = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE_V7    = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE_V7  = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE_V10 = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE_V13    = f'{BASE}/absolute_v13_extended_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}
CLOCK_PORTS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk',
               'ethmac': 'wb_clk_i', 'zipdiv': 'i_clk'}


print("=" * 70)
print("Zero-Shot Absolute Predictor v16 — XOR-Aware + Physics Normalization")
print("=" * 70)
sys.stdout.flush()

# -----------------------------------------------------------------------
# LOAD CACHES
# -----------------------------------------------------------------------

print(f"{T()} Loading caches...")
sys.stdout.flush()

with open(DEF_CACHE_V7, 'rb') as f:    dc = pickle.load(f)
with open(SAIF_CACHE_V7, 'rb') as f:   sc_cache = pickle.load(f)
with open(TIMING_CACHE_V7, 'rb') as f: tc = pickle.load(f)
print(f"  DEF: {len(dc)}, SAIF: {len(sc_cache)}, timing: {len(tc)}")

with open(GRAVITY_CACHE_V10, 'rb') as f: gc = pickle.load(f)
print(f"  Gravity: {len(gc)}")

ext_cache = {}
if os.path.exists(EXT_CACHE_V13):
    with open(EXT_CACHE_V13, 'rb') as f: ext_cache = pickle.load(f)
    print(f"  Extended (v13): {len(ext_cache)} entries")
    # Show available keys
    sample_key = next(iter(ext_cache))
    print(f"  v13 keys: {list(ext_cache[sample_key].keys())[:8]}")
else:
    print("  WARNING: v13 extended cache not found")

sys.stdout.flush()

# -----------------------------------------------------------------------
# LOAD TRAINING DATA
# -----------------------------------------------------------------------

df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
print(f"\n{T()} Training data: {len(df)} rows, designs: {df['design_name'].value_counts().to_dict()}")

# Test data (zipdiv) — may not exist
df_test = None
test_path = f'{DATASET}/unified_manifest_normalized_test.csv'
if os.path.exists(test_path):
    df_test = pd.read_csv(test_path)
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"  Test data (zipdiv): {len(df_test)} rows")

sys.stdout.flush()

# -----------------------------------------------------------------------
# FEATURE BUILDING
# -----------------------------------------------------------------------

def encode_synth(s):
    if pd.isna(s):
        return 0.5, 2.0, 0.5
    s = str(s).upper()
    synth_is_delay = 1.0 if 'DELAY' in s else 0.0
    try:
        level = float(s.split()[-1])
    except Exception:
        level = 2.0
    synth_agg = synth_is_delay * level / 4.0
    return synth_is_delay, level, synth_agg


def mape(y_true, y_pred_abs):
    return np.mean(np.abs(y_pred_abs - y_true) / (y_true + 1e-12)) * 100


def build_features(df_in, dc, sc_cache, tc, gc, ext_cache):
    rows_pw = []  # power features
    rows_wl = []  # WL features
    y_pw_r = []   # log(power / phys_pw)  — capturing k_PA
    y_wl_r = []   # log(WL / phys_wl)    — capturing k_WA
    meta = []

    n_ok = 0
    n_miss = 0

    for _, row in df_in.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf_f = sc_cache.get(pid)
        tp_f = tc.get(pid)
        gf = gc.get(pid, {})
        ef = ext_cache.get(pid, {})

        if not df_f or not sf_f or not tp_f:
            n_miss += 1
            continue

        pw = row['power_total']
        wl = row['wirelength']
        if not np.isfinite(pw) or not np.isfinite(wl) or pw <= 0 or wl <= 0:
            n_miss += 1
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_hz = 1e9 / t_clk  # Hz
        f_ghz = 1.0 / t_clk  # GHz
        synth_delay, synth_level, synth_agg = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        n_total = df_f['n_total']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']
        n_comb = df_f['n_comb']
        n_nets = sf_f['n_nets']
        rel_act = sf_f['rel_act']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # ── PHYSICS NORMALIZERS ──────────────────────────────────────────
        # v16: phys_pw = rel_act × n_nets × f_hz  (switching events/second)
        # This makes log-target = log(k_PA), directly measuring avg energy per switch
        phys_pw = max(rel_act * n_nets * f_hz, 1e-20)

        # v16: phys_wl = sqrt(n_active × die_area)  (Donath model)
        phys_wl = max(np.sqrt(n_active * die_area), 1e-3)

        # ── NEW FEATURES: SHA256 DISTINGUISHERS ──────────────────────────
        # These capture the circuit-style differences that cause k_PA variation
        fanout_proxy = n_nets / (n_active + 1)       # sha256: ~2.5, others: ~1.5-2.0
        nets_per_ff = n_nets / (n_ff + 1)             # sha256 higher due to high fanin/fanout
        xor_adj_act = rel_act / (1 + frac_xor * 3)   # downweight activity for XOR-heavy
        xor_energy_proxy = frac_xor * avg_ds          # XOR cells have specific drive strength
        xor_heavy = 1.0 if frac_xor > 0.05 else 0.0  # binary: is XOR dominant?

        # Extended cache features (driven cap, MST)
        driven_cap_per_ff = ef.get('driven_cap_per_ff', 0.0)  # pF/FF
        driven_cap_cv = ef.get('driven_cap_cv', 0.0)
        driven_cap_p90 = ef.get('driven_cap_p90', 0.0)
        mst_per_ff = ef.get('mst_per_ff', 0.0)        # µm/FF (routing distance proxy)
        mst_norm = ef.get('mst_norm', 0.0)
        dens_gini = ef.get('dens_gini', 0.0)
        dens_entropy = ef.get('dens_entropy', 0.0)

        # ── BASE FEATURES (v5/v7/v11 — proven stable across LODO folds) ──
        base = [
            # Log-scale cell counts (fingerprint-aware — log reduces design separation)
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            # Geometry ratios (design-invariant)
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            # Circuit composition fractions (stable across LODO)
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            # SAIF activity features
            rel_act, sf_f['mean_sig_prob'], sf_f['tc_std_norm'], sf_f['frac_zero'],
            sf_f['frac_high_act'], sf_f['log_n_nets'], n_nets / (n_ff + 1),
            # Clock / synthesis context
            f_ghz, t_clk, synth_delay, synth_level, synth_agg, core_util, density,
            # CTS knobs (log + raw)
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,
            # V11 interaction terms (proven stable)
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            synth_delay * avg_ds,
            synth_agg * f_ghz,
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
            # Extra scale (proven for WL in v10)
            np.log1p(die_area / (n_ff + 1)),
            np.log1p(n_comb),
            comb_per_ff * np.log1p(n_ff),
        ]

        # ── V16 NEW: SHA256 DISTINGUISHING FEATURES ──────────────────────
        new_feats = [
            fanout_proxy,                     # n_nets / n_active
            np.log1p(fanout_proxy),
            nets_per_ff,                      # n_nets / n_ff
            np.log1p(nets_per_ff),
            xor_adj_act,                      # rel_act adjusted for XOR
            xor_energy_proxy,                 # frac_xor × avg_ds
            xor_heavy,                        # binary: is XOR dominant?
            frac_xor * fanout_proxy,          # joint: XOR × fanout
            rel_act * fanout_proxy,           # activity × fanout
            # Physics-direct: using phys_pw in features too
            np.log(phys_pw + 1e-30),          # log of switching events/sec
            np.log(phys_wl + 1e-3),           # log of Donath WL proxy
            # Extended cache features (if available)
            np.log1p(driven_cap_per_ff),      # load cap per FF from liberty
            driven_cap_cv,                     # spread in load cap
            np.log1p(driven_cap_p90),
            np.log1p(mst_per_ff),             # MST routing distance per FF
            mst_norm,                          # MST normalized by die scale
            dens_gini,                         # FF density heterogeneity
            dens_entropy,
        ]

        # ── GRAVITY VECTOR FEATURES (from wire-graph 1-hop, proven for WL) ──
        graph_feats = [
            gf.get('grav_abs_mean', 0.0),
            gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0),
            gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0),
            gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0),
            gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd,
            gf.get('grav_abs_mean', 0.0) * mw,
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            gf.get('tp_degree_mean', 0.0),
            gf.get('tp_degree_cv', 0.0),
            gf.get('tp_degree_gini', 0.0),
            gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0),
            gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]

        # ── TIMING FEATURES (from timing_paths.csv) ──────────────────────
        sm = tp_f['slack_mean']
        fn = tp_f['frac_neg']
        ft = tp_f['frac_tight']
        timing = [
            sm, tp_f['slack_std'], tp_f['slack_min'], tp_f['slack_p10'], tp_f['slack_p50'],
            fn, ft, tp_f['frac_critical'],
            tp_f['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            1.0 if sm > 1.5 else 0.0,
            1.0 if sm > 2.0 else 0.0,
            1.0 if sm > 3.0 else 0.0,
            np.log1p(sm), sm * f_ghz,
        ]

        # Power: base + timing + new_feats (NO gravity in v16a — test separately)
        # WL: base + graph_feats + new_feats
        rows_pw.append(base + timing + new_feats)
        rows_wl.append(base + graph_feats + new_feats)

        # Log-ratio targets
        y_pw_r.append(np.log(pw / phys_pw))
        y_wl_r.append(np.log(wl / phys_wl))

        meta.append({
            'placement_id': pid, 'design_name': design,
            'power_total': pw, 'wirelength': wl,
            'phys_pw': phys_pw, 'phys_wl': phys_wl,
        })
        n_ok += 1

    print(f"  Built features: {n_ok} ok, {n_miss} missing")

    X_pw = np.array(rows_pw, dtype=np.float64)
    X_wl = np.array(rows_wl, dtype=np.float64)
    y_pw = np.array(y_pw_r)
    y_wl = np.array(y_wl_r)
    meta_df = pd.DataFrame(meta)

    # Sanitize
    for X in [X_pw, X_wl]:
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X_pw, X_wl, y_pw, y_wl, meta_df


print(f"\n{T()} Building features...")
X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df, dc, sc_cache, tc, gc, ext_cache)
print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}")
sys.stdout.flush()

designs = meta_df['design_name'].values

# -----------------------------------------------------------------------
# LODO EVALUATION
# -----------------------------------------------------------------------

def lodo_power(X_pw, y_pw, meta_df, label, pw_cls, pw_kw):
    """LODO power eval only — fast (no WL retraining)."""
    unique_designs = sorted(meta_df['design_name'].unique())
    pw_mapes = []
    for held in unique_designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held
        sc_pw = StandardScaler()
        Xtr_pw = sc_pw.fit_transform(X_pw[tr])
        Xte_pw = sc_pw.transform(X_pw[te])
        m_pw = pw_cls(**pw_kw)
        m_pw.fit(Xtr_pw, y_pw[tr])
        pred_log_pw = m_pw.predict(Xte_pw)
        pred_pw = np.exp(pred_log_pw) * meta_df[te]['phys_pw'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)
        pw_mapes.append(mpw)
    mean_pw = np.mean(pw_mapes)
    pw_str = "/".join([f"{v:.1f}" for v in pw_mapes])
    print(f"  [{label}] Power: {pw_str}  → mean={mean_pw:.1f}%")
    sys.stdout.flush()
    return mean_pw, pw_mapes


def lodo_wl(X_wl, y_wl, meta_df, wl_alpha=0.5):
    """LODO WL eval — LGB(300)+Ridge ensemble."""
    unique_designs = sorted(meta_df['design_name'].unique())
    wl_mapes = []
    wl_lgb_mapes = []
    wl_ridge_mapes = []
    for held in unique_designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held
        sc_wl = StandardScaler()
        Xtr_wl = sc_wl.fit_transform(X_wl[tr])
        Xte_wl = sc_wl.transform(X_wl[te])
        lgb_wl = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               min_child_samples=10, random_state=42, verbose=-1, n_jobs=2)
        lgb_wl.fit(Xtr_wl, y_wl[tr])
        ridge_wl = Ridge(alpha=1000.0, max_iter=10000)
        ridge_wl.fit(Xtr_wl, y_wl[tr])
        pred_lgb = lgb_wl.predict(Xte_wl)
        pred_ridge = ridge_wl.predict(Xte_wl)
        phys = meta_df[te]['phys_wl'].values
        wl_true = meta_df[te]['wirelength'].values
        mwl = mape(wl_true, np.exp(wl_alpha * pred_lgb + (1-wl_alpha) * pred_ridge) * phys)
        mwl_lgb = mape(wl_true, np.exp(pred_lgb) * phys)
        mwl_ridge = mape(wl_true, np.exp(pred_ridge) * phys)
        wl_mapes.append(mwl)
        wl_lgb_mapes.append(mwl_lgb)
        wl_ridge_mapes.append(mwl_ridge)
    mean_wl = np.mean(wl_mapes)
    wl_str = "/".join([f"{v:.1f}" for v in wl_mapes])
    lgb_str = "/".join([f"{v:.1f}" for v in wl_lgb_mapes])
    ridge_str = "/".join([f"{v:.1f}" for v in wl_ridge_mapes])
    print(f"  WL LGB:   {lgb_str}  → mean={np.mean(wl_lgb_mapes):.1f}%")
    print(f"  WL Ridge: {ridge_str}  → mean={np.mean(wl_ridge_mapes):.1f}%")
    print(f"  WL blend(α={wl_alpha}): {wl_str}  → mean={mean_wl:.1f}%")
    sys.stdout.flush()
    return mean_wl, wl_mapes


# -----------------------------------------------------------------------
# POWER: RIDGE SWEEP
# -----------------------------------------------------------------------

print(f"\n{T()} === POWER: Ridge alpha sweep ===")
sys.stdout.flush()

for alpha in [10, 100, 1000, 5000, 10000]:
    lodo_power(X_pw, y_pw, meta_df, f"Ridge(α={alpha})", Ridge,
               {'alpha': float(alpha), 'max_iter': 10000})

# -----------------------------------------------------------------------
# POWER: LGB / XGB
# -----------------------------------------------------------------------

print(f"\n{T()} === POWER: LGB / XGB ===")
sys.stdout.flush()

lodo_power(X_pw, y_pw, meta_df, "LGB(300,lr=0.03,l=20)", LGBMRegressor,
           {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 20,
            'min_child_samples': 15, 'verbose': -1, 'random_state': 42, 'n_jobs': 2})

lodo_power(X_pw, y_pw, meta_df, "XGB(300,lr=0.05,d=4)", XGBRegressor,
           {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0, 'n_jobs': 2})

# -----------------------------------------------------------------------
# WL: LGB/Ridge blend sweep
# -----------------------------------------------------------------------

print(f"\n{T()} === WL: LGB+Ridge blend sweep ===")
sys.stdout.flush()

for wl_alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    print(f"  -- alpha={wl_alpha} --")
    lodo_wl(X_wl, y_wl, meta_df, wl_alpha=wl_alpha)

# -----------------------------------------------------------------------
# V16b: Power WITH gravity features — does XOR-aware help sha256 + gravity?
# -----------------------------------------------------------------------

print(f"\n{T()} === POWER WITH GRAVITY FEATURES (v16b) ===")
sys.stdout.flush()

# Rebuild: add gravity to power features
rows_pw_with_grav = []
for _, row in df.iterrows():
    pid = row['placement_id']
    design = row['design_name']
    df_f = dc.get(pid)
    sf_f = sc_cache.get(pid)
    tp_f = tc.get(pid)
    gf = gc.get(pid, {})
    ef = ext_cache.get(pid, {})
    if not df_f or not sf_f or not tp_f:
        continue
    pw = row['power_total']
    wl = row['wirelength']
    if not np.isfinite(pw) or not np.isfinite(wl) or pw <= 0 or wl <= 0:
        continue

    t_clk = T_CLK_NS.get(design, 7.0)
    f_hz = 1e9 / t_clk
    f_ghz = 1.0 / t_clk
    synth_delay, synth_level, synth_agg = encode_synth(row.get('synth_strategy', 'AREA 2'))
    core_util = float(row.get('core_util', 55.0)) / 100.0
    density = float(row.get('density', 0.5))

    n_ff = df_f['n_ff']; n_active = df_f['n_active']
    die_area = df_f['die_area']; ff_hpwl = df_f['ff_hpwl']; ff_spacing = df_f['ff_spacing']
    avg_ds = df_f['avg_ds']; frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
    comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
    n_nets = sf_f['n_nets']; rel_act = sf_f['rel_act']
    cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
    mw = row['cts_max_wire']; bd = row['cts_buf_dist']
    phys_pw = max(rel_act * n_nets * f_hz, 1e-20)

    fanout_proxy = n_nets / (n_active + 1)
    nets_per_ff = n_nets / (n_ff + 1)
    xor_adj_act = rel_act / (1 + frac_xor * 3)
    xor_energy_proxy = frac_xor * avg_ds
    xor_heavy = 1.0 if frac_xor > 0.05 else 0.0
    driven_cap_per_ff = ef.get('driven_cap_per_ff', 0.0)
    driven_cap_cv = ef.get('driven_cap_cv', 0.0)
    driven_cap_p90 = ef.get('driven_cap_p90', 0.0)
    mst_per_ff = ef.get('mst_per_ff', 0.0)
    mst_norm = ef.get('mst_norm', 0.0)
    dens_gini = ef.get('dens_gini', 0.0)
    dens_entropy = ef.get('dens_entropy', 0.0)

    sm = tp_f['slack_mean']
    fn = tp_f['frac_neg']
    ft = tp_f['frac_tight']

    feats = [
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
        df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
        df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
        avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
        np.log1p(df_f['cap_proxy']),
        rel_act, sf_f['mean_sig_prob'], sf_f['tc_std_norm'], sf_f['frac_zero'],
        sf_f['frac_high_act'], sf_f['log_n_nets'], n_nets / (n_ff + 1),
        f_ghz, t_clk, synth_delay, synth_level, synth_agg, core_util, density,
        np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
        frac_xor * comb_per_ff, rel_act * frac_xor, rel_act * (1 - df_f['frac_ff_active']),
        synth_delay * avg_ds, synth_agg * f_ghz,
        np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
        np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs), core_util * density,
        np.log1p(n_active * rel_act * f_ghz), np.log1p(frac_xor * n_active),
        np.log1p(frac_mux * n_active), np.log1p(comb_per_ff * n_ff),
        np.log1p(die_area / (n_ff + 1)), np.log1p(n_comb), comb_per_ff * np.log1p(n_ff),
        # Timing
        sm, tp_f['slack_std'], tp_f['slack_min'], tp_f['slack_p10'], tp_f['slack_p50'],
        fn, ft, tp_f['frac_critical'],
        tp_f['n_paths'] / (n_ff + 1),
        sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
        1.0 if sm > 1.5 else 0.0, 1.0 if sm > 2.0 else 0.0, 1.0 if sm > 3.0 else 0.0,
        np.log1p(sm), sm * f_ghz,
        # New v16 features
        fanout_proxy, np.log1p(fanout_proxy), nets_per_ff, np.log1p(nets_per_ff),
        xor_adj_act, xor_energy_proxy, xor_heavy,
        frac_xor * fanout_proxy, rel_act * fanout_proxy,
        np.log(phys_pw + 1e-30), np.log(max(np.sqrt(n_active * die_area), 1e-3) + 1e-3),
        np.log1p(driven_cap_per_ff), driven_cap_cv, np.log1p(driven_cap_p90),
        np.log1p(mst_per_ff), mst_norm, dens_gini, dens_entropy,
        # GRAVITY FEATURES FOR POWER (v16b — test if helps)
        gf.get('grav_abs_mean', 0.0), gf.get('grav_abs_cv', 0.0),
        gf.get('grav_norm_mean', 0.0), gf.get('tp_degree_cv', 0.0),
        gf.get('tp_frac_involved', 0.0), gf.get('tp_paths_per_ff', 0.0),
    ]
    rows_pw_with_grav.append(feats)

# Build X_pw_grav matching the same rows as X_pw
X_pw_grav = np.array(rows_pw_with_grav, dtype=np.float64)
for c in range(X_pw_grav.shape[1]):
    bad = ~np.isfinite(X_pw_grav[:, c])
    if bad.any():
        X_pw_grav[bad, c] = np.nanmedian(X_pw_grav[~bad, c]) if (~bad).any() else 0.0

print(f"  X_pw_grav shape: {X_pw_grav.shape}")

# Only run if shapes match
if len(X_pw_grav) == len(X_pw):
    for alpha in [100, 1000, 5000]:
        lodo_power(X_pw_grav, y_pw, meta_df, f"v16b+grav Ridge(α={alpha})", Ridge,
                   {'alpha': float(alpha), 'max_iter': 10000})

    lodo_power(X_pw_grav, y_pw, meta_df, "v16b+grav XGB(300,d=4)", XGBRegressor,
               {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0, 'n_jobs': 2})

# -----------------------------------------------------------------------
# PER-DESIGN ANALYSIS: k_PA values (physics constants)
# -----------------------------------------------------------------------

print(f"\n{T()} === PHYSICS CONSTANT ANALYSIS ===")
sys.stdout.flush()

for design in sorted(meta_df['design_name'].unique()):
    mask = meta_df['design_name'] == design
    pw_true = meta_df[mask]['power_total'].values
    phys = meta_df[mask]['phys_pw'].values
    k_pa = pw_true / phys
    wl_true = meta_df[mask]['wirelength'].values
    phys_wl_v = meta_df[mask]['phys_wl'].values
    k_wa = wl_true / phys_wl_v
    print(f"  {design}: k_PA = {k_pa.mean():.3e} ± {k_pa.std():.3e}  "
          f"(CV={k_pa.std()/k_pa.mean():.3f})   "
          f"k_WA = {k_wa.mean():.3f} ± {k_wa.std():.3f}")

sys.stdout.flush()

# -----------------------------------------------------------------------
# ORACLE MAPE: best achievable with optimal per-design constant
# -----------------------------------------------------------------------

print(f"\n{T()} === ORACLE MAPE (per-design k* from test data) ===")
sys.stdout.flush()

for design in sorted(meta_df['design_name'].unique()):
    mask = meta_df['design_name'] == design
    pw_true = meta_df[mask]['power_total'].values
    phys = meta_df[mask]['phys_pw'].values
    k_star = pw_true.mean() / phys.mean()  # optimal constant
    pred_pw = k_star * phys
    oracle_pw_mape = mape(pw_true, pred_pw)

    wl_true = meta_df[mask]['wirelength'].values
    phys_wl_v = meta_df[mask]['phys_wl'].values
    k_star_wl = wl_true.mean() / phys_wl_v.mean()
    pred_wl = k_star_wl * phys_wl_v
    oracle_wl_mape = mape(wl_true, pred_wl)

    print(f"  {design}: Oracle power MAPE = {oracle_pw_mape:.1f}%  "
          f"Oracle WL MAPE = {oracle_wl_mape:.1f}%")

print(f"\n{T()} DONE")
