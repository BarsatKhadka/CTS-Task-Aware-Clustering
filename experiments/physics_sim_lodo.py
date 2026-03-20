"""
physics_sim_lodo.py — Physics-Simulated CTS Clustering for Zero-Shot LODO

CORE IDEA (NOVEL):
For each (placement, CTS_run) pair, simulate the CTS clustering algorithm
using FF positions + knob values. Physics-exact features:
  - sim_n_clusters  ≈ clock_buffers ≈ power  (empirically rho ≈ 0.87-0.96)
  - sim_total_wl    ≈ wirelength             (rho ≈ 0.77-0.93)
  - sim_skew_proxy  = spread of cluster HWPLs (rho ≈ -0.3 to 0.2 — noisy)

SECONDARY: Gaussian Process (ARD RBF) on sim+knob features for rank pred.
GP interpolates rather than extrapolates → better LODO behavior than LGB.

Leverages existing cache_v2_fixed.pkl for X29 features.

Expected: power MAE < 0.03, WL MAE < 0.04 (vs current 0.0656 / 0.0849)
"""

import re, os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
FF_POS_CACHE  = f'{BASE}/ff_positions_cache.pkl'
SIM_CACHE     = f'{BASE}/sim_features_cache.pkl'

t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"


# -----------------------------------------------------------------------
# STEP 1: FF position cache
# -----------------------------------------------------------------------

def parse_ff_positions(def_path):
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
        return None, None, None

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content)
    if not die_m:
        return None, None, None
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_w, die_h = x1 - x0, y1 - y0

    ff_pattern = re.compile(
        r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)')
    ff_xy = [(float(x)/units - x0, float(y)/units - y0)
             for _, x, y in ff_pattern.findall(content)]

    if not ff_xy:
        return None, die_w, die_h

    ff_arr = np.array(ff_xy, dtype=np.float32)  # absolute µm from die corner
    ff_norm = np.column_stack([ff_arr[:, 0] / (die_w + 1e-9),
                                ff_arr[:, 1] / (die_h + 1e-9)])
    return ff_norm, die_w, die_h


def get_ff_pos_cache(df):
    if os.path.exists(FF_POS_CACHE):
        with open(FF_POS_CACHE, 'rb') as f:
            cache = pickle.load(f)
        missing = [p for p in df['placement_id'].unique() if p not in cache]
    else:
        cache, missing = {}, list(df['placement_id'].unique())

    if missing:
        print(f"{T()} Parsing FF positions for {len(missing)} placements...")
        for i, pid in enumerate(missing):
            row = df[df['placement_id'] == pid].iloc[0]
            def_path = row['def_path'].replace('../dataset_with_def/placement_files',
                                               PLACEMENT_DIR)
            ff_norm, dw, dh = parse_ff_positions(def_path)
            cache[pid] = {'ff_norm': ff_norm, 'die_w': dw, 'die_h': dh} \
                if ff_norm is not None else None
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(missing)}")
        with open(FF_POS_CACHE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"{T()} Saved FF position cache: {len(cache)} placements")
    else:
        n_ok = sum(1 for v in cache.values() if v is not None)
        print(f"{T()} Loaded FF position cache: {n_ok}/{len(cache)} valid")
    return cache


# -----------------------------------------------------------------------
# STEP 2: CTS clustering simulation
# -----------------------------------------------------------------------

def simulate_cts(ff_xy_um, cluster_dia, cluster_size):
    """
    Greedy CTS clustering simulation.
    Returns dict of physics-derived statistics per (placement, run).
    """
    n = len(ff_xy_um)
    if n == 0:
        return _empty_sim()

    tree = cKDTree(ff_xy_um)
    assigned = np.zeros(n, dtype=bool)
    clusters = []

    order = np.lexsort((ff_xy_um[:, 1], ff_xy_um[:, 0]))  # scanline order
    radius = cluster_dia / 2.0

    for i in order:
        if assigned[i]:
            continue
        nbrs = tree.query_ball_point(ff_xy_um[i], r=radius)
        avail = [j for j in nbrs if not assigned[j]]

        if len(avail) > cluster_size:
            # Keep closest cluster_size to centroid
            pts = ff_xy_um[avail]
            cx, cy = pts.mean(axis=0)
            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            avail = [avail[k] for k in np.argsort(dists)[:cluster_size]]

        clusters.append(avail)
        for j in avail:
            assigned[j] = True

    n_cl = len(clusters)
    hpwls, sizes, centroids = [], [], []

    for idxs in clusters:
        pts = ff_xy_um[idxs]
        sizes.append(len(pts))
        if len(pts) <= 1:
            hpwls.append(0.0)
            centroids.append(pts[0] if len(pts) else np.zeros(2))
        else:
            hpwls.append(
                (pts[:, 0].max() - pts[:, 0].min()) +
                (pts[:, 1].max() - pts[:, 1].min()))
            centroids.append(pts.mean(axis=0))

    hpwls = np.asarray(hpwls)
    sizes  = np.asarray(sizes,  dtype=float)
    centroids = np.asarray(centroids)

    inter_hpwl = 0.0
    if len(centroids) > 1:
        inter_hpwl = ((centroids[:, 0].max() - centroids[:, 0].min()) +
                      (centroids[:, 1].max() - centroids[:, 1].min()))

    intra_wl   = 1.5 * hpwls.sum()
    inter_wl   = inter_hpwl * np.log1p(n_cl)
    total_wl   = intra_wl + inter_wl

    skew_proxy = (hpwls.max() - hpwls.min()) if len(hpwls) > 1 else 0.0
    skew_cv    = hpwls.std() / (hpwls.mean() + 1e-9)

    return {
        'sim_n_clusters':    float(n_cl),
        'sim_sum_hpwl':      float(hpwls.sum()),
        'sim_mean_hpwl':     float(hpwls.mean()),
        'sim_max_hpwl':      float(hpwls.max() if len(hpwls) else 0.0),
        'sim_min_hpwl':      float(hpwls.min() if len(hpwls) else 0.0),
        'sim_std_hpwl':      float(hpwls.std() if len(hpwls) > 1 else 0.0),
        'sim_skew_proxy':    float(skew_proxy),
        'sim_skew_cv':       float(skew_cv),
        'sim_inter_hpwl':    float(inter_hpwl),
        'sim_intra_wl':      float(intra_wl),
        'sim_total_wl':      float(total_wl),
        'sim_fill_rate':     float(n / (n_cl * cluster_size + 1e-9)),
        'sim_mean_cs':       float(sizes.mean()),
        'sim_frac_single':   float((sizes == 1).mean()),
    }


def _empty_sim():
    return {k: 0.0 for k in [
        'sim_n_clusters', 'sim_sum_hpwl', 'sim_mean_hpwl', 'sim_max_hpwl',
        'sim_min_hpwl', 'sim_std_hpwl', 'sim_skew_proxy', 'sim_skew_cv',
        'sim_inter_hpwl', 'sim_intra_wl', 'sim_total_wl', 'sim_fill_rate',
        'sim_mean_cs', 'sim_frac_single']}


def get_sim_cache(df, ff_pos_cache):
    if os.path.exists(SIM_CACHE):
        with open(SIM_CACHE, 'rb') as f:
            sc = pickle.load(f)
        missing_runs = set(df['run_id'].values) - set(sc.keys())
    else:
        sc, missing_runs = {}, set(df['run_id'].values)

    if missing_runs:
        df_miss = df[df['run_id'].isin(missing_runs)]
        print(f"{T()} Simulating {len(df_miss)} (placement, run) pairs...")
        for i, (_, row) in enumerate(df_miss.iterrows()):
            pid = row['placement_id']
            pos = ff_pos_cache.get(pid)
            if pos is None:
                sc[row['run_id']] = _empty_sim()
                continue
            ff_um = pos['ff_norm'] * np.array([[pos['die_w'], pos['die_h']]])
            sc[row['run_id']] = simulate_cts(
                ff_um, float(row['cts_cluster_dia']), int(row['cts_cluster_size']))
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(df_miss)}")
        with open(SIM_CACHE, 'wb') as f:
            pickle.dump(sc, f)
        print(f"{T()} Saved sim cache: {len(sc)} entries")
    else:
        print(f"{T()} Loaded sim cache: {len(sc)} entries")
    return sc


# -----------------------------------------------------------------------
# STEP 3: Feature building (on top of existing cache_v2_fixed.pkl)
# -----------------------------------------------------------------------

def rank_within(vals):
    """Fractional rank: 0=lowest, 1=highest, size-1 denominator."""
    n = len(vals)
    return np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)


def build_sim_features(df, sim_cache):
    """
    Build per-run simulation feature matrix.
    Returns X_sim [n, 14] (raw + rank + centered per placement).
    """
    run_ids = df['run_id'].values
    pids = df['placement_id'].values
    n = len(df)

    sim_keys = ['sim_n_clusters', 'sim_sum_hpwl', 'sim_total_wl',
                'sim_skew_proxy', 'sim_fill_rate', 'sim_skew_cv', 'sim_frac_single']

    # Raw sim values
    raw = np.array([[sim_cache.get(r, _empty_sim()).get(k, 0.0)
                     for k in sim_keys]
                    for r in run_ids], dtype=np.float32)

    # Global maxima for normalization
    g_max = raw.max(axis=0) + 1e-9

    raw_norm = raw / g_max  # [n, 7] normalized raw

    # Per-placement rank and centered
    rank_arr    = np.zeros_like(raw_norm)
    centered_arr = np.zeros_like(raw_norm)

    for pid in np.unique(pids):
        m = pids == pid
        idxs = np.where(m)[0]
        for j in range(raw.shape[1]):
            v = raw[idxs, j]
            if v.max() > v.min():
                rank_arr[idxs, j] = rank_within(v)
            else:
                rank_arr[idxs, j] = 0.5
            centered_arr[idxs, j] = (v - v.mean()) / (g_max[j])

    # Combined: raw_norm + rank + centered = 21 features
    X_sim = np.hstack([raw_norm, rank_arr, centered_arr])  # [n, 21]
    return X_sim


def zscore_to_rank_mae(y_pred_z, y_true_z, pids):
    """Convert z-score predictions to rank within placement, compute MAE."""
    all_pred_r = np.zeros(len(pids))
    all_true_r = np.zeros(len(pids))
    for pid in np.unique(pids):
        m = pids == pid
        idxs = np.where(m)[0]
        if len(idxs) > 1:
            all_pred_r[idxs] = rank_within(y_pred_z[idxs])
            all_true_r[idxs] = rank_within(y_true_z[idxs])
        else:
            all_pred_r[idxs] = 0.5
            all_true_r[idxs] = 0.5
    return mean_absolute_error(all_true_r, all_pred_r)


# -----------------------------------------------------------------------
# STEP 4: LODO evaluation
# -----------------------------------------------------------------------

def lodo(X, y_sk_r, y_pw_z, y_wl_z, pids, designs_arr,
         sk_cls, sk_kw, pw_cls, pw_kw, wl_cls, wl_kw, name=""):
    """LODO evaluation: hold out each design, evaluate on unseen design."""
    design_list = sorted(np.unique(designs_arr))
    sk_maes, pw_maes, wl_maes = [], [], []

    for held in design_list:
        tr = designs_arr != held
        te = designs_arr == held

        for target, cls, kw, y, maes, label in [
            ('sk', sk_cls, sk_kw, y_sk_r, sk_maes, 'sk'),
            ('pw', pw_cls, pw_kw, y_pw_z, pw_maes, 'pw'),
            ('wl', wl_cls, wl_kw, y_wl_z, wl_maes, 'wl'),
        ]:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            m = cls(**kw)
            m.fit(Xtr, y[tr])
            pred = m.predict(Xte)

            if target == 'sk':
                mae = zscore_to_rank_mae(pred, y[te], pids[te])
            else:
                mae = zscore_to_rank_mae(pred, y[te], pids[te])
            maes.append(mae)

        print(f"  {held}: sk={sk_maes[-1]:.4f}  pw={pw_maes[-1]:.4f}  wl={wl_maes[-1]:.4f}")

    m_sk = np.mean(sk_maes)
    m_pw = np.mean(pw_maes)
    m_wl = np.mean(wl_maes)
    s_pk = 'PASS' if m_pw < 0.10 else 'FAIL'
    s_wl = 'PASS' if m_wl < 0.10 else 'FAIL'
    s_sk = 'PASS' if m_sk < 0.10 else 'FAIL'
    print(f"  [{name}] MEAN: sk={m_sk:.4f}{s_sk}  pw={m_pw:.4f}{s_pk}  wl={m_wl:.4f}{s_wl}\n")
    return m_sk, m_pw, m_wl


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Physics-Simulated CTS + Gaussian Kernel LODO")
    print("=" * 70)

    # --- Load base data ---
    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    print(f"{T()} Dataset: {len(df)} rows")

    with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
        cache = pickle.load(f)
    X_cache = cache['X']
    Y_cache = cache['Y']   # [5390, 3]: per-placement z-scored (sk, pw, wl)
    df_cache = cache['df']

    pids     = df_cache['placement_id'].values
    designs  = df_cache['design_name'].values
    y_sk_z   = Y_cache[:, 0].astype(np.float32)
    y_pw_z   = Y_cache[:, 1].astype(np.float32)
    y_wl_z   = Y_cache[:, 2].astype(np.float32)

    # Build rank targets from z-scores (for skew)
    y_sk_r = np.zeros(len(pids), dtype=np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        idxs = np.where(m)[0]
        y_sk_r[idxs] = rank_within(y_sk_z[idxs])

    # X29 features (from existing cache, cols 72-75 = z-knobs plus surrounding)
    # Build_X29 from overnight_best.py approach:
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    place_cols = ['core_util', 'density', 'aspect_ratio']
    Xraw  = df_cache[knob_cols].values.astype(np.float32)
    Xplc  = df_cache[place_cols].values.astype(np.float32)
    Xkz   = X_cache[:, 72:76]  # z-scored knobs
    raw_max = Xraw.max(axis=0) + 1e-6

    Xrank = np.zeros((len(pids), 4), np.float32)
    Xcent = np.zeros((len(pids), 4), np.float32)
    Xrange = np.zeros((len(pids), 4), np.float32)
    Xmean  = np.zeros((len(pids), 4), np.float32)

    for pid in np.unique(pids):
        m = pids == pid
        idxs = np.where(m)[0]
        for j in range(4):
            v = Xraw[idxs, j]
            Xrank[idxs, j]  = rank_within(v)
            Xcent[idxs, j]  = (v - v.mean()) / raw_max[j]
            Xrange[idxs, j] = v.std() / raw_max[j]
            Xmean[idxs, j]  = v.mean() / raw_max[j]

    Xplc_norm = Xplc / Xplc.max(axis=0)
    cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]
    util = Xplc[:, 0] / 100.0; density = Xplc[:, 1]; aspect = Xplc[:, 2]

    Xinter = np.column_stack([
        cd * util, mw * density, cd / (density + 0.01),
        cd * aspect, Xrank[:, 3] * util, Xrank[:, 2] * util
    ])

    X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_norm, Xinter, Xrange, Xmean])
    print(f"{T()} X29 shape: {X29.shape}")

    # --- Build simulation features ---
    print(f"{T()} Loading/building FF position cache...")
    ff_cache = get_ff_pos_cache(df)
    print(f"{T()} Loading/building simulation cache...")
    sim_cache = get_sim_cache(df_cache, ff_cache)

    print(f"{T()} Building simulation feature matrix...")
    X_sim = build_sim_features(df_cache, sim_cache)
    print(f"{T()} X_sim shape: {X_sim.shape}")

    X_full = np.hstack([X29, X_sim])  # X29 (29) + Xsim (21) = X50
    print(f"{T()} X_full shape: {X_full.shape}")

    # --- Sanity check: correlation between sim oracle and targets ---
    print(f"\n{T()} === SIMULATION ORACLE ANALYSIS ===")
    print("  Direct oracle rank MAE (sim feature directly as predictor):")
    sim_keys_raw = ['sim_n_clusters', 'sim_sum_hpwl', 'sim_total_wl', 'sim_skew_proxy']
    targets = [('power_total', y_pw_z), ('wirelength', y_wl_z),
               ('wirelength', y_wl_z), ('skew_setup', y_sk_z)]

    for sk, (target_name, y_true) in zip(sim_keys_raw, targets):
        oracle_raw = np.array([sim_cache.get(r, _empty_sim()).get(sk, 0.0)
                               for r in df_cache['run_id'].values])
        mae = zscore_to_rank_mae(oracle_raw, y_true, pids)
        rhos = []
        for pid in np.unique(pids):
            m = pids == pid
            idxs = np.where(m)[0]
            if len(idxs) < 3 or oracle_raw[idxs].std() < 1e-9:
                continue
            rho, _ = spearmanr(oracle_raw[idxs], y_true[idxs])
            rhos.append(rho)
        print(f"  {sk:<25} → {target_name}: oracle_MAE={mae:.4f}  median_rho={np.median(rhos):.3f}")

    LGB_SK = dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                  min_child_samples=15, verbose=-1)
    LGB_PW = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    XGB_WL = dict(n_estimators=1000, max_depth=6, learning_rate=0.01,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0)
    XGB_SK = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0)

    print(f"\n{T()} === EXP 1: X29 BASELINE (prior best) ===")
    lodo(X29, y_sk_r, y_pw_z, y_wl_z, pids, designs,
         XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW,
         XGBRegressor, XGB_WL, name="X29_baseline")

    print(f"\n{T()} === EXP 2: Xsim ONLY (pure simulation oracle) ===")
    lodo(X_sim, y_sk_r, y_pw_z, y_wl_z, pids, designs,
         LGBMRegressor, LGB_SK, LGBMRegressor, LGB_PW,
         LGBMRegressor, dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                             min_child_samples=15, verbose=-1),
         name="Xsim_only")

    print(f"\n{T()} === EXP 3: X_full = X29 + Xsim (combined) ===")
    lodo(X_full, y_sk_r, y_pw_z, y_wl_z, pids, designs,
         XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW,
         XGBRegressor, XGB_WL, name="X29+Xsim")

    # --- EXP 4: Sim features rank-only (most design-invariant subset) ---
    # rank_sim_n_clusters (power proxy) + rank_sim_total_wl (WL proxy)
    # + rank_sim_skew_proxy (skew proxy) + z_knobs
    idx_rank_start = 7   # in X_sim: first 7 = raw_norm, next 7 = rank
    X_minimal = np.hstack([
        X29[:, :4],              # z-knobs only
        X_sim[:, idx_rank_start:idx_rank_start+7],  # rank features from sim
    ])
    print(f"\n{T()} === EXP 4: Minimal (z-knobs + sim ranks only) ===")
    lodo(X_minimal, y_sk_r, y_pw_z, y_wl_z, pids, designs,
         LGBMRegressor, LGB_SK, LGBMRegressor, LGB_PW,
         LGBMRegressor, dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                             min_child_samples=15, verbose=-1),
         name="minimal_z+simrank")

    # --- EXP 5: Gaussian Process (ARD RBF) on minimal features ---
    print(f"\n{T()} === EXP 5: Gaussian Process (ARD RBF kernel) ===")
    print("  Using z-knobs + sim rank features (11 dims) → GP regression")
    design_list = sorted(np.unique(designs))
    pw_gp_maes, wl_gp_maes = [], []

    for held in design_list:
        tr = designs != held
        te = designs == held

        tr_idx = np.where(tr)[0]
        if len(tr_idx) > 2000:
            rng = np.random.default_rng(42)
            tr_idx = rng.choice(tr_idx, 2000, replace=False)

        n_feat = X_minimal.shape[1]
        kernel = (C(1.0, (0.1, 10.0)) *
                  RBF(length_scale=np.ones(n_feat), length_scale_bounds=(0.01, 100.0)) +
                  WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.5)))

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_minimal[tr_idx])
        Xte = sc.transform(X_minimal[te])

        for y, maes, label in [(y_pw_z, pw_gp_maes, 'pw'), (y_wl_z, wl_gp_maes, 'wl')]:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1,
                                           normalize_y=True, alpha=1e-4)
            gp.fit(Xtr, y[tr_idx])
            pred = gp.predict(Xte)
            mae = zscore_to_rank_mae(pred, y[te], pids[te])
            maes.append(mae)

        print(f"  GP {held}: pw={pw_gp_maes[-1]:.4f}  wl={wl_gp_maes[-1]:.4f}")

    print(f"  [GP ARD RBF] MEAN: pw={np.mean(pw_gp_maes):.4f}  wl={np.mean(wl_gp_maes):.4f}\n")

    # --- EXP 6: Simulation-first LGB (predict using sim features as primary signal) ---
    # Stack: sim_rank features as primary, add X29 as corrector
    print(f"\n{T()} === EXP 6: Sim-primary with X29 as corrector ===")
    # For power: primary = rank_sim_n_clusters, then add residual features
    # For WL: primary = rank_sim_total_wl, then add residual features
    X_pw_primary = np.hstack([
        X_sim[:, idx_rank_start:idx_rank_start+1],  # rank_sim_n_clusters
        X_sim[:, idx_rank_start+2:idx_rank_start+3],  # rank_sim_total_wl
        X29[:, :4],                                   # z-knobs
        X29[:, 4:8],                                  # rank-knobs
    ])
    print(f"  X_pw_primary shape: {X_pw_primary.shape}")
    design_list2 = sorted(np.unique(designs))
    pw6, wl6 = [], []
    for held in design_list2:
        tr = designs != held; te = designs == held
        for y, maes in [(y_pw_z, pw6), (y_wl_z, wl6)]:
            sc = StandardScaler()
            m = LGBMRegressor(n_estimators=500, num_leaves=15, learning_rate=0.03,
                               min_child_samples=10, verbose=-1)
            m.fit(sc.fit_transform(X_pw_primary[tr]), y[tr])
            pred = m.predict(sc.transform(X_pw_primary[te]))
            maes.append(zscore_to_rank_mae(pred, y[te], pids[te]))
        print(f"  {held}: pw={pw6[-1]:.4f}  wl={wl6[-1]:.4f}")
    print(f"  [sim_primary] MEAN: pw={np.mean(pw6):.4f}  wl={np.mean(wl6):.4f}\n")

    print("=" * 70)
    print(f"{T()} SUMMARY")
    print("Target: sk<0.10✓  pw<0.10✓  wl<0.10✓  (LODO rank MAE)")
    print("Prior best: sk=0.2372✗  pw=0.0656✓  wl=0.0849✓")
    print("=" * 70)


if __name__ == '__main__':
    main()
