"""
skew_v2_spatial.py — T1-D: Systematic Skew Prediction with Spatial Critical-Path Features

Per-placement z-score normalization.
LODO evaluation on 4 designs.

Features:
  - Base aggregate features (from DEF cache)
  - CTS knobs + interactions
  - Critical-path spatial features (from skew_spatial_cache.pkl):
    * crit_mean/max/p90 dist (launch-capture FF physical distance)
    * crit_ff_hpwl, centroid offset, spread, boundary fraction
    * star/chain topology features
    * asymmetry, eccentricity
  - Interactions: CTS knobs × spatial features (the key physics)
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'

DEF_CACHE    = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE   = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE = f'{BASE}/absolute_v7_timing_cache.pkl'
SKEW_CACHE   = f'{BASE}/skew_spatial_cache.pkl'


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def build_skew_features(df_in, dc, sc_cache, tc, skc):
    """Build features for skew prediction."""
    rows, y, meta = [], [], []

    for _, row in df_in.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf = sc_cache.get(pid)
        tf = tc.get(pid)
        sk = skc.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        skew = row['skew_setup']
        if not np.isfinite(skew):
            continue

        n_ff = df_f['n_ff']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        comb_per_ff = df_f['comb_per_ff']
        rel_act = sf['rel_act']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        sm = tf['slack_mean']
        fn = tf['frac_neg']
        ft = tf['frac_tight']

        # Spatial features from critical paths
        crit_max  = sk.get('crit_max_dist', 0.0)
        crit_mean = sk.get('crit_mean_dist', 0.0)
        crit_p90  = sk.get('crit_p90_dist', 0.0)
        crit_hpwl = sk.get('crit_ff_hpwl', 0.0)
        crit_cx   = sk.get('crit_cx_offset', 0.0)
        crit_cy   = sk.get('crit_cy_offset', 0.0)
        crit_xs   = sk.get('crit_x_std', 0.0)
        crit_ys   = sk.get('crit_y_std', 0.0)
        crit_bnd  = sk.get('crit_frac_boundary', 0.0)
        crit_star = sk.get('crit_star_degree', 0.0)
        crit_chn  = sk.get('crit_chain_frac', 0.0)
        crit_asym = sk.get('crit_asymmetry', 0.0)
        crit_ecc  = sk.get('crit_eccentricity', 1.0)
        crit_dens = sk.get('crit_density_ratio', 1.0)
        crit_max_um  = sk.get('crit_max_dist_um', ff_hpwl)
        crit_mean_um = sk.get('crit_mean_dist_um', ff_hpwl / 2)

        # Physics-motivated interactions (THE KEY FEATURES FOR SKEW)
        # cluster_dia vs FF spacing: determines groupability
        cd_over_ff_spacing = cd / (ff_spacing + 1)
        # buf_dist vs max critical path: buffer steps available
        bd_over_crit_max = bd / (crit_max_um + 1)
        # max_wire vs critical path HPWL: wire budget
        mw_over_crit_hpwl = mw / (crit_max_um + 1)
        # cluster_dia vs critical FF spread
        cd_over_crit_spread = cd / (max(crit_xs, crit_ys) * max(die_area**0.5, 1) + 1)
        # Star topology × cluster_dia: star pattern = all paths from one FF = hard to balance
        star_times_cd = crit_star * cd
        # Asymmetry × max_wire: asymmetric placement needs longer wire to balance
        asym_times_mw = crit_asym * mw
        # Critical path density × cluster size
        dens_times_cs = crit_dens * cs

        feats = [
            # Placement geometry
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl),
            np.log1p(ff_spacing), df_f['die_aspect'],
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, comb_per_ff, avg_ds,
            rel_act, sf['mean_sig_prob'],

            # Timing stats
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'],
            fn, ft, tf['frac_critical'],
            np.log1p(tf['n_paths'] / (n_ff + 1)),

            # CTS knobs (raw + log)
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,

            # Critical path spatial features (15 dims)
            crit_max, crit_mean, crit_p90, crit_hpwl,
            crit_cx, crit_cy, crit_xs, crit_ys,
            crit_bnd, crit_star, crit_chn,
            crit_asym, crit_ecc, crit_dens,
            np.log1p(crit_max_um), np.log1p(crit_mean_um),

            # KEY INTERACTIONS: CTS knobs × spatial features
            cd_over_ff_spacing,
            bd_over_crit_max,
            mw_over_crit_hpwl,
            cd_over_crit_spread,
            star_times_cd,
            asym_times_mw,
            dens_times_cs,
            crit_max * cd,             # critical span × cluster_dia
            crit_asym * crit_max,      # asymmetry × max path
            fn * crit_star,            # frac_neg × star topology
            ft * crit_chn,             # frac_tight × chain topology
            crit_hpwl / (cs + 1),      # critical HPWL per cluster
            np.log1p(crit_max_um / (cd + 1)),
            np.log1p(crit_max_um / (bd + 1)),
            np.log1p(crit_max_um / (mw + 1)),
            crit_cx * cd, crit_cy * mw,
            np.log1p(n_ff / cs) * crit_hpwl,
        ]  # ~60 dims

        rows.append(feats)
        y.append(skew)
        meta.append({'placement_id': pid, 'design_name': design, 'skew': skew})

    X = np.array(rows, dtype=np.float64)
    y = np.array(y)
    meta_df = pd.DataFrame(meta)

    if X.ndim == 2 and X.shape[0] > 0:
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X, y, meta_df


def per_placement_normalize(y, meta_df):
    """Per-placement z-score normalization."""
    y_norm = np.zeros_like(y)
    mu_arr = np.zeros_like(y)
    sig_arr = np.ones_like(y)

    for pid, grp in meta_df.groupby('placement_id'):
        idx = grp.index.values
        vals = y[idx]
        mu = vals.mean()
        sig = vals.std()
        sig = max(sig, max(abs(mu) * 0.01, 1e-4))
        y_norm[idx] = (vals - mu) / sig
        mu_arr[idx] = mu
        sig_arr[idx] = sig

    return y_norm, mu_arr, sig_arr


def lodo_skew(X, y, meta_df, model_type='xgb'):
    designs = sorted(meta_df['design_name'].unique())
    results = {}

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        y_norm, mu_arr, sig_arr = per_placement_normalize(y, meta_df)

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        if model_type == 'xgb':
            m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0, n_jobs=1)
        elif model_type == 'lgb':
            m = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                              min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        else:
            m = Ridge(alpha=100.0)

        m.fit(Xtr, y_norm[tr])
        pred_norm = m.predict(Xte)
        # Denormalize
        pred_raw = pred_norm * sig_arr[te] + mu_arr[te]
        results[held] = mae(y[te], pred_raw)

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("Skew v2: Critical-Path Spatial Features + CTS Knob Interactions")
    print("=" * 70)
    sys.stdout.flush()

    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(SKEW_CACHE, 'rb') as f:   skc = pickle.load(f)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['skew_setup']).reset_index(drop=True)
    print(f"{T()} n={len(df)}, skew range=[{df['skew_setup'].min():.3f}, {df['skew_setup'].max():.3f}]")
    sys.stdout.flush()

    print(f"{T()} Building features...")
    X, y, meta_df = build_skew_features(df, dc, sc_cache, tc, skc)
    print(f"  X={X.shape}, {meta_df['design_name'].value_counts().to_dict()}")
    sys.stdout.flush()

    print(f"\n{T()} === LODO Skew MAE ===")
    print(f"  (Target: < 0.10, Previous best: 0.237)")
    sys.stdout.flush()

    for mtype in ['xgb', 'lgb']:
        res = lodo_skew(X, y, meta_df, model_type=mtype)
        vals = list(res.values())
        marker = ' ← HITS TARGET' if np.mean(vals) < 0.10 else ''
        print(f"  {mtype}: {' '.join(f'{d}={v:.4f}' for d,v in res.items())}  "
              f"mean={np.mean(vals):.4f}{marker}")
        sys.stdout.flush()

    # Also try ensemble XGB + Ridge
    print(f"\n{T()} === Feature importance (XGB) ===")
    y_norm, mu_arr, sig_arr = per_placement_normalize(y, meta_df)
    sc = StandardScaler()
    Xsc = sc.fit_transform(X)
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     verbosity=0, n_jobs=1)
    m.fit(Xsc, y_norm)
    imp = m.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    print(f"  Top 15 feature indices and importances:")
    for i in top_idx:
        print(f"    feat[{i:3d}] = {imp[i]:.4f}")

    print(f"\n{T()} DONE")
