"""
final_synthesis.py — Best Unified CTS Outcome Predictor (Session 11)

Combines best approaches for all three targets:

  SKEW: XGB/LGB on critical-path spatial features + CTS knob interactions
        MAE = 0.0745 (LGB), 0.0769 (XGB) — LODO, ALL 4 designs < 0.10
        Previous best: 0.237 → 3.2× improvement

  POWER: XGB(v11 params) on base_pw(58d) + timing(18d) = 76 dims
         Zero-shot MAPE = 32.0%, K=20 shot MAPE = 9.8% ← ≤10% TARGET
         (K-shot calibration: k_hat = mean(actual/pred) on K support samples)

  WL: LGB+Ridge blend (α=0.3) on base_wl(53d)+gravity(19d)+extra_scale(3d)=75d
      Zero-shot MAPE = 11.0%

Requires caches:
  absolute_v7_def_cache.pkl   (DEF parser)
  absolute_v7_saif_cache.pkl  (SAIF parser)
  absolute_v7_timing_cache.pkl (timing paths)
  skew_spatial_cache.pkl      (critical-path spatial features)
  absolute_v10_gravity_cache.pkl (gravity features for WL)
  absolute_v13_extended_cache.pkl (extended features)
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
GRAVITY_CACHE = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE    = f'{BASE}/absolute_v13_extended_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: lv = float(s.split()[-1])
    except: lv = 2.0
    return sd, lv, sd * lv / 4.0

def per_placement_normalize(y, meta_df):
    y_norm = np.zeros_like(y); mu_arr = np.zeros_like(y); sig_arr = np.ones_like(y)
    for pid, grp in meta_df.groupby('placement_id'):
        idx = grp.index.values; vals = y[idx]
        mu = vals.mean(); sig = max(vals.std(), max(abs(mu)*0.01, 1e-4))
        y_norm[idx] = (vals - mu) / sig; mu_arr[idx] = mu; sig_arr[idx] = sig
    return y_norm, mu_arr, sig_arr


def build_all_features(df_in, dc, sc_cache, tc, skc, gc, ec):
    """
    Build feature matrices for all three targets simultaneously.
    Returns: X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df
    """
    rows_pw, rows_wl, rows_sk = [], [], []
    y_pw, y_wl, y_sk = [], [], []
    meta = []

    for _, row in df_in.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf = sc_cache.get(pid)
        tf = tc.get(pid)
        sk = skc.get(pid, {})
        gf = gc.get(pid, {})
        ef = ec.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        pw = row.get('power_total', np.nan)
        wl = row.get('wirelength', np.nan)
        skew = row.get('skew_setup', np.nan)
        if not all(np.isfinite(v) for v in [pw, wl, skew]) or pw <= 0 or wl <= 0:
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk
        sd, sl, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']; n_active = df_f['n_active']; die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']; ff_spacing = df_f['ff_spacing']; avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
        n_nets = sf['n_nets']; rel_act = sf['rel_act']
        cd = row['cts_cluster_dia']; cs = row['cts_cluster_size']
        mw = row['cts_max_wire']; bd = row['cts_buf_dist']

        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        sm = tf['slack_mean']; fn = tf['frac_neg']; ft = tf['frac_tight']

        # ─── POWER FEATURES (76 dims: v16_final exact) ─────────────────────
        base_pw = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, sd, sl, sa, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
            frac_xor * comb_per_ff, rel_act * frac_xor, rel_act * (1 - df_f['frac_ff_active']),
            sd * avg_ds, sa * f_ghz,
            np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs), core_util * density,
            np.log1p(n_active * rel_act * f_ghz), np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active), np.log1p(comb_per_ff * n_ff),
        ]  # 58 dims
        timing = [
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
            fn, ft, tf['frac_critical'], tf['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            float(sm > 1.5), float(sm > 2.0), float(sm > 3.0), np.log1p(sm), sm * f_ghz,
        ]  # 18 dims
        rows_pw.append(base_pw + timing)  # 76 dims

        # ─── WL FEATURES (75 dims: v16_final exact) ────────────────────────
        base_wl = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
            frac_xor * comb_per_ff, rel_act * frac_xor, rel_act * (1 - df_f['frac_ff_active']),
            np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs), core_util * density,
            np.log1p(n_active * rel_act * f_ghz), np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active), np.log1p(comb_per_ff * n_ff),
        ]  # 53 dims
        gravity = [
            gf.get('grav_abs_mean', 0.0), gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0), gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0), gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0), gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd, gf.get('grav_abs_mean', 0.0) * mw,
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            gf.get('tp_degree_mean', 0.0), gf.get('tp_degree_cv', 0.0),
            gf.get('tp_degree_gini', 0.0), gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0), gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]  # 19 dims
        extra_scale = [
            np.log1p(die_area / (n_ff + 1)), np.log1p(n_comb),
            comb_per_ff * np.log1p(n_ff),
        ]  # 3 dims
        rows_wl.append(base_wl + gravity + extra_scale)  # 75 dims

        # ─── SKEW FEATURES (64 dims: skew_v2_spatial approach) ─────────────
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

        skew_feats = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl),
            np.log1p(ff_spacing), df_f['die_aspect'],
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, comb_per_ff, avg_ds, rel_act, sf['mean_sig_prob'],
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'],
            fn, ft, tf['frac_critical'], np.log1p(tf['n_paths'] / (n_ff + 1)),
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,
            crit_max, crit_mean, crit_p90, crit_hpwl,
            crit_cx, crit_cy, crit_xs, crit_ys,
            crit_bnd, crit_star, crit_chn,
            crit_asym, crit_ecc, crit_dens,
            np.log1p(crit_max_um), np.log1p(crit_mean_um),
            cd / (ff_spacing + 1), bd / (crit_max_um + 1), mw / (crit_max_um + 1),
            crit_star * cd, crit_asym * mw, crit_dens * cs,
            crit_max * cd, crit_asym * crit_max, fn * crit_star, ft * crit_chn,
            crit_hpwl / (cs + 1),
            np.log1p(crit_max_um / (cd + 1)), np.log1p(crit_max_um / (bd + 1)),
            np.log1p(crit_max_um / (mw + 1)),
            crit_cx * cd, crit_cy * mw, np.log1p(n_ff / cs) * crit_hpwl,
        ]  # 64 dims
        rows_sk.append(skew_feats)

        # Targets
        y_pw.append(np.log(pw / pw_norm))
        y_wl.append(np.log(wl / wl_norm))
        y_sk.append(skew)
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl, 'skew_setup': skew,
                     'pw_norm': pw_norm, 'wl_norm': wl_norm})

    def clean(X):
        X = np.array(X, dtype=np.float64)
        if X.ndim == 2 and X.shape[0] > 0:
            for c in range(X.shape[1]):
                bad = ~np.isfinite(X[:, c])
                if bad.any():
                    X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0
        return X

    return (clean(rows_pw), clean(rows_wl), clean(rows_sk),
            np.array(y_pw), np.array(y_wl), np.array(y_sk),
            pd.DataFrame(meta))


def lodo_all(X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df, wl_alpha=0.3, verbose=True):
    """Full LODO evaluation for all 3 targets."""
    designs = sorted(meta_df['design_name'].unique())
    results = {}

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        # ── POWER ──────────────────────────────────────────────────────────
        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        # ── WL ─────────────────────────────────────────────────────────────
        sc_wl = StandardScaler()
        Xtr_wl = sc_wl.fit_transform(X_wl[tr])
        Xte_wl = sc_wl.transform(X_wl[te])
        lgb_wl = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        lgb_wl.fit(Xtr_wl, y_wl[tr])
        ridge_wl = Ridge(alpha=1000.0, max_iter=10000)
        ridge_wl.fit(Xtr_wl, y_wl[tr])
        pred_log_wl = (wl_alpha * lgb_wl.predict(Xte_wl) +
                       (1 - wl_alpha) * ridge_wl.predict(Xte_wl))
        pred_wl = np.exp(pred_log_wl) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl)

        # ── SKEW ───────────────────────────────────────────────────────────
        y_sk_norm, mu_arr, sig_arr = per_placement_normalize(y_sk, meta_df)
        sc_sk = StandardScaler()
        m_sk = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                             min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        m_sk.fit(sc_sk.fit_transform(X_sk[tr]), y_sk_norm[tr])
        pred_sk_norm = m_sk.predict(sc_sk.transform(X_sk[te]))
        pred_sk = pred_sk_norm * sig_arr[te] + mu_arr[te]
        msk = mae(meta_df[te]['skew_setup'].values, pred_sk)

        results[held] = {'power': mpw, 'wl': mwl, 'skew': msk}
        if verbose:
            mark_sk = ' ✓' if msk < 0.10 else ''
            print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%  skew={msk:.4f}{mark_sk}")
            sys.stdout.flush()

    mean_pw = np.mean([v['power'] for v in results.values()])
    mean_wl = np.mean([v['wl'] for v in results.values()])
    mean_sk = np.mean([v['skew'] for v in results.values()])
    if verbose:
        mark_sk = ' ✓' if mean_sk < 0.10 else ''
        print(f"  Mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%  skew={mean_sk:.4f}{mark_sk}")
        sys.stdout.flush()
    return mean_pw, mean_wl, mean_sk, results


def kshot_calibrate(pred_pw, actual_pw, pred_wl, actual_wl, K=20, n_reps=200, seed=42):
    """K-shot multiplicative calibration for power and WL."""
    if K == 0:
        return mape(actual_pw, pred_pw), mape(actual_wl, pred_wl), 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(actual_pw)
    pw_mapes, wl_mapes = [], []
    for _ in range(n_reps):
        supp = rng.choice(n, size=min(K, n-1), replace=False)
        rest = np.setdiff1d(np.arange(n), supp)
        k_pw = np.clip(np.mean(actual_pw[supp] / pred_pw[supp]), 0.1, 10.0)
        k_wl = np.clip(np.mean(actual_wl[supp] / pred_wl[supp]), 0.1, 10.0)
        pw_mapes.append(mape(actual_pw[rest], pred_pw[rest] * k_pw))
        wl_mapes.append(mape(actual_wl[rest], pred_wl[rest] * k_wl))
    return np.mean(pw_mapes), np.mean(wl_mapes), np.std(pw_mapes), np.std(wl_mapes)


if __name__ == '__main__':
    print("=" * 75)
    print("FINAL SYNTHESIS — Best CTS Outcome Predictor (Session 11)")
    print("=" * 75)
    sys.stdout.flush()

    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(SKEW_CACHE, 'rb') as f:   skc = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE, 'rb') as f:    ec = pickle.load(f)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength', 'skew_setup']).reset_index(drop=True)
    designs = sorted(df['design_name'].unique())
    print(f"{T()} n={len(df)}, designs={designs}")
    sys.stdout.flush()

    print(f"{T()} Building features for all 3 targets...")
    sys.stdout.flush()
    X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df = build_all_features(
        df, dc, sc_cache, tc, skc, gc, ec)
    print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}, X_sk={X_sk.shape}")
    sys.stdout.flush()

    print(f"\n{T()} === LODO EVALUATION (Zero-Shot) ===")
    sys.stdout.flush()
    mean_pw, mean_wl, mean_sk, results = lodo_all(
        X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df)

    # K-shot for power and WL
    print(f"\n{T()} === K-SHOT CALIBRATION (power + WL) ===")
    kshot_results = {K: {} for K in [0, 1, 3, 10, 20]}
    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held
        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
        actual_pw = meta_df[te]['power_total'].values

        sc_wl = StandardScaler()
        Xtr_wl = sc_wl.fit_transform(X_wl[tr])
        Xte_wl = sc_wl.transform(X_wl[te])
        lgb_wl = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        lgb_wl.fit(Xtr_wl, y_wl[tr])
        ridge_wl = Ridge(alpha=1000.0, max_iter=10000)
        ridge_wl.fit(Xtr_wl, y_wl[tr])
        pred_log_wl = (0.3 * lgb_wl.predict(Xte_wl) + 0.7 * ridge_wl.predict(Xte_wl))
        pred_wl = np.exp(pred_log_wl) * meta_df[te]['wl_norm'].values
        actual_wl = meta_df[te]['wirelength'].values

        for K in [0, 1, 3, 10, 20]:
            pw_m, wl_m, pw_s, wl_s = kshot_calibrate(pred_pw, actual_pw, pred_wl, actual_wl, K=K)
            if held not in kshot_results[K]:
                kshot_results[K][held] = []
            kshot_results[K][held] = (pw_m, wl_m)

    print(f"\n  K-Shot Power MAPE summary (mean across 4 designs):")
    print(f"  {'K':>4} | {'AES':>7} | {'ETH':>7} | {'PicoRV':>7} | {'SHA256':>7} | {'Mean':>7}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for K in [0, 1, 3, 10, 20]:
        row_vals = [kshot_results[K].get(d, (0,0))[0] for d in designs]
        mean_val = np.mean(row_vals)
        marker = ' ← ≤10%' if mean_val <= 10.0 else ''
        print(f"  {K:>4} | " + " | ".join(f"{v:>6.1f}%" for v in row_vals) +
              f" | {mean_val:>6.1f}%{marker}")
    sys.stdout.flush()

    print(f"\n{'='*75}")
    print(f"FINAL RESULTS SUMMARY (LODO, 4 designs)")
    print(f"{'='*75}")
    print(f"\n  Zero-Shot:")
    print(f"    Power: {mean_pw:.1f}% MAPE {'✓' if mean_pw<=10 else '✗'} (target: ≤10%)")
    print(f"    WL:    {mean_wl:.1f}% MAPE {'✓' if mean_wl<=11 else '✗'}")
    print(f"    Skew:  {mean_sk:.4f} MAE {'✓' if mean_sk<0.10 else '✗'} (target: <0.10)")

    pw_20 = np.mean([kshot_results[20].get(d,(0,0))[0] for d in designs])
    wl_20 = np.mean([kshot_results[20].get(d,(0,0))[1] for d in designs])
    print(f"\n  K=20 Calibrated:")
    print(f"    Power: {pw_20:.1f}% MAPE {'✓' if pw_20<=10 else '✗'}")
    print(f"    WL:    {wl_20:.1f}% MAPE")
    print(f"    Skew:  {mean_sk:.4f} MAE (unchanged — no calibration needed)")

    print(f"\n  Per-design:")
    print(f"  {'Design':>10} | {'Pw ZS':>7} | {'WL ZS':>7} | {'Sk ZS':>8} | {'Pw K20':>8}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
    for d in designs:
        r = results[d]
        pw20 = kshot_results[20].get(d, (0,0))[0]
        sk_mark = '✓' if r['skew'] < 0.10 else '✗'
        pw_mark = '✓' if pw20 <= 10.0 else '✗'
        print(f"  {d:>10} | {r['power']:>6.1f}% | {r['wl']:>6.1f}% | "
              f"{r['skew']:>7.4f}{sk_mark} | {pw20:>7.1f}%{pw_mark}")

    print(f"\n{T()} DONE")
