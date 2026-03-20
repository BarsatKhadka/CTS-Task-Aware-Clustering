"""
absolute_v17_kshot.py — K-Shot Multiplicative Calibration on top of v16_final

Zero-shot base: power=32.0% / WL=11.0% mean MAPE (LODO, 4 designs)

K-shot calibration:
  After zero-shot prediction, observe K labeled samples from the test design.
  Compute k_hat = mean(actual[supp] / pred[supp])
  Apply  pred_cal = pred[rest] × k_hat  (multiplicative bias correction)

Expected targets:
  K=1  → ~13% power
  K=3  → ~10.5% power  (hits ≤10% target?)
  K=10 → ~10% power

Usage:
  python3 absolute_v17_kshot.py
"""

import re, os, sys, time
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

DEF_CACHE    = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE   = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE    = f'{BASE}/absolute_v13_extended_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    synth_delay = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return synth_delay, level, synth_delay * level / 4.0


def build_features(df_in, dc, sc_cache, tc, gc, ec):
    """Identical to v16_final — DO NOT MODIFY."""
    rows_pw, rows_wl = [], []
    y_pw, y_wl = [], []
    meta = []

    for _, row in df_in.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf = sc_cache.get(pid)
        tf = tc.get(pid)
        gf = gc.get(pid, {})
        ef = ec.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        pw = row['power_total']
        wl = row['wirelength']
        if not (np.isfinite(pw) and np.isfinite(wl) and pw > 0 and wl > 0):
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk
        sd, sl, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']
        n_comb = df_f['n_comb']
        n_nets = sf['n_nets']
        rel_act = sf['rel_act']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        sm = tf['slack_mean']
        fn = tf['frac_neg']
        ft = tf['frac_tight']

        # POWER BASE: 58 dims WITH synth (v11 exact)
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
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            sd * avg_ds, sa * f_ghz,
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
        ]  # 58 dims

        # WL BASE: 53 dims NO synth
        base = [
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
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
        ]  # 53 dims

        timing = [
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
            fn, ft, tf['frac_critical'],
            tf['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            float(sm > 1.5), float(sm > 2.0), float(sm > 3.0),
            np.log1p(sm), sm * f_ghz,
        ]  # 18 dims

        gravity = [
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
        ]  # 19 dims

        extra_scale = [
            np.log1p(die_area / (n_ff + 1)),
            np.log1p(n_comb),
            comb_per_ff * np.log1p(n_ff),
        ]  # 3 dims

        rows_pw.append(base_pw + timing)       # 76 dims
        rows_wl.append(base + gravity + extra_scale)  # 75 dims

        y_pw.append(np.log(pw / pw_norm))
        y_wl.append(np.log(wl / wl_norm))
        meta.append({
            'placement_id': pid, 'design_name': design,
            'power_total': pw, 'wirelength': wl,
            'pw_norm': pw_norm, 'wl_norm': wl_norm,
        })

    X_pw = np.array(rows_pw, dtype=np.float64)
    X_wl = np.array(rows_wl, dtype=np.float64)
    y_pw_arr = np.array(y_pw)
    y_wl_arr = np.array(y_wl)
    meta_df = pd.DataFrame(meta)

    for X in [X_pw, X_wl]:
        if X.ndim < 2 or X.shape[0] == 0: continue
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X_pw, X_wl, y_pw_arr, y_wl_arr, meta_df


def train_fold(held, X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3):
    """
    Train on all designs except held, return raw (W, µm) predictions for held design.
    Returns: pred_pw, actual_pw, pred_wl, actual_wl  (all in original units)
    """
    tr = meta_df['design_name'] != held
    te = meta_df['design_name'] == held

    # Power: XGB (v11 params)
    sc_pw = StandardScaler()
    m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        verbosity=0, n_jobs=1)
    m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
    pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values

    # WL: LGB + Ridge blend
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

    actual_pw = meta_df[te]['power_total'].values
    actual_wl = meta_df[te]['wirelength'].values

    return pred_pw, actual_pw, pred_wl, actual_wl


def kshot_eval(pred_pw, actual_pw, pred_wl, actual_wl,
               K_values=(0, 1, 2, 3, 5, 10, 20), n_reps=200, seed=42):
    """
    K-shot multiplicative calibration.

    For K=0: no calibration, return MAPE of pred vs actual directly.
    For K>0: repeat n_reps times:
        - sample K support indices uniformly at random (without replacement)
        - k_hat_pw = mean(actual_pw[supp] / pred_pw[supp])
        - k_hat_wl = mean(actual_wl[supp] / pred_wl[supp])
        - evaluate MAPE on remaining n-K samples with calibrated prediction
    Returns dict: K -> {pw_mean, pw_std, wl_mean, wl_std}
    """
    rng = np.random.default_rng(seed)
    n = len(actual_pw)
    results = {}

    for K in K_values:
        if K == 0:
            results[0] = {
                'pw_mean': mape(actual_pw, pred_pw),
                'pw_std': 0.0,
                'wl_mean': mape(actual_wl, pred_wl),
                'wl_std': 0.0,
            }
            continue

        if K >= n:
            continue

        pw_mapes, wl_mapes = [], []
        for _ in range(n_reps):
            supp = rng.choice(n, size=K, replace=False)
            rest = np.setdiff1d(np.arange(n), supp)

            k_hat_pw = np.mean(actual_pw[supp] / pred_pw[supp])
            k_hat_wl = np.mean(actual_wl[supp] / pred_wl[supp])

            # Clamp k_hat to reasonable range to avoid outlier-driven explosions
            k_hat_pw = np.clip(k_hat_pw, 0.1, 10.0)
            k_hat_wl = np.clip(k_hat_wl, 0.1, 10.0)

            pw_mapes.append(mape(actual_pw[rest], pred_pw[rest] * k_hat_pw))
            wl_mapes.append(mape(actual_wl[rest], pred_wl[rest] * k_hat_wl))

        results[K] = {
            'pw_mean': np.mean(pw_mapes),
            'pw_std':  np.std(pw_mapes),
            'wl_mean': np.mean(wl_mapes),
            'wl_std':  np.std(wl_mapes),
        }

    return results


def kshot_by_placement(pred_pw, actual_pw, pred_wl, actual_wl, meta_te,
                        K_placements=(1, 2, 3, 5), n_reps=200, seed=42):
    """
    Placement-level K-shot: use all 10 CTS runs from K placements as support,
    calibrate remaining placements.

    This is more realistic: in practice you'd tape out K placements of the new design
    and measure, then predict the rest.
    """
    rng = np.random.default_rng(seed + 1000)
    placement_ids = meta_te['placement_id'].values
    unique_placements = np.unique(placement_ids)
    n_placements = len(unique_placements)

    results = {}
    for Kp in K_placements:
        if Kp >= n_placements:
            continue
        pw_mapes, wl_mapes = [], []
        for _ in range(n_reps):
            supp_pids = rng.choice(unique_placements, size=Kp, replace=False)
            supp_mask = np.isin(placement_ids, supp_pids)
            rest_mask = ~supp_mask

            if rest_mask.sum() == 0:
                continue

            k_hat_pw = np.mean(actual_pw[supp_mask] / pred_pw[supp_mask])
            k_hat_wl = np.mean(actual_wl[supp_mask] / pred_wl[supp_mask])

            k_hat_pw = np.clip(k_hat_pw, 0.1, 10.0)
            k_hat_wl = np.clip(k_hat_wl, 0.1, 10.0)

            pw_mapes.append(mape(actual_pw[rest_mask], pred_pw[rest_mask] * k_hat_pw))
            wl_mapes.append(mape(actual_wl[rest_mask], pred_wl[rest_mask] * k_hat_wl))

        results[Kp] = {
            'pw_mean': np.mean(pw_mapes),
            'pw_std':  np.std(pw_mapes),
            'wl_mean': np.mean(wl_mapes),
            'wl_std':  np.std(wl_mapes),
        }
    return results


if __name__ == '__main__':
    print("=" * 75)
    print("K-Shot Multiplicative Calibration on top of Zero-Shot Absolute Predictor")
    print("Base: power=32.0%  WL=11.0%  (v16_final LODO)")
    print("=" * 75)
    sys.stdout.flush()

    # Load caches
    print(f"{T()} Loading caches...")
    sys.stdout.flush()
    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE, 'rb') as f:    ec = pickle.load(f)
    print(f"  DEF:{len(dc)} SAIF:{len(sc_cache)} Timing:{len(tc)} Gravity:{len(gc)} Ext:{len(ec)}")
    sys.stdout.flush()

    # Load data
    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    designs = sorted(df['design_name'].unique())
    print(f"{T()} Rows: {len(df)}, Designs: {designs}")
    sys.stdout.flush()

    # Build features
    print(f"{T()} Building features...")
    sys.stdout.flush()
    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df, dc, sc_cache, tc, gc, ec)
    print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}, n={len(meta_df)}")
    sys.stdout.flush()

    K_VALUES = (0, 1, 2, 3, 5, 10, 20, 50)
    K_PLACE  = (1, 2, 3, 5, 10)
    N_REPS   = 200

    # Per-design LODO + K-shot
    all_kshot = {K: {'pw': [], 'wl': []} for K in K_VALUES}
    all_kplace = {Kp: {'pw': [], 'wl': []} for Kp in K_PLACE}

    print(f"\n{T()} === LODO + K-SHOT CALIBRATION ===\n")
    sys.stdout.flush()

    for held in designs:
        te_mask = meta_df['design_name'] == held
        n_te = te_mask.sum()
        n_placements = meta_df[te_mask]['placement_id'].nunique()
        print(f"{'─'*75}")
        print(f"Held: {held}  ({n_te} samples, {n_placements} placements)")
        sys.stdout.flush()

        pred_pw, actual_pw, pred_wl, actual_wl = train_fold(
            held, X_pw, X_wl, y_pw, y_wl, meta_df)

        meta_te = meta_df[te_mask].reset_index(drop=True)

        # K-shot (random samples)
        ks_res = kshot_eval(pred_pw, actual_pw, pred_wl, actual_wl,
                            K_values=K_VALUES, n_reps=N_REPS)
        # K-shot (whole placements)
        kp_res = kshot_by_placement(pred_pw, actual_pw, pred_wl, actual_wl, meta_te,
                                     K_placements=K_PLACE, n_reps=N_REPS)

        print(f"\n  Random-sample calibration ({N_REPS} reps):")
        print(f"  {'K':>5} | {'Power MAPE':>12} | {'WL MAPE':>12}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}")
        for K, v in ks_res.items():
            marker = ' ←TARGET' if v['pw_mean'] <= 10.0 else ''
            print(f"  {K:>5} | {v['pw_mean']:>8.1f}% ±{v['pw_std']:>4.1f} | "
                  f"{v['wl_mean']:>8.1f}% ±{v['wl_std']:>4.1f}{marker}")
            all_kshot[K]['pw'].append(v['pw_mean'])
            all_kshot[K]['wl'].append(v['wl_mean'])
        sys.stdout.flush()

        print(f"\n  Placement-level calibration ({N_REPS} reps):")
        print(f"  {'Kp':>5} | {'Power MAPE':>12} | {'WL MAPE':>12}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}")
        for Kp, v in kp_res.items():
            marker = ' ←TARGET' if v['pw_mean'] <= 10.0 else ''
            print(f"  {Kp:>5} | {v['pw_mean']:>8.1f}% ±{v['pw_std']:>4.1f} | "
                  f"{v['wl_mean']:>8.1f}% ±{v['wl_std']:>4.1f}{marker}")
            all_kplace[Kp]['pw'].append(v['pw_mean'])
            all_kplace[Kp]['wl'].append(v['wl_mean'])
        sys.stdout.flush()

    # Summary table
    print(f"\n{'='*75}")
    print("MEAN ACROSS ALL 4 DESIGNS")
    print(f"{'='*75}")

    print(f"\n  Random-sample calibration (mean MAPE over {N_REPS} reps × 4 designs):")
    print(f"  {'K':>5} | {'Power MAPE':>12} | {'WL MAPE':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}")
    for K in K_VALUES:
        vals = all_kshot[K]
        if not vals['pw']: continue
        pw_m = np.mean(vals['pw'])
        wl_m = np.mean(vals['wl'])
        marker = ' ←≤10% TARGET' if pw_m <= 10.0 else ''
        print(f"  {K:>5} | {pw_m:>11.1f}% | {wl_m:>11.1f}%{marker}")

    print(f"\n  Placement-level calibration (mean MAPE over {N_REPS} reps × 4 designs):")
    print(f"  {'Kp':>5} | {'Power MAPE':>12} | {'WL MAPE':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}")
    for Kp in K_PLACE:
        vals = all_kplace[Kp]
        if not vals['pw']: continue
        pw_m = np.mean(vals['pw'])
        wl_m = np.mean(vals['wl'])
        marker = ' ←≤10% TARGET' if pw_m <= 10.0 else ''
        print(f"  {Kp:>5} | {pw_m:>11.1f}% | {wl_m:>11.1f}%{marker}")

    print(f"\n{T()} DONE")
    sys.stdout.flush()
