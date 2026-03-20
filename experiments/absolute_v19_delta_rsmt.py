"""
absolute_v19_delta_rsmt.py — T1-C + T2-E

T1-C: Power delta decomposition
  P_wire_est = 0.5 × net_hpwl_sum × C_per_um × V_dd² × f × rel_act
  Train on log(P_total - P_wire_est) to isolate CTS-dependent power.
  Also test: predict P_total normally but add log(wire_cap) + rsmt features.

T2-E: RSMT-normalized WL
  Instead of sqrt(n_ff × die_area) normalization, use rsmt_total as baseline.
  Train on log(actual_wl / rsmt_total) — should be ~constant ~1.1-1.5.

T2-F: Add RUDY congestion features to WL.
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
GRAVITY_CACHE = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE    = f'{BASE}/absolute_v13_extended_cache.pkl'
NET_CACHE    = f'{BASE}/net_features_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}
VDD = 1.8  # sky130 nominal


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    sd = 1.0 if 'DELAY' in s else 0.0
    try: lv = float(s.split()[-1])
    except: lv = 2.0
    return sd, lv, sd * lv / 4.0


def build_features(df_in, dc, sc_cache, tc, gc, ec, nc, mode='augment'):
    """
    mode='augment': v16_final features + net_cache features (wire_cap, rsmt, RUDY)
    mode='delta':   predict log(P_delta) where P_delta = P_total - P_wire_est
    mode='rsmt_wl': WL normalized by rsmt_total instead of sqrt(n_ff*die_area)
    """
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
        nf = nc.get(pid, {})

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

        # Net-level features
        net_hpwl_sum = nf.get('net_hpwl_sum', ff_hpwl * n_ff)
        rsmt_total = nf.get('rsmt_total', net_hpwl_sum * 1.1)
        wire_cap_total = nf.get('wire_cap_total', net_hpwl_sum * 0.2e-15)
        log_wire_cap = nf.get('log_wire_cap', np.log1p(wire_cap_total * 1e15))
        rudy_mean = nf.get('rudy_mean', 50.0)
        rudy_max = nf.get('rudy_max', 100.0)
        rudy_cv = nf.get('rudy_cv', 0.5)
        net_deg_mean = nf.get('net_degree_mean', 4.0)
        frac_hf = nf.get('frac_high_fanout', 0.05)

        # Power wire estimate: 0.5 * C_wire * V^2 * f * alpha
        p_wire_est = 0.5 * wire_cap_total * (VDD ** 2) * (f_ghz * 1e9) * rel_act

        # Normalizers (v11 proven for base)
        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)
        rsmt_norm = max(rsmt_total, 1e3)  # RSMT-based WL norm

        sm = tf['slack_mean']
        fn = tf['frac_neg']
        ft = tf['frac_tight']

        # BASE POWER FEATURES (v16_final exact, 58 dims WITH synth)
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

        # NET/WIRE FEATURES (10 dims — T1-B + T1-C + T2-F)
        net_feats = [
            log_wire_cap,                           # log estimated wire cap (fF)
            np.log1p(net_hpwl_sum),                 # log total net HPWL
            np.log1p(rsmt_total),                   # log RSMT estimate
            np.log1p(p_wire_est * 1e6),             # log P_wire_est (µW)
            np.log1p(wire_cap_total / (n_ff + 1) * 1e15),  # wire cap per FF (fF)
            np.log1p(net_hpwl_sum / (n_ff + 1)),   # net HPWL per FF
            net_deg_mean, frac_hf,                  # fanout stats
            rudy_mean / 100.0, rudy_cv,             # RUDY congestion
        ]  # 10 dims

        timing = [
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
            fn, ft, tf['frac_critical'],
            tf['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            float(sm > 1.5), float(sm > 2.0), float(sm > 3.0),
            np.log1p(sm), sm * f_ghz,
        ]  # 18 dims

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
            np.log1p(die_area / (n_ff + 1)),
            np.log1p(n_comb),
            comb_per_ff * np.log1p(n_ff),
        ]  # 3 dims

        # WL net features (for RUDY + RSMT)
        wl_net_feats = [
            np.log1p(rsmt_total), np.log1p(rsmt_total / (n_ff + 1)),
            rudy_mean / 100.0, rudy_max / 100.0, rudy_cv,
            np.log1p(net_hpwl_sum), net_deg_mean, frac_hf,
        ]  # 8 dims

        # Power row: base_pw(58) + net_feats(10) + timing(18) = 86 dims
        rows_pw.append(base_pw + net_feats + timing)

        # WL row: base_wl(53) + gravity(19) + extra_scale(3) + wl_net_feats(8) = 83 dims
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
        rows_wl.append(base_wl + gravity + extra_scale + wl_net_feats)

        # Targets
        if mode == 'delta' and p_wire_est > 0 and pw - p_wire_est > 0:
            y_pw.append(np.log((pw - p_wire_est) / pw_norm))
            meta.append({'placement_id': pid, 'design_name': design,
                         'power_total': pw, 'wirelength': wl,
                         'pw_norm': pw_norm, 'wl_norm': wl_norm,
                         'p_wire_est': p_wire_est,
                         'rsmt_norm': rsmt_norm})
        else:
            y_pw.append(np.log(pw / pw_norm))
            meta.append({'placement_id': pid, 'design_name': design,
                         'power_total': pw, 'wirelength': wl,
                         'pw_norm': pw_norm, 'wl_norm': wl_norm,
                         'p_wire_est': p_wire_est,
                         'rsmt_norm': rsmt_norm})

        if mode == 'rsmt_wl':
            y_wl.append(np.log(wl / rsmt_norm))
        else:
            y_wl.append(np.log(wl / wl_norm))

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


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3, mode='augment'):
    designs = sorted(meta_df['design_name'].unique())
    results = {}

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        # Power
        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_log_pw = m_pw.predict(sc_pw.transform(X_pw[te]))

        if mode == 'delta':
            # Reconstruct: P_total = P_delta + P_wire_est
            p_wire = meta_df[te]['p_wire_est'].values
            pred_pw = np.exp(pred_log_pw) * meta_df[te]['pw_norm'].values + p_wire
        else:
            pred_pw = np.exp(pred_log_pw) * meta_df[te]['pw_norm'].values

        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        # WL
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

        if mode == 'rsmt_wl':
            pred_wl = np.exp(pred_log_wl) * meta_df[te]['rsmt_norm'].values
        else:
            pred_wl = np.exp(pred_log_wl) * meta_df[te]['wl_norm'].values

        mwl = mape(meta_df[te]['wirelength'].values, pred_wl)
        results[held] = {'power': mpw, 'wl': mwl}

    mean_pw = np.mean([v['power'] for v in results.values()])
    mean_wl = np.mean([v['wl'] for v in results.values()])
    return mean_pw, mean_wl, results


if __name__ == '__main__':
    print("=" * 70)
    print("v19: Power Delta (T1-C) + RSMT WL (T2-E) + RUDY (T2-F)")
    print("=" * 70)
    sys.stdout.flush()

    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE, 'rb') as f:    ec = pickle.load(f)
    with open(NET_CACHE, 'rb') as f:    nc = pickle.load(f)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"{T()} n={len(df)}")
    sys.stdout.flush()

    # Check P_wire_est vs actual power
    print(f"\n{T()} === P_wire_est analysis ===")
    for design in ['aes','ethmac','picorv32','sha256']:
        rows = df[df['design_name']==design]
        pw_est_list, pw_true_list = [], []
        for _,row in rows.iterrows():
            pid=row['placement_id']; df_f=dc.get(pid); sf=sc_cache.get(pid)
            nf=nc.get(pid,{})
            if not df_f or not sf: continue
            f_ghz=1/{'aes':7,'picorv32':5,'sha256':9,'ethmac':9}[design]
            wct=nf.get('wire_cap_total', df_f['ff_hpwl']*df_f['n_ff']*0.2e-15)
            p_est=0.5*wct*(VDD**2)*(f_ghz*1e9)*sf['rel_act']
            pw_est_list.append(p_est); pw_true_list.append(row['power_total'])
        ratio = np.array(pw_est_list) / (np.array(pw_true_list)+1e-12)
        print(f"  {design}: P_wire_est/P_total = [{ratio.min():.3f},{ratio.max():.3f}] mean={ratio.mean():.3f}")
    sys.stdout.flush()

    # Mode 1: Augment v16_final with net features (no delta)
    print(f"\n{T()} === Mode 1: Augment v16_final with net features ===")
    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(
        df, dc, sc_cache, tc, gc, ec, nc, mode='augment')
    print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}")
    pw1, wl1, r1 = lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3, mode='augment')
    pw_str1 = '  '.join(f'{d}={v["power"]:.1f}%' for d,v in r1.items())
    wl_str1 = '  '.join(f'{d}={v["wl"]:.1f}%' for d,v in r1.items())
    print(f"  Power: {pw_str1}  mean={pw1:.1f}%")
    print(f"  WL:    {wl_str1}  mean={wl1:.1f}%")
    sys.stdout.flush()

    # Mode 2: Power delta decomposition
    print(f"\n{T()} === Mode 2: Power delta (P_total - P_wire_est) ===")
    X_pw2, X_wl2, y_pw2, y_wl2, meta_df2 = build_features(
        df, dc, sc_cache, tc, gc, ec, nc, mode='delta')
    pw2, wl2, r2 = lodo_eval(X_pw2, X_wl2, y_pw2, y_wl2, meta_df2, wl_alpha=0.3, mode='delta')
    pw_str2 = '  '.join(f'{d}={v["power"]:.1f}%' for d,v in r2.items())
    wl_str2 = '  '.join(f'{d}={v["wl"]:.1f}%' for d,v in r2.items())
    print(f"  Power: {pw_str2}  mean={pw2:.1f}%")
    print(f"  WL:    {wl_str2}  mean={wl2:.1f}%")
    sys.stdout.flush()

    # Mode 3: RSMT-normalized WL
    print(f"\n{T()} === Mode 3: RSMT-normalized WL ===")
    X_pw3, X_wl3, y_pw3, y_wl3, meta_df3 = build_features(
        df, dc, sc_cache, tc, gc, ec, nc, mode='rsmt_wl')
    pw3, wl3, r3 = lodo_eval(X_pw3, X_wl3, y_pw3, y_wl3, meta_df3, wl_alpha=0.3, mode='rsmt_wl')
    pw_str3 = '  '.join(f'{d}={v["power"]:.1f}%' for d,v in r3.items())
    wl_str3 = '  '.join(f'{d}={v["wl"]:.1f}%' for d,v in r3.items())
    print(f"  Power: {pw_str3}  mean={pw3:.1f}%")
    print(f"  WL:    {wl_str3}  mean={wl3:.1f}%")
    sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Baseline (v16_final):     power=32.0%  WL=11.0%")
    print(f"  Mode1 (augment):          power={pw1:.1f}%   WL={wl1:.1f}%")
    print(f"  Mode2 (delta decomp):     power={pw2:.1f}%   WL={wl2:.1f}%")
    print(f"  Mode3 (RSMT WL norm):     power={pw3:.1f}%   WL={wl3:.1f}%")
    print(f"\n{T()} DONE")
