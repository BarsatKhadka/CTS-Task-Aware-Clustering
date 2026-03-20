"""
absolute_v18_glitch.py — T1-A: Glitch-Aware Activity Correction

Hypothesis: SHA256 rel_act=0.104 is 2x OOD. Correct with:
  effective_activity = rel_act / (1 + coef * (comb_per_ff - 1))
coef=0.3 pulls SHA256 eff_act=[0.044,0.055] vs training range [0.011,0.041].

T1-B also included: wire capacitance per placement from DEF net HPWLs.
  estimated_wire_cap = sum_nets(HPWL_net) * 0.2e-15  (sky130 M2: ~0.2 fF/um)
  Add log(estimated_wire_cap) as power feature.
"""

import os, sys, time
import numpy as np
import pandas as pd
import pickle
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


def build_features(df_in, dc, sc_cache, tc, gc, ec, glitch_coef=0.3, use_wirecap=True):
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

        # T1-A: Glitch-corrected activity
        eff_act = rel_act / max(1.0 + glitch_coef * (comb_per_ff - 1), 0.1)

        # T1-B: Wire capacitance from DEF net HPWLs
        net_hpwl_sum = df_f.get('net_hpwl_sum', ff_hpwl * n_ff)  # fallback
        wire_cap_est = net_hpwl_sum * 0.2e-15  # sky130: 0.2 fF/um
        log_wire_cap = np.log1p(wire_cap_est * 1e15)  # in fF

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        sm = tf['slack_mean']
        fn = tf['frac_neg']
        ft = tf['frac_tight']

        # POWER BASE: 58 dims v11-exact but WITH glitch correction
        # Replace rel_act with eff_act in all positions
        base_pw = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            eff_act,  # <-- GLITCH CORRECTED
            sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, sd, sl, sa, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
            frac_xor * comb_per_ff,
            eff_act * frac_xor,   # <-- GLITCH CORRECTED interaction
            eff_act * (1 - df_f['frac_ff_active']),  # <-- GLITCH CORRECTED
            sd * avg_ds, sa * f_ghz,
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * eff_act * f_ghz),  # <-- GLITCH CORRECTED
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
        ]  # 58 dims

        # Also keep raw rel_act as extra feature so model can decide
        extra_pw = [rel_act, np.log1p(rel_act), glitch_coef * comb_per_ff]  # 3 dims
        if use_wirecap:
            extra_pw += [log_wire_cap, np.log1p(net_hpwl_sum)]  # +2 dims

        # WL BASE: 53 dims no synth (unchanged from v16_final)
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
            gf.get('grav_abs_mean', 0.0), gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0), gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0), gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0), gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd,
            gf.get('grav_abs_mean', 0.0) * mw,
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

        rows_pw.append(base_pw + extra_pw + timing)
        rows_wl.append(base + gravity + extra_scale)

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


def lodo_power(X_pw, y_pw, meta_df):
    designs = sorted(meta_df['design_name'].unique())
    res = {}
    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held
        sc = StandardScaler()
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0, n_jobs=1)
        m.fit(sc.fit_transform(X_pw[tr]), y_pw[tr])
        pred = np.exp(m.predict(sc.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
        res[held] = mape(meta_df[te]['power_total'].values, pred)
    return res


if __name__ == '__main__':
    print("=" * 70)
    print("v18: Glitch Correction (T1-A) + Wire Cap Feature (T1-B)")
    print("=" * 70)
    sys.stdout.flush()

    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE, 'rb') as f:    ec = pickle.load(f)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"{T()} n={len(df)}")
    sys.stdout.flush()

    # Check if DEF cache has net_hpwl_sum
    sample_key = list(dc.keys())[0]
    has_net_hpwl = 'net_hpwl_sum' in dc[sample_key]
    print(f"  DEF cache has net_hpwl_sum: {has_net_hpwl}")
    if has_net_hpwl:
        vals = [dc[k]['net_hpwl_sum'] for k in list(dc.keys())[:10]]
        print(f"  net_hpwl_sum sample: {[f'{v:.0f}' for v in vals]}")
    sys.stdout.flush()

    print(f"\n{T()} === Baseline (v16_final params, no glitch correction) ===")
    X_pw0, X_wl0, y_pw0, y_wl0, meta0 = build_features(
        df, dc, sc_cache, tc, gc, ec, glitch_coef=0.0, use_wirecap=False)
    r0 = lodo_power(X_pw0, y_pw0, meta0)
    print(f"  {' '.join(f'{d}={v:.1f}%' for d,v in r0.items())}  mean={np.mean(list(r0.values())):.1f}%")
    sys.stdout.flush()

    # Sweep glitch coefficients
    print(f"\n{T()} === Glitch Correction Sweep (no wire cap) ===")
    for coef in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        X_pw, X_wl, y_pw, y_wl, meta_df = build_features(
            df, dc, sc_cache, tc, gc, ec, glitch_coef=coef, use_wirecap=False)
        res = lodo_power(X_pw, y_pw, meta_df)
        vals = list(res.values())
        marker = ' ← BEST' if np.mean(vals) < 30.0 else ''
        print(f"  coef={coef}: {' '.join(f'{d}={v:.1f}%' for d,v in res.items())}  "
              f"mean={np.mean(vals):.1f}%{marker}")
        sys.stdout.flush()

    # Best coef + wire cap
    if has_net_hpwl:
        print(f"\n{T()} === Best Glitch Coef + Wire Cap Feature ===")
        for coef in [0.2, 0.3, 0.4]:
            X_pw, X_wl, y_pw, y_wl, meta_df = build_features(
                df, dc, sc_cache, tc, gc, ec, glitch_coef=coef, use_wirecap=True)
            res = lodo_power(X_pw, y_pw, meta_df)
            vals = list(res.values())
            print(f"  coef={coef}+wirecap: {' '.join(f'{d}={v:.1f}%' for d,v in res.items())}  "
                  f"mean={np.mean(vals):.1f}%")
            sys.stdout.flush()
    else:
        print(f"\n  NOTE: net_hpwl_sum not in DEF cache — need to reparse DEF files")
        print(f"  Will compute on-the-fly from placement files...")
        # Could parse DEF files here, but that's a separate task

    print(f"\n{T()} DONE")
